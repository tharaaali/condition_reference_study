"""Stream AMPERE rotor measurements and compare fixed versus load references.
"""

from __future__ import annotations

import argparse
import csv
from io import BytesIO
import itertools
import json
import logging
import re
import struct
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np

from condition_reference import robust_spectral_score, roc_auc


LOGGER = logging.getLogger(__name__)
EXPECTED_ARCHIVE_BYTES = 5_733_250_855
EXPECTED_ARCHIVE_ENTRIES = 3_261
ROTOR_ROOT = "Detection_and_diagnostics_of_rotor_faults"
SAMPLE_RATE_HZ = 20_000.0
WINDOW_SAMPLES = 20_000
FREQUENCIES_HZ = np.fft.rfftfreq(WINDOW_SAMPLES, d=1.0 / SAMPLE_RATE_HZ)[5:1001]
TRIAL_RE = re.compile(r"^(?P<trial>\d{3}).*\.mat$", re.IGNORECASE)

STATE_BY_SOURCE = {
    "Healthy_motor": "normal",
    "One_broken_rotor_bar": "one_broken_rotor_bar",
    "Three_broken_rotor_bars": "three_broken_rotor_bars",
    "Four_broken_rotor_bars": "four_broken_rotor_bars",
    "Bearing_degradation": "bearing_degradation",
}
LOAD_BY_OP = {
    "25hz_0%_1500rpm": 0,
    "25hz_25%_1500rpm": 25,
    "25hz_50%_1500rpm": 50,
    "25hz_75%_1500rpm": 75,
    "25hz_100%_1500rpm": 100,
}


@dataclass(frozen=True)
class Capture:
    member_path: str
    source_condition: str
    state: str
    load_percent: int
    trial: int
    compressed_bytes: int
    uncompressed_bytes: int
    crc32: int


def discover_captures(archive_path: Path) -> list[Capture]:
    """Read only the central directory and select the 400 rotor MAT members."""

    if archive_path.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise ValueError("AMPERE archive byte count differs from the pinned release")
    captures: list[Capture] = []
    with ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if len(infos) != EXPECTED_ARCHIVE_ENTRIES:
            raise ValueError("AMPERE archive entry count differs from the pinned release")
        for info in infos:
            pure = PurePosixPath(info.filename)
            if (
                info.is_dir()
                or len(pure.parts) != 4
                or pure.parts[0] != ROTOR_ROOT
                or pure.suffix.lower() != ".mat"
            ):
                continue
            source_condition, op_code = pure.parts[1:3]
            match = TRIAL_RE.fullmatch(pure.name)
            if (
                source_condition not in STATE_BY_SOURCE
                or op_code not in LOAD_BY_OP
                or match is None
            ):
                raise ValueError(f"Unexpected AMPERE rotor member: {info.filename}")
            captures.append(
                Capture(
                    member_path=info.filename,
                    source_condition=source_condition,
                    state=STATE_BY_SOURCE[source_condition],
                    load_percent=LOAD_BY_OP[op_code],
                    trial=int(match.group("trial")),
                    compressed_bytes=int(info.compress_size),
                    uncompressed_bytes=int(info.file_size),
                    crc32=int(info.CRC),
                )
            )
    captures.sort(
        key=lambda item: (item.source_condition, item.load_percent, item.trial)
    )
    if len(captures) != 400:
        raise ValueError(f"Expected 400 AMPERE rotor MAT members, found {len(captures)}")
    groups: dict[tuple[str, int], list[Capture]] = defaultdict(list)
    for capture in captures:
        groups[(capture.source_condition, capture.load_percent)].append(capture)
    if len(groups) != 25 or any(len(group) != 16 for group in groups.values()):
        raise ValueError("AMPERE rotor condition-load grid is incomplete")
    return captures


def _load_mat_v4_currents(payload: bytes) -> np.ndarray:
    """Read current columns 5--7 from the source's little-endian MAT v4 array."""

    if len(payload) < 24:
        raise ValueError("AMPERE MAT v4 member is truncated")
    mopt, rows, columns, imagf, name_length = struct.unpack_from("<5i", payload, 0)
    if mopt != 10 or columns != 11 or imagf != 0 or not 1 <= name_length <= 256:
        raise ValueError("Unexpected AMPERE MAT v4 header")
    data_offset = 20 + name_length
    expected = data_offset + rows * columns * np.dtype("<f4").itemsize
    if len(payload) != expected:
        raise ValueError("AMPERE MAT v4 payload length differs from its header")
    flat = np.frombuffer(payload, dtype="<f4", count=rows * columns, offset=data_offset)
    frame = flat.reshape((rows, columns), order="F")
    currents = np.asarray(frame[:, 4:7], dtype=np.float64)
    if rows < WINDOW_SAMPLES or not np.isfinite(currents).all():
        raise ValueError("AMPERE current array is short or nonfinite")
    return currents


def _load_csv_currents(payload: bytes) -> np.ndarray:
    """Read only the three current columns from one paired CSV representation."""

    try:
        currents = np.loadtxt(
            BytesIO(payload), delimiter=",", usecols=(4, 5, 6), dtype=np.float64
        )
    except ValueError as exc:
        raise ValueError("AMPERE CSV current columns are not numeric") from exc
    if (
        currents.ndim != 2
        or currents.shape[1] != 3
        or currents.shape[0] < WINDOW_SAMPLES
        or not np.isfinite(currents).all()
    ):
        raise ValueError("AMPERE CSV current array is short or nonfinite")
    return currents


def _spectral_feature(currents: np.ndarray) -> np.ndarray:
    taper = np.hanning(WINDOW_SAMPLES)[:, np.newaxis]
    rows: list[np.ndarray] = []
    for start in range(0, currents.shape[0] - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
        segment = currents[start : start + WINDOW_SAMPLES]
        segment = segment - np.mean(segment, axis=0, keepdims=True)
        spectrum = np.fft.rfft(segment * taper, axis=0)
        power = np.mean(np.abs(spectrum) ** 2, axis=1)
        rows.append(np.log10(np.maximum(power[5:1001], 1.0e-18)))
    if not rows:
        raise ValueError("AMPERE record produced no complete one-second FFT window")
    return np.median(np.vstack(rows), axis=0)


def extract_features(
    archive_path: Path, captures: list[Capture]
) -> tuple[np.ndarray, dict[str, int]]:
    """Stream and release one compressed MAT member at a time."""

    features: list[np.ndarray] = []
    total_compressed = 0
    total_uncompressed = 0
    mat_members_read = 0
    csv_members_read = 0
    with ZipFile(archive_path) as archive:
        for index, capture in enumerate(captures, start=1):
            info = archive.getinfo(capture.member_path)
            if (
                info.CRC != capture.crc32
                or info.file_size != capture.uncompressed_bytes
                or info.compress_size != capture.compressed_bytes
            ):
                raise ValueError("AMPERE member metadata changed during the run")
            with archive.open(info) as stream:
                prefix = stream.read(20)
                if len(prefix) != 20:
                    raise ValueError("AMPERE MAT member is truncated")
                if struct.unpack_from("<i", prefix, 0)[0] == 10:
                    payload = prefix + stream.read()
                    currents = _load_mat_v4_currents(payload)
                    total_compressed += int(info.compress_size)
                    total_uncompressed += int(info.file_size)
                    mat_members_read += 1
                else:
                    # MATLAB v5 occurs for the four-bar group and one three-bar
                    # file.  To keep the runner NumPy-only and memory bounded,
                    # use its paired CSV rather than unpacking the full archive
                    # or adding SciPy solely for 81 members.
                    csv_path = str(PurePosixPath(capture.member_path).with_suffix(".csv"))
                    csv_info = archive.getinfo(csv_path)
                    currents = _load_csv_currents(archive.read(csv_info))
                    total_compressed += int(csv_info.compress_size)
                    total_uncompressed += int(csv_info.file_size)
                    csv_members_read += 1
            features.append(_spectral_feature(currents))
            if index == 1 or index % 25 == 0 or index == len(captures):
                LOGGER.info("processed_rotor_mat count=%d total=%d", index, len(captures))
    return np.vstack(features), {
        "archive_entries": EXPECTED_ARCHIVE_ENTRIES,
        "selected_rotor_measurement_pairs": len(captures),
        "mat_v4_members_read": mat_members_read,
        "csv_fallback_members_read": csv_members_read,
        "selected_representation_compressed_bytes": total_compressed,
        "selected_representation_uncompressed_bytes": total_uncompressed,
        "stator_members_read": 0,
        "files_extracted": 0,
    }


def save_cache(
    path: Path,
    captures: list[Capture],
    features: np.ndarray,
    io_summary: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        member_paths=np.asarray([item.member_path for item in captures], dtype=str),
        source_conditions=np.asarray(
            [item.source_condition for item in captures], dtype=str
        ),
        states=np.asarray([item.state for item in captures], dtype=str),
        loads=np.asarray([item.load_percent for item in captures], dtype=np.int64),
        trials=np.asarray([item.trial for item in captures], dtype=np.int64),
        compressed_bytes=np.asarray(
            [item.compressed_bytes for item in captures], dtype=np.int64
        ),
        uncompressed_bytes=np.asarray(
            [item.uncompressed_bytes for item in captures], dtype=np.int64
        ),
        crc32=np.asarray([item.crc32 for item in captures], dtype=np.int64),
        frequencies_hz=FREQUENCIES_HZ,
        features=features,
        io_summary_json=np.asarray(json.dumps(io_summary)),
    )


def load_cache(path: Path) -> tuple[list[Capture], np.ndarray, dict[str, int]]:
    with np.load(path, allow_pickle=False) as payload:
        captures = [
            Capture(
                member_path=str(member),
                source_condition=str(condition),
                state=str(state),
                load_percent=int(load),
                trial=int(trial),
                compressed_bytes=int(compressed),
                uncompressed_bytes=int(uncompressed),
                crc32=int(crc),
            )
            for member, condition, state, load, trial, compressed, uncompressed, crc in zip(
                payload["member_paths"],
                payload["source_conditions"],
                payload["states"],
                payload["loads"],
                payload["trials"],
                payload["compressed_bytes"],
                payload["uncompressed_bytes"],
                payload["crc32"],
            )
        ]
        return captures, payload["features"], json.loads(str(payload["io_summary_json"]))


def _shared_floor(reference: np.ndarray) -> np.ndarray:
    center = np.median(reference, axis=0)
    mad = 1.4826 * np.median(np.abs(reference - center), axis=0)
    positive = mad[mad > 1.0e-9]
    scalar = float(np.median(positive)) * 0.05 if positive.size else 1.0e-6
    return np.maximum(0.25 * mad, max(scalar, 1.0e-6))


def score_and_aggregate(
    captures: list[Capture], features: np.ndarray
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    healthy_by_load: dict[int, list[int]] = defaultdict(list)
    for index, capture in enumerate(captures):
        if capture.state == "normal":
            healthy_by_load[capture.load_percent].append(index)
    if set(healthy_by_load) != set(LOAD_BY_OP.values()):
        raise ValueError("AMPERE healthy load inventory is incomplete")

    reference_by_load: dict[int, np.ndarray] = {}
    evaluation_healthy: list[int] = []
    for load, indices in sorted(healthy_by_load.items()):
        ordered = sorted(indices, key=lambda value: captures[value].trial)
        reference_by_load[load] = np.asarray(ordered[::2], dtype=np.int64)
        evaluation_healthy.extend(ordered[1::2])
    pooled_reference = np.concatenate(list(reference_by_load.values()))
    query_indices = np.asarray(
        sorted(
            evaluation_healthy
            + [index for index, item in enumerate(captures) if item.state != "normal"]
        ),
        dtype=np.int64,
    )
    floor = _shared_floor(features[pooled_reference])
    fixed = robust_spectral_score(
        features[pooled_reference], features[query_indices], scale_floor=floor
    )
    matched = np.empty(query_indices.size, dtype=np.float64)
    for load, reference in reference_by_load.items():
        local = np.flatnonzero(
            np.asarray([captures[int(index)].load_percent == load for index in query_indices])
        )
        matched[local] = robust_spectral_score(
            features[reference], features[query_indices[local]], scale_floor=floor
        )

    capture_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for local, source_index in enumerate(query_indices):
        capture = captures[int(source_index)]
        grouped[(capture.source_condition, capture.load_percent)].append(local)
        capture_rows.append(
            {
                "member_path": capture.member_path,
                "source_condition": capture.source_condition,
                "state": capture.state,
                "label": int(capture.state != "normal"),
                "load_percent": capture.load_percent,
                "trial": capture.trial,
                "fixed_pooled": float(fixed[local]),
                "load_matched": float(matched[local]),
            }
        )
    block_rows: list[dict[str, object]] = []
    for (condition, load), members in sorted(grouped.items()):
        state = STATE_BY_SOURCE[condition]
        block_rows.append(
            {
                "block_id": f"ampere:rotor:{condition}:load_{load:03d}",
                "source_condition": condition,
                "state": state,
                "label": int(state != "normal"),
                "load_percent": load,
                "query_capture_count": len(members),
                "fixed_pooled": float(np.median(fixed[members])),
                "load_matched": float(np.median(matched[members])),
            }
        )
    if len(block_rows) != 25:
        raise ValueError(f"Expected 25 AMPERE experiment blocks, found {len(block_rows)}")
    split = {
        "healthy_reference_captures": int(pooled_reference.size),
        "healthy_evaluation_captures": len(evaluation_healthy),
        "fault_evaluation_captures": int(sum(item.state != "normal" for item in captures)),
        "reference_captures_per_load": {
            str(load): int(indices.size) for load, indices in reference_by_load.items()
        },
        "split_rule": "within each healthy load, trials 1,3,...,15 are reference and 2,4,...,16 are evaluation",
    }
    return block_rows, capture_rows, split


def _auc(rows: list[dict[str, object]], method: str) -> float:
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    values = np.asarray([row[method] for row in rows], dtype=np.float64)
    return roc_auc(labels, values)


def _bootstrap(
    rows: list[dict[str, object]], *, replicates: int, seed: int
) -> dict[str, np.ndarray]:
    by_load: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_load[int(row["load_percent"])].append(row)
    loads = sorted(by_load)
    rng = np.random.default_rng(seed)
    output = {
        "fixed_pooled": np.empty(replicates),
        "load_matched": np.empty(replicates),
        "load_matched_minus_fixed_pooled": np.empty(replicates),
    }
    for replicate in range(replicates):
        sample = rng.choice(loads, size=len(loads), replace=True)
        selected = [row for load in sample for row in by_load[int(load)]]
        fixed = _auc(selected, "fixed_pooled")
        matched = _auc(selected, "load_matched")
        output["fixed_pooled"][replicate] = fixed
        output["load_matched"][replicate] = matched
        output["load_matched_minus_fixed_pooled"][replicate] = matched - fixed
    return output


def _rank_fraction(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks / values.size


def _exact_load_cluster_permutation(rows: list[dict[str, object]]) -> float:
    """Enumerate all 32 paired swaps of the five complete load clusters."""

    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    loads = np.asarray([row["load_percent"] for row in rows], dtype=np.int64)
    fixed = _rank_fraction(
        np.asarray([row["fixed_pooled"] for row in rows], dtype=np.float64)
    )
    matched = _rank_fraction(
        np.asarray([row["load_matched"] for row in rows], dtype=np.float64)
    )
    observed = roc_auc(labels, matched) - roc_auc(labels, fixed)
    differences = []
    unique_loads = sorted(set(int(value) for value in loads))
    for choices in itertools.product((False, True), repeat=len(unique_loads)):
        swap = np.zeros(labels.size, dtype=bool)
        for load, choice in zip(unique_loads, choices):
            if choice:
                swap |= loads == load
        first = np.where(swap, fixed, matched)
        second = np.where(swap, matched, fixed)
        differences.append(roc_auc(labels, first) - roc_auc(labels, second))
    return float(np.mean(np.abs(differences) >= abs(observed) - 1.0e-15))


def evaluate(
    block_rows: list[dict[str, object]],
    capture_rows: list[dict[str, object]],
    io_summary: dict[str, int],
    split: dict[str, object],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, object]:
    labels = np.asarray([row["label"] for row in block_rows], dtype=np.int64)
    bootstrap = _bootstrap(block_rows, replicates=bootstrap_replicates, seed=seed)
    methods: dict[str, object] = {}
    for method in ("fixed_pooled", "load_matched"):
        values = np.asarray([row[method] for row in block_rows], dtype=np.float64)
        healthy = values[labels == 0]
        threshold = float(np.quantile(healthy, 0.95, method="higher"))
        methods[method] = {
            "block_roc_auc": roc_auc(labels, values),
            "block_roc_auc_load_cluster_bootstrap_interval_95": [
                float(np.quantile(bootstrap[method], 0.025)),
                float(np.quantile(bootstrap[method], 0.975)),
            ],
            "descriptive_healthy_false_positive_rate": float(
                np.mean(healthy >= threshold)
            ),
            "descriptive_fault_true_positive_rate": float(
                np.mean(values[labels == 1] >= threshold)
            ),
            "threshold_healthy_95th_percentile": threshold,
            "optimistic_capture_level_roc_auc": _auc(capture_rows, method),
        }
    per_fault = []
    for state in sorted(set(str(row["state"]) for row in block_rows) - {"normal"}):
        subset = [row for row in block_rows if row["state"] in ("normal", state)]
        per_fault.append(
            {
                "fault_family": state,
                "fixed_pooled_roc_auc": _auc(subset, "fixed_pooled"),
                "load_matched_roc_auc": _auc(subset, "load_matched"),
            }
        )
    return {
        "study": {
            "dataset": "AMPERE rotor branch, release 2023-09-25",
            "analysis_unit": "source condition-by-load experiment block",
            "condition_reference": "declared load only; supply frequency is fixed",
            "feature": "median one-second three-channel log-power FFT, 5-1000 Hz",
            "scope_limitation": "single motor/test bench; independence of 16 numbered files per experiment is unresolved",
            "seed": seed,
        },
        "memory_and_io": io_summary,
        "split": split,
        "counts": {
            "source_rotor_mat_members": 400,
            "source_experiment_blocks": 25,
            "healthy_blocks": 5,
            "fault_blocks": 20,
            "fft_features": int(FREQUENCIES_HZ.size),
        },
        "methods": methods,
        "primary_comparison": {
            "auc_difference": float(
                methods["load_matched"]["block_roc_auc"]
                - methods["fixed_pooled"]["block_roc_auc"]
            ),
            "load_cluster_bootstrap_interval_95": [
                float(
                    np.quantile(
                        bootstrap["load_matched_minus_fixed_pooled"], 0.025
                    )
                ),
                float(
                    np.quantile(
                        bootstrap["load_matched_minus_fixed_pooled"], 0.975
                    )
                ),
            ],
            "exact_five_load_cluster_permutation_p_two_sided": (
                _exact_load_cluster_permutation(block_rows)
            ),
        },
        "per_fault_family": per_fault,
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/ampere"))
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260927)
    return parser


def main() -> int:
    args = _parser().parse_args()
    archive_path = args.archive.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "experiment.log", encoding="utf-8"),
        ],
    )
    cache_path = (
        args.feature_cache.resolve()
        if args.feature_cache is not None
        else output_dir / "ampere_rotor_fft_features.npz"
    )
    started = time.perf_counter()
    if cache_path.exists():
        LOGGER.info("loading_feature_cache path=%s", cache_path)
        captures, features, io_summary = load_cache(cache_path)
    else:
        captures = discover_captures(archive_path)
        LOGGER.info(
            "selected_members count=%d compressed_gib=%.3f uncompressed_gib=%.3f",
            len(captures),
            sum(item.compressed_bytes for item in captures) / 2**30,
            sum(item.uncompressed_bytes for item in captures) / 2**30,
        )
        features, io_summary = extract_features(archive_path, captures)
        save_cache(cache_path, captures, features, io_summary)
        LOGGER.info("saved_feature_cache path=%s", cache_path)
    block_rows, capture_rows, split = score_and_aggregate(captures, features)
    results = evaluate(
        block_rows,
        capture_rows,
        io_summary,
        split,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    write_csv(block_rows, output_dir / "block_scores.csv")
    write_csv(capture_rows, output_dir / "capture_scores_optimistic.csv")
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "completed elapsed_s=%.3f fixed_auc=%.6f load_matched_auc=%.6f delta=%.6f",
        time.perf_counter() - started,
        results["methods"]["fixed_pooled"]["block_roc_auc"],
        results["methods"]["load_matched"]["block_roc_auc"],
        results["primary_comparison"]["auc_difference"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
