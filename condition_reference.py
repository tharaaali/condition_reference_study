"""Condition-matched versus fixed healthy spectral references.

The module deliberately implements a small, transparent one-class experiment:
one-second Hann-window FFTs, a robust diagonal distance, and parent-level
evaluation.  No fault labels are used to fit a reference or select features.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)

STATE_BY_PREFIX = {
    "1st": "rbd",
    "2nd": "normal",
    "3rd": "bearing",
    "4th": "itsc",
}


@dataclass(frozen=True)
class ParentRecord:
    """One independent three-phase acquisition."""

    parent_id: str
    state: str
    load_setting: int
    phase_paths: tuple[Path, Path, Path]


@dataclass(frozen=True)
class FeatureTable:
    """Parent-level spectral features and metadata."""

    parent_ids: np.ndarray
    states: np.ndarray
    loads: np.ndarray
    frequencies_hz: np.ndarray
    features: np.ndarray

    @property
    def labels(self) -> np.ndarray:
        """Return zero for healthy and one for anomalous parents."""

        return (self.states != "normal").astype(np.int64)


def discover_engine2_records(root: Path) -> list[ParentRecord]:
    """Discover complete three-phase LIMAN-C current acquisitions.

    Parameters
    ----------
    root:
        Dataset root containing ``experiment_1/current``.

    Returns
    -------
    list[ParentRecord]
        Deterministically ordered complete three-phase parents.
    """

    current_root = Path(root) / "experiment_1" / "current"
    if not current_root.is_dir():
        raise FileNotFoundError(f"Engine2 current directory not found: {current_root}")
    records: list[ParentRecord] = []
    for condition_dir in sorted(current_root.iterdir(), key=lambda path: path.name):
        if not condition_dir.is_dir() or "_load_" not in condition_dir.name:
            continue
        prefix, load_text = condition_dir.name.split("_load_", maxsplit=1)
        if prefix not in STATE_BY_PREFIX:
            continue
        phase_files = []
        for phase in (1, 2, 3):
            phase_dir = condition_dir / str(phase)
            if not phase_dir.is_dir():
                phase_files.append({})
                continue
            phase_files.append({path.stem: path for path in phase_dir.glob("*.csv")})
        shared = set(phase_files[0]) & set(phase_files[1]) & set(phase_files[2])
        for parent_id in sorted(shared):
            records.append(
                ParentRecord(
                    parent_id=parent_id,
                    state=STATE_BY_PREFIX[prefix],
                    load_setting=int(load_text),
                    phase_paths=tuple(
                        phase_files[index][parent_id] for index in range(3)
                    ),
                )
            )
    if not records:
        raise ValueError(f"No complete Engine2 records found under {current_root}")
    return records


def read_engine2_parent(record: ParentRecord) -> tuple[np.ndarray, np.ndarray]:
    """Load one parent and apply the dataset's locked phase repair.

    The source time coordinate is milliseconds.  Raw channel 3 is advanced by
    one sample, then channels are mapped as A=1, B=3, C=2, matching the locked
    dataset receipt.
    """

    times: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for path in record.phase_paths:
        data = np.genfromtxt(
            path,
            delimiter=",",
            skip_header=1,
            usecols=(1, 2),
            dtype=np.float64,
            invalid_raise=False,
        )
        if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] != 2:
            raise ValueError(f"Unexpected Engine2 CSV layout: {path}")
        times.append(data[:, 0])
        values.append(data[:, 1])
    sample_count = min(value.size for value in values)
    time_s = times[0][:sample_count]
    if float(np.nanmedian(np.diff(time_s))) > 0.01:
        time_s = time_s / 1000.0
    raw = np.column_stack([value[:sample_count] for value in values])
    raw = _interpolate_nonfinite(raw)
    # Locked receipt: raw channel 3 is shifted +1 sample before permutation.
    raw = raw[:-1].copy()
    raw[:, 2] = _interpolate_nonfinite(
        values[2][1:sample_count, np.newaxis]
    )[:, 0]
    currents_abc = raw[:, [0, 2, 1]]
    return time_s[:-1], currents_abc


def spectral_feature(
    time_s: np.ndarray,
    currents_abc: np.ndarray,
    *,
    window_seconds: float = 1.0,
    minimum_hz: float = 5.0,
    maximum_hz: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one median log-power FFT vector for a parent acquisition."""

    dt = float(np.median(np.diff(time_s)))
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("Non-positive Engine2 sample interval")
    sample_rate_hz = 1.0 / dt
    window_samples = int(round(window_seconds * sample_rate_hz))
    if window_samples < 32 or currents_abc.shape[0] < window_samples:
        raise ValueError("Acquisition is too short for the configured FFT window")
    frequency = np.fft.rfftfreq(window_samples, d=dt)
    keep = (frequency >= minimum_hz) & (frequency <= maximum_hz)
    taper = np.hanning(window_samples)[:, np.newaxis]
    window_features: list[np.ndarray] = []
    for start in range(0, currents_abc.shape[0] - window_samples + 1, window_samples):
        segment = currents_abc[start : start + window_samples]
        segment = segment - np.mean(segment, axis=0, keepdims=True)
        spectrum = np.fft.rfft(segment * taper, axis=0)
        power = np.mean(np.abs(spectrum) ** 2, axis=1)
        window_features.append(np.log10(np.maximum(power[keep], 1.0e-18)))
    if not window_features:
        raise ValueError("No complete FFT windows were produced")
    return frequency[keep], np.median(np.vstack(window_features), axis=0)


def extract_feature_table(
    records: Sequence[ParentRecord],
    *,
    maximum_hz: float = 1000.0,
) -> FeatureTable:
    """Materialize parent-level FFT features for all source records."""

    feature_rows: list[np.ndarray] = []
    frequency: np.ndarray | None = None
    for index, record in enumerate(records, start=1):
        time_s, currents = read_engine2_parent(record)
        current_frequency, feature = spectral_feature(
            time_s, currents, maximum_hz=maximum_hz
        )
        if frequency is None:
            frequency = current_frequency
        elif frequency.shape != current_frequency.shape or not np.allclose(
            frequency, current_frequency, atol=1.0e-6
        ):
            raise ValueError("FFT frequency grids differ between parents")
        feature_rows.append(feature)
        if index == 1 or index % 50 == 0 or index == len(records):
            LOGGER.info("extracted_fft_features parents=%d total=%d", index, len(records))
    assert frequency is not None
    return FeatureTable(
        parent_ids=np.asarray([record.parent_id for record in records], dtype=str),
        states=np.asarray([record.state for record in records], dtype=str),
        loads=np.asarray([record.load_setting for record in records], dtype=np.int64),
        frequencies_hz=frequency,
        features=np.vstack(feature_rows),
    )


def save_feature_table(table: FeatureTable, path: Path) -> None:
    """Save a compressed feature table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        parent_ids=table.parent_ids,
        states=table.states,
        loads=table.loads,
        frequencies_hz=table.frequencies_hz,
        features=table.features,
    )


def load_feature_table(path: Path) -> FeatureTable:
    """Load a feature table created by :func:`save_feature_table`."""

    with np.load(path, allow_pickle=False) as payload:
        return FeatureTable(
            parent_ids=payload["parent_ids"],
            states=payload["states"],
            loads=payload["loads"],
            frequencies_hz=payload["frequencies_hz"],
            features=payload["features"],
        )


@dataclass(frozen=True)
class HealthyReferenceDetector:
    """Fitted robust one-class detector for spectral feature vectors.

    Use :meth:`fit` with healthy feature rows, then call :meth:`score` on new
    rows. Larger scores mean greater departure from the healthy reference.
    """

    center: np.ndarray
    scale: np.ndarray
    squared_residual_clip: float = 100.0

    @classmethod
    def fit(
        cls,
        reference: np.ndarray,
        *,
        scale_floor: np.ndarray | None = None,
    ) -> "HealthyReferenceDetector":
        reference = np.asarray(reference, dtype=np.float64)
        if reference.ndim != 2 or reference.shape[0] < 2:
            raise ValueError("Reference features must contain at least two rows")
        if not np.isfinite(reference).all():
            raise ValueError("Reference features contain a nonfinite value")
        center = np.median(reference, axis=0)
        mad = 1.4826 * np.median(np.abs(reference - center), axis=0)
        if scale_floor is None:
            positive = mad[mad > 1.0e-9]
            scalar_floor = (
                float(np.median(positive)) * 0.05 if positive.size else 1.0e-6
            )
            scale = np.maximum(mad, max(scalar_floor, 1.0e-6))
        else:
            floor = np.asarray(scale_floor, dtype=np.float64)
            if floor.shape != center.shape or not np.isfinite(floor).all():
                raise ValueError("Scale floor must be finite and match feature width")
            scale = np.maximum(mad, floor)
        return cls(center=center, scale=scale)

    def score(self, query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != self.center.size:
            raise ValueError("Query features must be a matrix of matching width")
        if not np.isfinite(query).all():
            raise ValueError("Query features contain a nonfinite value")
        residual = (query - self.center) / self.scale
        return np.sqrt(
            np.mean(
                np.minimum(
                    residual * residual,
                    self.squared_residual_clip,
                ),
                axis=1,
            )
        )


def robust_spectral_score(
    reference: np.ndarray,
    query: np.ndarray,
    *,
    scale_floor: np.ndarray | None = None,
) -> np.ndarray:
    """Score query spectra with a robust diagonal RMS distance."""

    return HealthyReferenceDetector.fit(
        reference,
        scale_floor=scale_floor,
    ).score(query)


def score_reference_strategies(table: FeatureTable) -> dict[str, np.ndarray]:
    """Score fixed, condition-matched, and zero-load healthy references.

    Healthy parents are scored leave-one-parent-out.  Fault parents are scored
    against all eligible healthy parents.  The scale floor is frozen from the
    pooled healthy set and shared by all strategies.
    """

    healthy = table.labels == 0
    healthy_indices = np.flatnonzero(healthy)
    pooled = table.features[healthy]
    pooled_center = np.median(pooled, axis=0)
    pooled_mad = 1.4826 * np.median(np.abs(pooled - pooled_center), axis=0)
    positive = pooled_mad[pooled_mad > 1.0e-9]
    scalar_floor = float(np.median(positive)) * 0.05 if positive.size else 1.0e-6
    shared_floor = np.maximum(0.25 * pooled_mad, max(scalar_floor, 1.0e-6))
    scores = {
        "fixed_pooled": np.empty(table.features.shape[0], dtype=np.float64),
        "condition_matched": np.empty(table.features.shape[0], dtype=np.float64),
        "fixed_zero_load": np.empty(table.features.shape[0], dtype=np.float64),
    }
    for index in range(table.features.shape[0]):
        exclusion = healthy_indices != index
        eligible_pooled = healthy_indices[exclusion]
        eligible_condition = eligible_pooled[
            table.loads[eligible_pooled] == table.loads[index]
        ]
        eligible_zero = eligible_pooled[table.loads[eligible_pooled] == 0]
        if eligible_condition.size < 3:
            raise ValueError(
                f"Fewer than three healthy references at load {table.loads[index]}"
            )
        if eligible_zero.size < 3:
            raise ValueError("Fewer than three zero-load healthy references")
        query = table.features[index : index + 1]
        scores["fixed_pooled"][index] = robust_spectral_score(
            table.features[eligible_pooled], query, scale_floor=shared_floor
        )[0]
        scores["condition_matched"][index] = robust_spectral_score(
            table.features[eligible_condition], query, scale_floor=shared_floor
        )[0]
        scores["fixed_zero_load"][index] = robust_spectral_score(
            table.features[eligible_zero], query, scale_floor=shared_floor
        )[0]
    return scores


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute binary ROC-AUC with average ranks for ties."""

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels == 1
    negative = labels == 0
    n_positive = int(np.sum(positive))
    n_negative = int(np.sum(negative))
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = float(np.sum(ranks[positive]))
    return (
        positive_rank_sum - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)


def evaluate(
    table: FeatureTable,
    scores: dict[str, np.ndarray],
    *,
    bootstrap_replicates: int = 2000,
    permutation_replicates: int = 5000,
    seed: int = 20260925,
) -> dict[str, object]:
    """Evaluate parent-level discrimination and the paired AUC difference."""

    labels = table.labels
    healthy = labels == 0
    methods: dict[str, object] = {}
    for method, values in scores.items():
        threshold = float(np.quantile(values[healthy], 0.95, method="higher"))
        predicted = values >= threshold
        methods[method] = {
            "roc_auc": roc_auc(labels, values),
            "threshold_healthy_95th_percentile": threshold,
            "healthy_false_positive_rate": float(np.mean(predicted[healthy])),
            "fault_true_positive_rate": float(np.mean(predicted[~healthy])),
            "score_median_healthy": float(np.median(values[healthy])),
            "score_median_fault": float(np.median(values[~healthy])),
        }
    stratified = _paired_stratified_bootstrap(
        table,
        scores,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    delta = stratified["condition_matched_minus_fixed_pooled"]
    methods["condition_matched"]["roc_auc_interval_95"] = [
        float(np.quantile(stratified["condition_matched"], 0.025)),
        float(np.quantile(stratified["condition_matched"], 0.975)),
    ]
    methods["fixed_pooled"]["roc_auc_interval_95"] = [
        float(np.quantile(stratified["fixed_pooled"], 0.025)),
        float(np.quantile(stratified["fixed_pooled"], 0.975)),
    ]
    comparison = {
        "auc_difference": float(
            methods["condition_matched"]["roc_auc"]
            - methods["fixed_pooled"]["roc_auc"]
        ),
        "bootstrap_interval_95": [
            float(np.quantile(delta, 0.025)),
            float(np.quantile(delta, 0.975)),
        ],
        "paired_permutation_p_two_sided": _paired_permutation_p_value(
            labels,
            scores["condition_matched"],
            scores["fixed_pooled"],
            replicates=permutation_replicates,
            seed=seed + 1,
        ),
    }
    per_load: list[dict[str, object]] = []
    for load in sorted(set(int(value) for value in table.loads)):
        mask = table.loads == load
        row: dict[str, object] = {
            "load_setting": load,
            "healthy_parents": int(np.sum(mask & healthy)),
            "fault_parents": int(np.sum(mask & ~healthy)),
        }
        for method, values in scores.items():
            row[f"{method}_roc_auc"] = roc_auc(labels[mask], values[mask])
            row[f"{method}_healthy_score_median"] = float(
                np.median(values[mask & healthy])
            )
            threshold = float(
                methods[method]["threshold_healthy_95th_percentile"]
            )
            row[f"{method}_healthy_false_positive_rate"] = float(
                np.mean(values[mask & healthy] >= threshold)
            )
            row[f"{method}_fault_true_positive_rate"] = float(
                np.mean(values[mask & ~healthy] >= threshold)
            )
        per_load.append(row)
    per_fault: list[dict[str, object]] = []
    for state in sorted(set(table.states) - {"normal"}):
        mask = (table.states == "normal") | (table.states == state)
        row = {"fault_family": state, "fault_parents": int(np.sum(table.states == state))}
        for method, values in scores.items():
            row[f"{method}_roc_auc"] = roc_auc(labels[mask], values[mask])
        per_fault.append(row)
    frequency_sensitivity: list[dict[str, float]] = []
    for maximum_hz in (250.0, 500.0, 1000.0):
        keep = table.frequencies_hz <= maximum_hz
        band_table = FeatureTable(
            parent_ids=table.parent_ids,
            states=table.states,
            loads=table.loads,
            frequencies_hz=table.frequencies_hz[keep],
            features=table.features[:, keep],
        )
        band_scores = (
            scores
            if maximum_hz == 1000.0
            else score_reference_strategies(band_table)
        )
        fixed_auc = roc_auc(labels, band_scores["fixed_pooled"])
        conditional_auc = roc_auc(labels, band_scores["condition_matched"])
        frequency_sensitivity.append(
            {
                "maximum_hz": maximum_hz,
                "fft_features": int(np.sum(keep)),
                "fixed_pooled_roc_auc": fixed_auc,
                "condition_matched_roc_auc": conditional_auc,
                "auc_difference": conditional_auc - fixed_auc,
            }
        )
    return {
        "study": {
            "dataset": "LIMAN-C / engine2",
            "analysis_unit": "complete three-phase acquisition parent",
            "healthy_reference_fit_labels": "healthy only",
            "feature": "median one-second three-phase log-power FFT, 5-1000 Hz",
            "score": "robust diagonal clipped RMS z-distance",
            "primary_comparison": "condition_matched versus fixed_pooled",
            "scope_limitation": (
                "fault state is perfectly confounded with source motor group; "
                "results are a cross-motor/source-group stress test"
            ),
            "seed": seed,
        },
        "counts": {
            "parents": int(table.features.shape[0]),
            "healthy_parents": int(np.sum(healthy)),
            "fault_parents": int(np.sum(~healthy)),
            "fft_features": int(table.features.shape[1]),
        },
        "methods": methods,
        "primary_comparison": comparison,
        "per_load": per_load,
        "per_fault_family": per_fault,
        "frequency_band_sensitivity": frequency_sensitivity,
        "healthy_condition_structure": _healthy_condition_structure(table),
    }


def write_parent_scores(
    table: FeatureTable, scores: dict[str, np.ndarray], path: Path
) -> None:
    """Write auditable parent-level scores."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["parent_id", "state", "label", "load_setting", *scores.keys()]
        )
        for index in range(table.features.shape[0]):
            writer.writerow(
                [
                    table.parent_ids[index],
                    table.states[index],
                    int(table.labels[index]),
                    int(table.loads[index]),
                    *(float(values[index]) for values in scores.values()),
                ]
            )


def write_json(payload: dict[str, object], path: Path) -> None:
    """Write stable human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _interpolate_nonfinite(values: np.ndarray) -> np.ndarray:
    repaired = np.asarray(values, dtype=np.float64).copy()
    for column in range(repaired.shape[1]):
        finite = np.isfinite(repaired[:, column])
        if np.sum(finite) < 2:
            raise ValueError("Current channel has fewer than two finite samples")
        if not np.all(finite):
            indices = np.arange(repaired.shape[0])
            repaired[:, column] = np.interp(
                indices, indices[finite], repaired[finite, column]
            )
    return repaired


def _paired_stratified_bootstrap(
    table: FeatureTable,
    scores: dict[str, np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    strata: list[np.ndarray] = []
    for state in sorted(set(str(value) for value in table.states)):
        for load in sorted(set(int(value) for value in table.loads)):
            indices = np.flatnonzero((table.states == state) & (table.loads == load))
            if indices.size:
                strata.append(indices)
    output = {
        "condition_matched": np.empty(replicates),
        "fixed_pooled": np.empty(replicates),
        "condition_matched_minus_fixed_pooled": np.empty(replicates),
    }
    for replicate in range(replicates):
        sampled = np.concatenate(
            [rng.choice(indices, size=indices.size, replace=True) for indices in strata]
        )
        conditional_auc = roc_auc(
            table.labels[sampled], scores["condition_matched"][sampled]
        )
        fixed_auc = roc_auc(table.labels[sampled], scores["fixed_pooled"][sampled])
        output["condition_matched"][replicate] = conditional_auc
        output["fixed_pooled"][replicate] = fixed_auc
        output["condition_matched_minus_fixed_pooled"][replicate] = (
            conditional_auc - fixed_auc
        )
    return output


def _paired_permutation_p_value(
    labels: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> float:
    observed = roc_auc(labels, first) - roc_auc(labels, second)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(replicates):
        swap = rng.random(labels.size) < 0.5
        permuted_first = np.where(swap, second, first)
        permuted_second = np.where(swap, first, second)
        permuted = roc_auc(labels, permuted_first) - roc_auc(labels, permuted_second)
        exceedances += abs(permuted) >= abs(observed)
    return float((exceedances + 1) / (replicates + 1))


def _healthy_condition_structure(table: FeatureTable) -> dict[str, object]:
    """Describe load information using healthy parents only."""

    healthy_indices = np.flatnonzero(table.labels == 0)
    healthy_features = table.features[healthy_indices]
    healthy_loads = table.loads[healthy_indices]
    center = np.median(healthy_features, axis=0)
    mad = 1.4826 * np.median(np.abs(healthy_features - center), axis=0)
    positive = mad[mad > 1.0e-9]
    scalar_floor = float(np.median(positive)) * 0.05 if positive.size else 1.0e-6
    scale = np.maximum(mad, max(scalar_floor, 1.0e-6))
    normalized = (healthy_features - center) / scale
    grand_mean = np.mean(normalized, axis=0)
    total_sum_squares = float(np.sum((normalized - grand_mean) ** 2))
    within_sum_squares = 0.0
    load_values = sorted(set(int(value) for value in healthy_loads))
    for load in load_values:
        rows = normalized[healthy_loads == load]
        within_sum_squares += float(np.sum((rows - np.mean(rows, axis=0)) ** 2))
    between_fraction = (
        1.0 - within_sum_squares / total_sum_squares
        if total_sum_squares > 0.0
        else float("nan")
    )
    predictions: list[int] = []
    truths: list[int] = []
    for local_index, global_index in enumerate(healthy_indices):
        candidates: list[tuple[float, int]] = []
        query = table.features[global_index]
        for load in load_values:
            reference_indices = healthy_indices[
                (healthy_loads == load) & (healthy_indices != global_index)
            ]
            load_center = np.median(table.features[reference_indices], axis=0)
            distance = float(np.sqrt(np.mean(((query - load_center) / scale) ** 2)))
            candidates.append((distance, load))
        predictions.append(min(candidates)[1])
        truths.append(int(healthy_loads[local_index]))
    predictions_array = np.asarray(predictions, dtype=np.int64)
    truths_array = np.asarray(truths, dtype=np.int64)
    return {
        "between_load_fraction_of_robust_scaled_variance": between_fraction,
        "leave_one_parent_out_nearest_load_centroid_accuracy": float(
            np.mean(predictions_array == truths_array)
        ),
        "per_load_accuracy": {
            str(load): float(
                np.mean(predictions_array[truths_array == load] == load)
            )
            for load in load_values
        },
    }
