"""Analyze load structure and healthy-training size on LIMAN-C and AMPERE.

This script uses only the compressed FFT feature caches produced by
``run_engine2.py`` and ``run_ampere.py``.  It does not reopen or extract either
raw dataset.  Fault labels are used only for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from condition_reference import robust_spectral_score, roc_auc


@dataclass(frozen=True)
class StudyTable:
    name: str
    identifiers: np.ndarray
    states: np.ndarray
    loads: np.ndarray
    features: np.ndarray
    reference_pool: np.ndarray
    healthy_evaluation: np.ndarray
    fault_evaluation: np.ndarray
    aggregate_blocks: bool


def _shared_floor(reference: np.ndarray) -> np.ndarray:
    center = np.median(reference, axis=0)
    mad = 1.4826 * np.median(np.abs(reference - center), axis=0)
    positive = mad[mad > 1.0e-9]
    scalar = float(np.median(positive)) * 0.05 if positive.size else 1.0e-6
    return np.maximum(0.25 * mad, max(scalar, 1.0e-6))


def _load_ampere(path: Path) -> StudyTable:
    with np.load(path, allow_pickle=False) as payload:
        identifiers = payload["member_paths"]
        states = payload["states"]
        loads = payload["loads"]
        trials = payload["trials"]
        features = payload["features"]
    healthy = states == "normal"
    reference = np.flatnonzero(healthy & (trials % 2 == 1))
    evaluation = np.flatnonzero(healthy & (trials % 2 == 0))
    return StudyTable(
        name="AMPERE",
        identifiers=identifiers,
        states=states,
        loads=loads,
        features=features,
        reference_pool=reference,
        healthy_evaluation=evaluation,
        fault_evaluation=np.flatnonzero(~healthy),
        aggregate_blocks=True,
    )


def _load_limanc(path: Path) -> StudyTable:
    with np.load(path, allow_pickle=False) as payload:
        identifiers = payload["parent_ids"]
        states = payload["states"]
        loads = payload["loads"]
        features = payload["features"]
    healthy = states == "normal"
    reference: list[int] = []
    evaluation: list[int] = []
    for load in sorted(set(int(value) for value in loads[healthy])):
        indices = np.flatnonzero(healthy & (loads == load))
        ordered = indices[np.argsort(identifiers[indices], kind="mergesort")]
        reference.extend(int(value) for value in ordered[::2])
        evaluation.extend(int(value) for value in ordered[1::2])
    return StudyTable(
        name="LIMAN-C",
        identifiers=identifiers,
        states=states,
        loads=loads,
        features=features,
        reference_pool=np.asarray(reference, dtype=np.int64),
        healthy_evaluation=np.asarray(evaluation, dtype=np.int64),
        fault_evaluation=np.flatnonzero(~healthy),
        aggregate_blocks=False,
    )


def _load_structure(table: StudyTable) -> dict[str, object]:
    healthy = np.concatenate([table.reference_pool, table.healthy_evaluation])
    x = table.features[healthy]
    y = table.loads[healthy]
    center = np.median(x, axis=0)
    mad = 1.4826 * np.median(np.abs(x - center), axis=0)
    positive = mad[mad > 1.0e-9]
    scalar = float(np.median(positive)) * 0.05 if positive.size else 1.0e-6
    scale = np.maximum(mad, max(scalar, 1.0e-6))
    normalized = (x - center) / scale
    total = float(np.sum((normalized - np.mean(normalized, axis=0)) ** 2))
    within = 0.0
    for load in sorted(set(int(value) for value in y)):
        rows = normalized[y == load]
        within += float(np.sum((rows - np.mean(rows, axis=0)) ** 2))
    between_fraction = 1.0 - within / total

    reference = table.reference_pool
    floor = _shared_floor(table.features[reference])
    predictions: list[int] = []
    truths: list[int] = []
    loads = sorted(set(int(value) for value in table.loads[reference]))
    for index in table.healthy_evaluation:
        candidates = []
        for load in loads:
            load_reference = reference[table.loads[reference] == load]
            score = robust_spectral_score(
                table.features[load_reference],
                table.features[index : index + 1],
                scale_floor=floor,
            )[0]
            candidates.append((float(score), load))
        predictions.append(min(candidates)[1])
        truths.append(int(table.loads[index]))
    prediction = np.asarray(predictions, dtype=np.int64)
    truth = np.asarray(truths, dtype=np.int64)
    return {
        "healthy_records": int(healthy.size),
        "reference_records": int(reference.size),
        "held_out_healthy_records": int(table.healthy_evaluation.size),
        "between_load_fraction_of_robust_scaled_variance": float(between_fraction),
        "held_out_nearest_load_reference_accuracy": float(np.mean(prediction == truth)),
        "per_load_accuracy": {
            str(load): float(np.mean(prediction[truth == load] == load))
            for load in loads
        },
    }


def _block_auc(
    table: StudyTable,
    query: np.ndarray,
    fixed: np.ndarray,
    matched: np.ndarray,
) -> tuple[float, float]:
    labels = (table.states[query] != "normal").astype(np.int64)
    if not table.aggregate_blocks:
        return roc_auc(labels, fixed), roc_auc(labels, matched)
    groups: dict[tuple[str, int], list[int]] = {}
    for local, index in enumerate(query):
        key = (str(table.states[index]), int(table.loads[index]))
        groups.setdefault(key, []).append(local)
    block_labels: list[int] = []
    block_fixed: list[float] = []
    block_matched: list[float] = []
    for (state, _load), members in sorted(groups.items()):
        block_labels.append(int(state != "normal"))
        block_fixed.append(float(np.median(fixed[members])))
        block_matched.append(float(np.median(matched[members])))
    return (
        roc_auc(np.asarray(block_labels), np.asarray(block_fixed)),
        roc_auc(np.asarray(block_labels), np.asarray(block_matched)),
    )


def _training_size(
    table: StudyTable,
    sizes: list[int],
    *,
    repeats: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_load = {
        load: table.reference_pool[table.loads[table.reference_pool] == load]
        for load in sorted(set(int(value) for value in table.loads[table.reference_pool]))
    }
    if max(sizes) > min(indices.size for indices in by_load.values()):
        raise ValueError(f"Training size exceeds a reference pool in {table.name}")
    query = np.concatenate([table.healthy_evaluation, table.fault_evaluation])
    rng = np.random.default_rng(seed)
    raw: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for size in sizes:
        fixed_values: list[float] = []
        matched_values: list[float] = []
        for repeat in range(repeats):
            chosen_by_load = {
                load: np.sort(rng.choice(indices, size=size, replace=False))
                for load, indices in by_load.items()
            }
            pooled = np.concatenate(list(chosen_by_load.values()))
            floor = _shared_floor(table.features[pooled])
            fixed = robust_spectral_score(
                table.features[pooled], table.features[query], scale_floor=floor
            )
            matched = np.empty(query.size, dtype=np.float64)
            for load, reference in chosen_by_load.items():
                local = np.flatnonzero(table.loads[query] == load)
                matched[local] = robust_spectral_score(
                    table.features[reference],
                    table.features[query[local]],
                    scale_floor=floor,
                )
            fixed_auc, matched_auc = _block_auc(table, query, fixed, matched)
            fixed_values.append(float(fixed_auc))
            matched_values.append(float(matched_auc))
            raw.append(
                {
                    "dataset": table.name,
                    "healthy_references_per_load": size,
                    "total_global_references": size * len(by_load),
                    "repeat": repeat,
                    "fixed_pooled_auc": float(fixed_auc),
                    "load_matched_auc": float(matched_auc),
                    "auc_difference": float(matched_auc - fixed_auc),
                }
            )
        fixed_array = np.asarray(fixed_values)
        matched_array = np.asarray(matched_values)
        difference = matched_array - fixed_array
        row: dict[str, object] = {
            "dataset": table.name,
            "healthy_references_per_load": size,
            "total_global_references": size * len(by_load),
            "repeats": repeats,
        }
        for name, values in (
            ("fixed_pooled", fixed_array),
            ("load_matched", matched_array),
            ("difference", difference),
        ):
            row[f"{name}_median"] = float(np.median(values))
            row[f"{name}_q25"] = float(np.quantile(values, 0.25))
            row[f"{name}_q75"] = float(np.quantile(values, 0.75))
            row[f"{name}_q05"] = float(np.quantile(values, 0.05))
            row[f"{name}_q95"] = float(np.quantile(values, 0.95))
        summary.append(row)
    return summary, raw


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _training_figure(rows: list[dict[str, object]], path: Path) -> None:
    width, height = 1400, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(25, bold=True)
    axis_font = _font(20)
    small_font = _font(17)
    colors = {"fixed_pooled": "#C44E52", "load_matched": "#2C7FB8"}
    for panel, dataset in enumerate(("LIMAN-C", "AMPERE")):
        left = 85 + panel * 690
        top, plot_width, plot_height = 70, 560, 390
        bottom = top + plot_height
        subset = [row for row in rows if row["dataset"] == dataset]
        sizes = [int(row["healthy_references_per_load"]) for row in subset]
        x_min, x_max = min(sizes), max(sizes)
        x = lambda value: left + (value - x_min) / max(x_max - x_min, 1) * plot_width
        y = lambda value: bottom - (value - 0.5) / 0.5 * plot_height
        draw.text((left + plot_width / 2, 22), dataset, fill="black", font=title_font, anchor="ma")
        draw.line((left, top, left, bottom), fill="black", width=2)
        draw.line((left, bottom, left + plot_width, bottom), fill="black", width=2)
        for tick in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            yy = y(tick)
            draw.line((left, yy, left + plot_width, yy), fill="#DDDDDD", width=1)
            draw.text((left - 12, yy), f"{tick:.1f}", fill="black", font=small_font, anchor="rm")
        for size in sizes:
            xx = x(size)
            draw.line((xx, bottom, xx, bottom + 6), fill="black", width=1)
            draw.text((xx, bottom + 10), str(size), fill="black", font=small_font, anchor="ma")
        for method in ("fixed_pooled", "load_matched"):
            points = [(x(int(row["healthy_references_per_load"])), y(float(row[f"{method}_median"]))) for row in subset]
            draw.line(points, fill=colors[method], width=4, joint="curve")
            for xx, yy in points:
                draw.ellipse((xx - 5, yy - 5, xx + 5, yy + 5), fill=colors[method])
        draw.text((left + plot_width / 2, bottom + 45), "Healthy references per load", fill="black", font=axis_font, anchor="ma")
    draw.text((85, 50), "ROC-AUC", fill="black", font=small_font, anchor="ls")
    draw.text((775, 50), "ROC-AUC", fill="black", font=small_font, anchor="ls")
    draw.line((500, 525, 540, 525), fill=colors["fixed_pooled"], width=4)
    draw.text((550, 525), "Fixed pooled", fill="black", font=small_font, anchor="lm")
    draw.line((720, 525, 760, 525), fill=colors["load_matched"], width=4)
    draw.text((770, 525), "Load matched", fill="black", font=small_font, anchor="lm")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(200, 200))


def _structure_figure(structure: dict[str, dict[str, object]], path: Path) -> None:
    width, height = 1000, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(25, bold=True)
    axis_font = _font(20)
    small_font = _font(18)
    left, top, bottom = 105, 70, 450
    plot_width = 820
    y = lambda value: bottom - value * (bottom - top)
    draw.text((width / 2, 24), "Load information in healthy spectra", fill="black", font=title_font, anchor="ma")
    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, left + plot_width, bottom), fill="black", width=2)
    for tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        yy = y(tick)
        draw.line((left, yy, left + plot_width, yy), fill="#DDDDDD", width=1)
        draw.text((left - 12, yy), f"{tick:.1f}", fill="black", font=small_font, anchor="rm")
    colors = {"LIMAN-C": "#4C72B0", "AMPERE": "#55A868"}
    metrics = [
        ("between_load_fraction_of_robust_scaled_variance", "Between-load variance fraction"),
        ("held_out_nearest_load_reference_accuracy", "Held-out load accuracy"),
    ]
    centers = (330, 700)
    for center, (metric, label) in zip(centers, metrics):
        for offset, dataset in ((-70, "LIMAN-C"), (20, "AMPERE")):
            value = float(structure[dataset][metric])
            x0 = center + offset
            draw.rectangle((x0, y(value), x0 + 50, bottom), fill=colors[dataset])
            draw.text((x0 + 25, y(value) - 8), f"{value:.2f}", fill="black", font=small_font, anchor="ms")
        draw.text((center, bottom + 22), label, fill="black", font=small_font, anchor="ma")
    draw.rectangle((360, 515, 382, 537), fill=colors["LIMAN-C"])
    draw.text((392, 526), "LIMAN-C", fill="black", font=small_font, anchor="lm")
    draw.rectangle((560, 515, 582, 537), fill=colors["AMPERE"])
    draw.text((592, 526), "AMPERE", fill="black", font=small_font, anchor="lm")
    draw.text((left, 54), "Metric value", fill="black", font=small_font, anchor="ls")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(200, 200))


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limanc-cache", type=Path, required=True)
    parser.add_argument("--ampere-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/two_dataset"))
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    limanc = _load_limanc(args.limanc_cache.resolve())
    ampere = _load_ampere(args.ampere_cache.resolve())
    structures = {
        limanc.name: _load_structure(limanc),
        ampere.name: _load_structure(ampere),
    }
    limanc_summary, limanc_raw = _training_size(
        limanc, [2, 4, 6, 8, 10], repeats=args.repeats, seed=args.seed
    )
    ampere_summary, ampere_raw = _training_size(
        ampere, [2, 4, 6, 8], repeats=args.repeats, seed=args.seed + 1
    )
    summary = limanc_summary + ampere_summary
    raw = limanc_raw + ampere_raw
    result = {
        "study": {
            "datasets": ["LIMAN-C", "AMPERE rotor branch"],
            "feature_input": "cached 5-1000 Hz three-phase log-power FFT vectors",
            "fault_labels_used_for_fitting": False,
            "training_size_design": (
                "fixed held-out healthy evaluation set; equal k references selected "
                "per load; fixed and matched rules use the same selected records"
            ),
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "healthy_load_structure": structures,
        "training_size_summary": summary,
    }
    (output / "two_dataset_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(summary, output / "training_size_summary.csv")
    _write_csv(raw, output / "training_size_repeats.csv")
    _training_figure(summary, output / "training_size_auc.png")
    _structure_figure(structures, output / "healthy_load_structure.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
