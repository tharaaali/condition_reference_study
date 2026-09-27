"""Run the complete LIMAN-C and AMPERE healthy-reference study.

The command creates or reuses FFT feature caches, evaluates the global and
load-matched one-class detectors, and runs the healthy-load and training-size
analyses reported in the paper.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def resolve_limanc_root(path: Path) -> Path:
    """Return the directory that directly contains ``experiment_1``."""

    candidate = path.expanduser().resolve()
    if (candidate / "experiment_1" / "current").is_dir():
        return candidate
    matches = sorted(
        item.parent.parent
        for item in candidate.rglob("current")
        if item.is_dir() and item.parent.name == "experiment_1"
    )
    unique = list(dict.fromkeys(matches))
    if len(unique) == 1:
        return unique[0]
    if not unique:
        raise FileNotFoundError(
            "Could not find LIMAN-C's experiment_1/current directory under "
            f"{candidate}. See the expected layout in README.md."
        )
    raise ValueError(
        "More than one LIMAN-C candidate was found. Pass the directory that "
        f"directly contains experiment_1. Candidates: {unique}"
    )


def validate_ampere_archive(path: Path) -> Path:
    archive = path.expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"AMPERE ZIP not found: {archive}")
    if archive.suffix.lower() != ".zip":
        raise ValueError(f"AMPERE input must be the downloaded ZIP: {archive}")
    return archive


def run_command(arguments: list[str]) -> None:
    printable = subprocess.list2cmdline(arguments)
    print(f"\n> {printable}", flush=True)
    subprocess.run(arguments, cwd=ROOT, check=True)


def print_summary(output: Path, elapsed_seconds: float) -> None:
    limanc = json.loads((output / "limanc" / "results.json").read_text(encoding="utf-8"))
    ampere = json.loads((output / "ampere" / "results.json").read_text(encoding="utf-8"))
    combined = json.loads(
        (output / "two_dataset" / "two_dataset_results.json").read_text(
            encoding="utf-8"
        )
    )
    limanc_fixed = limanc["methods"]["fixed_pooled"]["roc_auc"]
    limanc_matched = limanc["methods"]["condition_matched"]["roc_auc"]
    ampere_fixed = ampere["methods"]["fixed_pooled"]["block_roc_auc"]
    ampere_matched = ampere["methods"]["load_matched"]["block_roc_auc"]
    structure = combined["healthy_load_structure"]
    print("\nStudy complete")
    print("--------------")
    print(f"LIMAN-C ROC-AUC: {limanc_fixed:.3f} global -> {limanc_matched:.3f} matched")
    print(f"AMPERE ROC-AUC:  {ampere_fixed:.3f} global -> {ampere_matched:.3f} matched")
    print(
        "Healthy load accuracy: "
        f"{structure['LIMAN-C']['held_out_nearest_load_reference_accuracy']:.3f} "
        "LIMAN-C, "
        f"{structure['AMPERE']['held_out_nearest_load_reference_accuracy']:.3f} "
        "AMPERE"
    )
    print(f"Elapsed: {elapsed_seconds:.1f} s")
    print(f"Results: {output}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--limanc-root",
        type=Path,
        required=True,
        help="Extracted LIMAN-C directory containing experiment_1/current.",
    )
    value.add_argument(
        "--ampere-archive",
        type=Path,
        required=True,
        help="Downloaded AMPERE ZIP. It is streamed and is never extracted.",
    )
    value.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Results and compressed feature caches (default: ./output).",
    )
    value.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer resamples for a pipeline check; not paper reproduction.",
    )
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        limanc_root = resolve_limanc_root(args.limanc_root)
        ampere_archive = validate_ampere_archive(args.ampere_archive)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2

    output = args.output_dir.expanduser().resolve()
    limanc_output = output / "limanc"
    ampere_output = output / "ampere"
    combined_output = output / "two_dataset"
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    limanc_command = [
        sys.executable,
        str(ROOT / "run_engine2.py"),
        "--data-root",
        str(limanc_root),
        "--output-dir",
        str(limanc_output),
    ]
    ampere_command = [
        sys.executable,
        str(ROOT / "run_ampere.py"),
        "--archive",
        str(ampere_archive),
        "--output-dir",
        str(ampere_output),
    ]
    combined_command = [
        sys.executable,
        str(ROOT / "run_two_dataset_questions.py"),
        "--limanc-cache",
        str(limanc_output / "engine2_fft_features.npz"),
        "--ampere-cache",
        str(ampere_output / "ampere_rotor_fft_features.npz"),
        "--output-dir",
        str(combined_output),
    ]
    if args.quick:
        limanc_command.extend(
            ["--bootstrap-replicates", "200", "--permutation-replicates", "500"]
        )
        ampere_command.extend(["--bootstrap-replicates", "500"])
        combined_command.extend(["--repeats", "20"])

    try:
        run_command(limanc_command)
        run_command(ampere_command)
        run_command(combined_command)
        print_summary(output, time.perf_counter() - started)
    except subprocess.CalledProcessError as exc:
        print(
            f"\nStudy stopped because one stage returned exit code {exc.returncode}.",
            file=sys.stderr,
        )
        return int(exc.returncode or 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
