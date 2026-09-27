"""Run the fixed-versus-condition-matched reference experiment on LIMAN-C."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from condition_reference import (
    discover_engine2_records,
    evaluate,
    extract_feature_table,
    load_feature_table,
    save_feature_table,
    score_reference_strategies,
    write_json,
    write_parent_scores,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--permutation-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260925)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "experiment.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
    )
    started = time.perf_counter()
    cache_path = (
        args.feature_cache.resolve()
        if args.feature_cache is not None
        else output_dir / "engine2_fft_features.npz"
    )
    if cache_path.exists():
        logging.info("loading_feature_cache path=%s", cache_path)
        table = load_feature_table(cache_path)
    else:
        records = discover_engine2_records(args.data_root.resolve())
        logging.info("discovered_parents count=%d", len(records))
        table = extract_feature_table(records)
        save_feature_table(table, cache_path)
        logging.info("saved_feature_cache path=%s", cache_path)
    scores = score_reference_strategies(table)
    results = evaluate(
        table,
        scores,
        bootstrap_replicates=args.bootstrap_replicates,
        permutation_replicates=args.permutation_replicates,
        seed=args.seed,
    )
    write_parent_scores(table, scores, output_dir / "parent_scores.csv")
    write_json(results, output_dir / "results.json")
    logging.info(
        "completed elapsed_s=%.3f fixed_auc=%.6f conditional_auc=%.6f delta=%.6f",
        time.perf_counter() - started,
        results["methods"]["fixed_pooled"]["roc_auc"],
        results["methods"]["condition_matched"]["roc_auc"],
        results["primary_comparison"]["auc_difference"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

