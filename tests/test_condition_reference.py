"""Small deterministic tests for the conference experiment."""

from __future__ import annotations

import unittest

import numpy as np

from condition_reference import (
    FeatureTable,
    HealthyReferenceDetector,
    roc_auc,
    score_reference_strategies,
)


class ConditionReferenceTests(unittest.TestCase):
    def test_detector_fits_healthy_rows_and_scores_queries(self) -> None:
        healthy = np.asarray([[0.0, 1.0], [0.1, 0.9], [-0.1, 1.1]])
        detector = HealthyReferenceDetector.fit(healthy)
        scores = detector.score(np.asarray([[0.0, 1.0], [3.0, 4.0]]))
        self.assertEqual(scores.shape, (2,))
        self.assertLess(scores[0], scores[1])

    def test_auc_is_one_for_perfect_ranking(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.0, 0.1, 0.8, 0.9])
        self.assertEqual(roc_auc(labels, scores), 1.0)

    def test_condition_matching_removes_load_shift(self) -> None:
        rng = np.random.default_rng(42)
        rows: list[np.ndarray] = []
        states: list[str] = []
        loads: list[int] = []
        parent_ids: list[str] = []
        for load, offset in ((0, 0.0), (100, 5.0)):
            for index in range(8):
                rows.append(offset + rng.normal(0.0, 0.1, size=6))
                states.append("normal")
                loads.append(load)
                parent_ids.append(f"h-{load}-{index}")
            for index in range(8):
                rows.append(offset + 2.0 + rng.normal(0.0, 0.1, size=6))
                states.append("fault")
                loads.append(load)
                parent_ids.append(f"f-{load}-{index}")
        table = FeatureTable(
            parent_ids=np.asarray(parent_ids),
            states=np.asarray(states),
            loads=np.asarray(loads),
            frequencies_hz=np.arange(6, dtype=float),
            features=np.vstack(rows),
        )
        scores = score_reference_strategies(table)
        self.assertGreater(
            roc_auc(table.labels, scores["condition_matched"]),
            roc_auc(table.labels, scores["fixed_pooled"]),
        )


if __name__ == "__main__":
    unittest.main()
