"""Focused checks for the first-layer comparison's scaling and oracle modes."""

import unittest

import numpy as np

from evaluate_first_layer import compare_reference, evaluate
from first_layer_adapter import ContinuousOracleBridge


class FirstLayerMetricTests(unittest.TestCase):
    def test_permutation_positive_scaling_and_sign(self):
        true_w = np.eye(2)
        true_b = np.array([0.2, -0.3])
        recovered_w = true_w[:, [1, 0]] * np.array([3.0, 2.0])
        recovered_b = true_b[[1, 0]] * np.array([3.0, 2.0])
        signed = evaluate(true_w, true_b, recovered_w, recovered_b)
        self.assertEqual(signed["coverage"]["error_le_0.01"], 1.0)

        recovered_w[:, 0] *= -1
        recovered_b[0] *= -1
        signed = evaluate(true_w, true_b, recovered_w, recovered_b)
        unsigned = evaluate(true_w, true_b, recovered_w, recovered_b, unsigned=True)
        self.assertEqual(signed["coverage"]["error_le_0.01"], 0.5)
        self.assertEqual(unsigned["coverage"]["error_le_0.01"], 1.0)

    def test_continuous_and_grid_reference_differ_below_input_step(self):
        # W/q = 1 and B = 0, so the only difference is input rounding.
        bridge = ContinuousOracleBridge(
            [np.array([[2]], dtype=np.int64)], [np.array([0], dtype=np.int64)],
            2, (1, 1), grid_input=False,
        )
        grid = ContinuousOracleBridge(
            [np.array([[2]], dtype=np.int64)], [np.array([0], dtype=np.int64)],
            2, (1, 1), grid_input=True,
        )
        query = np.array([[0.2]])
        self.assertAlmostEqual(bridge.predict_manual_fast(query, None, None)[0, 0], 0.2)
        self.assertAlmostEqual(grid.predict_manual_fast(query, None, None)[0, 0], 0.0)
        self.assertEqual((bridge.total_queries, grid.total_queries), (1, 1))

    def test_zero_coverage_reference_cannot_claim_parity(self):
        empty = {
            "target_first_layer_sha256": "same",
            "metric": "unsigned_augmented_l2_projective_error",
            "coverage": {"error_le_0.01": 0.0, "error_le_0.1": 0.0},
            "error_summary": {"median": 1.0, "p90": 1.0, "max": 1.0},
        }
        result = compare_reference(empty, empty, 0.001)
        self.assertFalse(result["reference_nontrivial"])
        self.assertFalse(result["matches_float_reference"])


if __name__ == "__main__":
    unittest.main()
