"""Checks for the diagnostic's scoring and query-geometry measurements."""

from __future__ import annotations

import unittest

import numpy as np

from probe_liu_ratio_grid_epsilon import (
    QueryRouter, choose_diagnostic_points, prepare_point_centers, projective_error,
)


class RatioGridEpsilonTests(unittest.TestCase):
    def test_projective_error_ignores_sign_and_nonzero_scale(self) -> None:
        true = np.array([3.0, 4.0])
        self.assertAlmostEqual(projective_error(np.array([-6.0, -8.0]), true)["unsigned_projective_error"], 0.0)
        self.assertAlmostEqual(projective_error(np.array([-4.0, 3.0]), true)["unsigned_projective_error"], 1.0)

    def test_collision_summary_is_about_input_codes(self) -> None:
        router = QueryRouter(q=2)
        router.input_codes = [
            np.array([0, 0]), np.array([0, 0]),
            np.array([1, 0]), np.array([2, 0]),
        ]
        summary = router.collision_summary()
        self.assertEqual(summary["completed_four_query_stencils"], 1)
        self.assertEqual(summary["identical_input_code_pairs"], 1)
        self.assertEqual(summary["identical_input_code_pair_fraction"], 0.5)
        self.assertEqual(summary["identical_input_code_coordinate_fraction"], 0.75)

    def test_selected_points_are_on_target_hyperplane(self) -> None:
        weights = [
            np.array([[2, 1], [1, -2]], dtype=np.int64),
            np.array([[1], [1]], dtype=np.int64),
        ]
        biases = [
            np.array([1, -1], dtype=np.int64),
            np.array([0], dtype=np.int64),
        ]
        kwargs = dict(
            neuron_index=0, seed=42, point_scale=3.0,
            candidate_pool=8, point_count=2, max_eps=0.25,
        )
        points = choose_diagnostic_points(weights, biases, q=4, **kwargs)
        self.assertEqual(points, choose_diagnostic_points(weights, biases, q=4, **kwargs))
        self.assertEqual(len(points), 2)
        for item in points:
            point = np.asarray(item["point"])
            self.assertAlmostEqual(float(point @ (weights[0][:, 0] / 4) + 1 / 4), 0.0)

        exact = prepare_point_centers(points, weights, biases, 4, neuron_index=0, snap_to_grid=False)
        snapped = prepare_point_centers(points, weights, biases, 4, neuron_index=0, snap_to_grid=True)
        for base, unsnapped, rounded in zip(points, exact, snapped):
            self.assertEqual(base["point"], unsnapped["supplied_center"])
            self.assertEqual(unsnapped["snapped_point"], rounded["supplied_center"])
            self.assertEqual(unsnapped["snapped_point"], rounded["snapped_point"])
            self.assertAlmostEqual(rounded["supplied_center_target_preactivation_residual"],
                                   rounded["snapped_target_preactivation_residual"])
            self.assertLessEqual(rounded["snap_linf_input_lsbs"], 0.5)


if __name__ == "__main__":
    unittest.main()
