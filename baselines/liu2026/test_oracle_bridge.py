"""Focused checks for oracle aggregation; these never invoke Liu extraction."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from oracle_bridge import AttackBudgetExceeded, ProbTruncBridge


class SequenceOracle:
    def __init__(self, responses: list[int]):
        self.responses = iter(responses)
        self.total_queries = 0

    def prob_oracle(self, codes: np.ndarray) -> np.ndarray:
        value = next(self.responses)
        self.total_queries += codes.shape[0]
        return np.full((codes.shape[0], 1), value, dtype=np.int64)


def bridge_with_sequence(responses: list[int], *, repeats: int = 1, max_queries=None):
    weights = [np.array([[2]], dtype=np.int64)]
    biases = [np.array([0], dtype=np.int64)]
    bridge = ProbTruncBridge(
        weights, biases, 2, (1, 1), noise_seed=123,
        repeats=repeats, max_queries=max_queries,
    )
    bridge.oracle = SequenceOracle(responses)
    bridge._global_config = SimpleNamespace(
        query_count=0, set_save_queries=True, SAVED_QUERIES=[]
    )
    return bridge


class BridgeAggregationTests(unittest.TestCase):
    def test_real_oracle_repetitions_are_independent_of_attack_rng(self):
        weights = [np.array([[1]], dtype=np.int64)]
        biases = [np.array([0], dtype=np.int64)]
        average = ProbTruncBridge(weights, biases, 10, (1, 1), noise_seed=17, repeats=5)
        single = ProbTruncBridge(weights, biases, 10, (1, 1), noise_seed=17)
        np.random.seed(54321)
        state_before = np.random.get_state()
        sample = np.array([1.3])

        observed = average.predict_manual_fast(sample, None, None)
        state_after = np.random.get_state()
        self.assertEqual(state_before[0], state_after[0])
        np.testing.assert_array_equal(state_before[1], state_after[1])
        self.assertEqual(state_before[2:], state_after[2:])

        raw_outputs = [single.predict_manual_fast(sample, None, None) for _ in range(5)]
        np.testing.assert_allclose(observed, np.mean(raw_outputs, axis=0))
        self.assertEqual(average.oracle.total_queries, 5)
        self.assertEqual(single.oracle.total_queries, 5)

    def test_mean_and_physical_query_count(self):
        bridge = bridge_with_sequence([1, 2, 3], repeats=3, max_queries=6)
        inputs = np.array([[1.0], [2.0]])
        # The upstream positional parameters are deliberately wrong; oracle
        # outputs may only come from the attached integer target.
        actual = bridge.predict_manual_fast(inputs, [np.array([[999]])], [np.array([999])])
        np.testing.assert_array_equal(actual, np.array([[1.0], [1.0]]))
        self.assertEqual(bridge.oracle.total_queries, 6)
        self.assertEqual(bridge._global_config.query_count, 6)
        self.assertEqual(bridge.logical_queries, 2)
        self.assertEqual(len(bridge._global_config.SAVED_QUERIES), 2)

    def test_default_single_query_and_budget_precheck(self):
        single = bridge_with_sequence([3])
        np.testing.assert_array_equal(
            single.predict_manual_fast(np.array([1.0]), None, None),
            np.array([[1.5]]),
        )
        self.assertEqual(single.oracle.total_queries, 1)
        self.assertEqual(single.logical_queries, 1)

        capped = bridge_with_sequence([1, 2, 3], repeats=3, max_queries=5)
        with self.assertRaises(AttackBudgetExceeded):
            capped.predict_manual_fast(np.array([[1.0], [2.0]]), None, None)
        self.assertEqual(capped.oracle.total_queries, 0)
        self.assertEqual(capped._global_config.query_count, 0)

    def test_partial_failure_counts_physical_queries(self):
        bridge = bridge_with_sequence([1], repeats=2)
        with self.assertRaises(StopIteration):
            bridge.predict_manual_fast(np.array([1.0]), None, None)
        self.assertEqual(bridge.oracle.total_queries, 1)
        self.assertEqual(bridge._global_config.query_count, 1)
        self.assertEqual(bridge.logical_queries, 0)
        self.assertEqual(bridge._global_config.SAVED_QUERIES, [])

    def test_repeats_must_be_positive_integer(self):
        for invalid in (0, -1, 1.5, True):
            with self.subTest(repeats=invalid), self.assertRaises(ValueError):
                bridge_with_sequence([1], repeats=invalid)


if __name__ == "__main__":
    unittest.main()
