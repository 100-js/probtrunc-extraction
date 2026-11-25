import unittest

import numpy as np

from probtrunc_extraction.extractors.common import (
    find_rank_full_sub_indices,
    primitive_vector,
    solve_last_layer_flooded_pinv,
    solve_last_layer_lll,
)


class _StopBeforeSolving(Exception):
    pass


class ExtractorCommonTests(unittest.TestCase):
    def test_primitive_vector_canonicalizes_gcd_and_sign(self):
        v = np.array([0, -6, 9, -15], dtype=np.int64)
        prim = primitive_vector(v)
        np.testing.assert_array_equal(prim, np.array([0, 2, -3, 5], dtype=np.int64))
        self.assertIsNone(primitive_vector(np.zeros(4, dtype=np.int64)))

    def test_greedy_and_qr_both_produce_full_rank_submatrices(self):
        rng = np.random.default_rng(7)
        mat = np.zeros((30, 5), dtype=np.int64)
        mat[5:, :] = rng.integers(-50, 50, size=(25, 5))
        for method in ("greedy", "qr"):
            indices = find_rank_full_sub_indices(mat, 5, method=method)
            self.assertEqual(len(indices), 5)
            self.assertEqual(np.linalg.matrix_rank(mat[indices, :]), 5)

    def test_last_layer_query_counts_match_repetitions(self):
        weights = [np.eye(20, dtype=np.int64), np.eye(20, dtype=np.int64)]
        biases = [np.zeros(20, dtype=np.int64), np.zeros(20, dtype=np.int64)]

        for reps, expected in ((1, 30), (16, 480)):
            calls = []

            def oracle(batch, sf_leak=False):
                calls.append(len(batch))
                raise _StopBeforeSolving

            with self.subTest(solver="lll", reps=reps):
                with self.assertRaises(_StopBeforeSolving):
                    solve_last_layer_lll(oracle, weights, biases, 10, 64, last_layer_repetitions=reps)
                self.assertEqual(calls, [expected])

        calls = []

        def oracle_pinv(batch, sf_leak=False):
            calls.append(len(batch))
            raise _StopBeforeSolving

        with self.assertRaises(_StopBeforeSolving):
            solve_last_layer_flooded_pinv(oracle_pinv, weights, biases, 10, 4096, last_layer_repetitions=64)
        self.assertEqual(calls, [3840])


if __name__ == "__main__":
    unittest.main()
