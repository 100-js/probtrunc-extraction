import unittest

import numpy as np

from probtrunc_extraction.oracle import CPUOracleCounter


class OracleTests(unittest.TestCase):
    def test_int64_and_float64_backends_match_below_2_to_53(self):
        W = [np.array([[3, -2], [1, 4]], dtype=np.int64), np.array([[2], [-3]], dtype=np.int64)]
        B = [np.array([5, -1], dtype=np.int64), np.array([7], dtype=np.int64)]
        X = np.array([[100, -50], [25, 80]], dtype=np.int64)

        np.random.seed(123)
        o_int = CPUOracleCounter(W, B, 64, (2, 2, 1), backend="int64", track_bitwidth=True)
        y_int = o_int.prob_oracle(X)

        np.random.seed(123)
        o_flt = CPUOracleCounter(W, B, 64, (2, 2, 1), backend="float64", track_bitwidth=True)
        y_flt = o_flt.prob_oracle(X)

        np.testing.assert_array_equal(y_int, y_flt)
        self.assertEqual(o_int.total_queries, 2)
        report = o_int.bitwidth_report()
        self.assertEqual(report["input_max_abs"], 100)
        self.assertEqual(len(report["layers"]), 2)
        self.assertGreater(report["layers"][0]["pre_trunc_max_abs"], 0)

    def test_int64_preserves_exact_odd_product_above_2_to_53(self):
        # Product: (2^53 + 1) * 3 = 3 * 2^53 + 3 (odd, > 2^54), exact only in int64.
        val = (1 << 53) + 1
        W = [np.array([[3]], dtype=np.int64)]
        B = [np.array([0], dtype=np.int64)]
        o_int = CPUOracleCounter(W, B, 8, (1, 1), backend="int64", track_bitwidth=True)
        o_int.prob_oracle(np.array([[val]], dtype=np.int64))
        self.assertEqual(o_int.bitwidth_report()["layers"][0]["pre_trunc_max_abs"], val * 3)

    def test_overflow_guard_rejects_excessive_inputs(self):
        W = [np.array([[1000]], dtype=np.int64)]
        B = [np.array([0], dtype=np.int64)]
        oracle = CPUOracleCounter(W, B, 64, (1, 1), backend="int64")
        with self.assertRaises(OverflowError):
            oracle.prob_oracle(np.array([[1 << 58]], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
