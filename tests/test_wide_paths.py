from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from probtrunc_extraction.extractors.common import solve_last_layer_flooded_pinv
from probtrunc_extraction.extractors.mnist import decrypt_B2, decrypt_L1, decrypt_L_last, decrypt_W2
from probtrunc_extraction.oracle import CPUOracleCounter
from probtrunc_extraction.targets.checkpoints import Checkpoint, save_checkpoint
from probtrunc_extraction.targets.config import ExtractionConfig
from probtrunc_extraction.runners.benchmark import run_trial


class _StopBeforeSolving(Exception):
    pass


class WideMNISTPathTests(unittest.TestCase):
    def test_flooded_pinv_wide_query_count(self):
        weights = [np.eye(128, dtype=np.int64), np.eye(128, 64, dtype=np.int64)]
        biases = [np.zeros(128, dtype=np.int64), np.zeros(64, dtype=np.int64)]
        calls = []

        def oracle(batch, sf_leak=False):
            calls.append(len(batch))
            raise _StopBeforeSolving

        with self.assertRaises(_StopBeforeSolving):
            solve_last_layer_flooded_pinv(oracle, weights, biases, 10, 64, last_layer_repetitions=64)
        # 3 scales * 64 hidden neurons * 64 repetitions = 12,288 physical queries
        self.assertEqual(calls, [12288])

    def test_wide_profile_end_to_end_on_synthetic_network(self):
        rng = np.random.default_rng(1234)
        q = 64
        arch = (16, 6, 4, 5)
        W0 = rng.integers(-40, 41, size=(16, 6), dtype=np.int64)
        B0 = rng.integers(-20, 21, size=(6,), dtype=np.int64)
        W1 = rng.integers(-40, 41, size=(6, 4), dtype=np.int64)
        B1 = rng.integers(-20, 21, size=(4,), dtype=np.int64)
        W2 = rng.integers(-40, 41, size=(4, 5), dtype=np.int64)
        B2 = rng.integers(-20, 21, size=(5,), dtype=np.int64)
        # Ensure mixed signs in every column so no dead/monotonic neurons exist.
        for W in (W0, W1, W2):
            W[0, :] = np.abs(W[0, :]) + 5
            W[1, :] = -(np.abs(W[1, :]) + 5)
        # Ensure every column of W0 and W1 is coprime (GCD == 1) and has >=10 nonzeros in W0.
        W0[0, :] = 17
        W0[1, :] = -19
        W0[2:, :] = np.where(W0[2:, :] == 0, 3, W0[2:, :])
        W1[0, :] = 13
        W1[1, :] = -17

        ckpt = Checkpoint(q=q, weights=(W0, W1, W2), biases=(B0, B1, B2))
        ext = ExtractionConfig(
            extractor="mnist",
            oracle_backend="float64",
            trial_seeds=(1,),
            w1_anchors=200,
            w1_steps_per_anchor=10,
            w1_queries=1,
            b1_anchors=10,
            b1_queries=100,
            rank1_consensus=True,
            b1_high_snr_filter=True,
            w2_walk_mode="sparsity_masked",
            w2_inv_backend="float64",
            w2_max_anchors=300,
            w2_steps_per_anchor=15,
            w2_p1_queries=1,
            w2_p2_queries=0,
            w2_p3_queries=1,
            b2_num_bases=10,
            b2_test_queries=200,
            last_layer_solver="flooded_pinv",
            last_layer_repetitions=64,
            bias_tolerances=None,
        )

        with TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "synthetic_wide.npz"
            save_checkpoint(ckpt_path, ckpt)
            res = run_trial(ckpt_path, arch, q, 1, 1, ext)

        self.assertTrue(res["L1_success"], msg=f"L1 failed: {res}")
        self.assertTrue(res["W2_success"], msg=f"W2 failed: {res}")
        self.assertTrue(res["B2_success"], msg=f"B2 failed: {res}")
        self.assertTrue(res["L_last_success"], msg=f"L_last failed: {res}")
        self.assertTrue(res["end_to_end_success"], msg=f"End-to-end failed: {res}")
        self.assertEqual(res["L1_max_weight_err"], 0)
        self.assertEqual(res["W2_max_weight_err"], 0)
        self.assertEqual(res["L_last_max_weight_err"], 0)


if __name__ == "__main__":
    unittest.main()
