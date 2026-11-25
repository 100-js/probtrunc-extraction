import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from probtrunc_extraction.evaluation.liu_metrics import score_liu_run


class LiuMetricsTests(unittest.TestCase):
    def test_score_liu_run_matches_permuted_scaled_parameters(self):
        rng = np.random.default_rng(0)
        W_true = rng.integers(-200, 200, size=(64, 20), dtype=np.int64)
        B_true = rng.integers(-100, 100, size=(20,), dtype=np.int64)
        perm = rng.permutation(20)
        scales = rng.uniform(0.5, 2.0, size=(20,))

        sig_w = W_true[:, perm].astype(np.float64) * (-scales[None, :])
        sig_b = B_true[perm].astype(np.float64) * (-scales)

        rec_w = W_true[:, perm].astype(np.float64) * scales[None, :]
        bias_noise = np.linspace(0.5, 4.0, 20)
        rec_b = (B_true[perm].astype(np.float64) + bias_noise) * scales

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.npz"
            np.savez(target, q=512, num_layers=1, W_0=W_true, B_0=B_true)

            run_dir = root / "run"
            run_dir.mkdir()
            np.savez(run_dir / "signature_checkpoint_1.npz", extracted_normal=sig_w, extracted_bias=sig_b)
            np.savez(run_dir / "recovered_1.npz", extracted_normal=rec_w, extracted_bias=rec_b)
            (run_dir / "run_manifest.json").write_text(
                json.dumps({"status": "completed", "mode": "pt", "repeats": 1, "physical_oracle_queries": 12345}),
                encoding="utf-8",
            )

            report = score_liu_run(target, run_dir=run_dir)
            self.assertEqual(report["signature"]["matching_directions"], 20)
            self.assertAlmostEqual(report["signature"]["max_chordal_error"], 0.0, places=10)
            self.assertEqual(report["final_first_layer"]["positive_sign_matches"], 20)
            self.assertAlmostEqual(report["final_first_layer"]["absolute_bias_error_lsb"]["min"], 0.5, places=6)
            self.assertAlmostEqual(report["final_first_layer"]["absolute_bias_error_lsb"]["max"], 4.0, places=6)
            self.assertEqual(report["manifest"]["physical_oracle_queries"], 12345)


if __name__ == "__main__":
    unittest.main()
