from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from probtrunc_extraction.targets.checkpoints import load_checkpoint, save_checkpoint
from probtrunc_extraction.targets.config import load_benchmark_config

ROOT = Path(__file__).resolve().parents[1]


class CheckpointTests(unittest.TestCase):
    def test_all_configs_match_bundled_checkpoints(self):
        config_paths = sorted((ROOT / "configs").glob("*.toml"))
        self.assertEqual(len(config_paths), 5)
        for path in config_paths:
            cfg = load_benchmark_config(path)
            target = cfg.target
            group_dir = ROOT / "checkpoints" / target.checkpoint_group
            for index in range(1, len(target.model_seeds) + 1):
                with self.subTest(config=path.name, model=index):
                    ckpt = load_checkpoint(target.checkpoint_path(group_dir, index))
                    self.assertEqual(ckpt.architecture, target.architecture)
                    self.assertEqual(ckpt.q, target.q)

    def test_round_trip_preserves_int64_arrays(self):
        cfg = load_benchmark_config(ROOT / "configs" / "digits_64_20_20_10_s9.toml")
        ckpt = load_checkpoint(
            cfg.target.checkpoint_path(ROOT / "checkpoints" / cfg.target.checkpoint_group, 1)
        )
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "sub" / "model.npz"
            save_checkpoint(dest, ckpt)
            reloaded = load_checkpoint(dest)
        self.assertEqual(reloaded.q, ckpt.q)
        for a, b in zip(reloaded.weights + reloaded.biases, ckpt.weights + ckpt.biases, strict=True):
            np.testing.assert_array_equal(a, b, strict=True)


if __name__ == "__main__":
    unittest.main()
