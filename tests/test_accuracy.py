from pathlib import Path
import unittest

from probtrunc_extraction.evaluation.accuracy import score_checkpoint
from probtrunc_extraction.targets.config import load_target_config

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"


def _find_local_mnist() -> Path | None:
    for candidate in (ROOT / "data" / "mnist.npz", ROOT.parent / "data" / "mnist.npz"):
        if candidate.is_file():
            return candidate
    return None


class AccuracyTests(unittest.TestCase):
    def test_original_digits_holdouts(self):
        config = load_target_config(ROOT / "configs" / "digits_64_20_20_10_s9.toml")
        results = [
            score_checkpoint(config, config.checkpoint_path(CHECKPOINTS / "digits", index))
            for index in range(1, len(config.model_seeds) + 1)
        ]
        self.assertEqual(sum(r["samples"] for r in results), 1800)
        self.assertEqual(sum(r["quantized_correct"] for r in results), 1744)

    def test_mnist_wide_scores_official_test_split(self):
        mnist_file = _find_local_mnist()
        if mnist_file is None:
            self.skipTest("Local mnist.npz not present for offline unit test")
        config = load_target_config(ROOT / "configs" / "mnist_784_128_64_10_s6.toml")
        results = [
            score_checkpoint(
                config,
                config.checkpoint_path(CHECKPOINTS / "mnist-wide", index),
                mnist_file,
                allow_download=False,
            )
            for index in range(1, len(config.model_seeds) + 1)
        ]
        self.assertTrue(all(r["split"] == "official-test" and r["samples"] == 10000 for r in results))
        self.assertEqual(sum(r["quantized_correct"] for r in results), 28866)


if __name__ == "__main__":
    unittest.main()
