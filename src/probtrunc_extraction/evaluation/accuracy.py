"""Deterministic fixed-point and floating-point accuracy evaluation for target models."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..fixedpoint import INT64_MAX, fits_int64
from ..targets import digits, mnist
from ..targets.checkpoints import Checkpoint, load_checkpoint
from ..targets.config import TargetConfig, load_target_config


def _max_magnitude(values: np.ndarray) -> int:
    return max(abs(int(values.min())), abs(int(values.max())))


def _quantized_predictions(images: np.ndarray, checkpoint: Checkpoint) -> np.ndarray:
    q = checkpoint.q
    if q > INT64_MAX:
        raise ValueError("Checkpoint scale must fit in signed int64")
    encoded = np.rint(images * q)
    if not np.isfinite(encoded).all() or encoded.min() < -(1 << 63) or encoded.max() >= (1 << 63):
        raise OverflowError("Encoded input image does not fit in signed int64")
    activations = encoded.astype(np.int64)
    for layer, (weight, bias) in enumerate(zip(checkpoint.weights, checkpoint.biases, strict=True)):
        accumulator_bound = _max_magnitude(activations) * _max_magnitude(weight) * weight.shape[0]
        if not fits_int64(accumulator_bound):
            raise OverflowError(f"Layer {layer + 1} cannot be accumulated safely in signed int64")
        accumulator = activations @ weight
        quotient, remainder = np.divmod(accumulator, q)
        increment = (remainder > q // 2) | (
            (q % 2 == 0) & (remainder == q // 2) & (quotient % 2 == 1)
        )
        scaled = quotient + increment
        if not fits_int64(int(scaled.min()) + int(bias.min())) or not fits_int64(
            int(scaled.max()) + int(bias.max())
        ):
            raise OverflowError(f"Layer {layer + 1} bias addition can overflow signed int64")
        activations = scaled + bias
        if layer < len(checkpoint.weights) - 1:
            activations = np.maximum(activations, 0)
    return np.argmax(activations, axis=1)


def _float_predictions(images: np.ndarray, checkpoint: Checkpoint) -> np.ndarray:
    activations = images
    for layer, (weight, bias) in enumerate(zip(checkpoint.weights, checkpoint.biases, strict=True)):
        activations = activations @ (weight / checkpoint.q) + (bias / checkpoint.q)
        if layer < len(checkpoint.weights) - 1:
            activations = np.maximum(activations, 0)
    return np.argmax(activations, axis=1)


def score_checkpoint(
    config: TargetConfig | str | Path,
    checkpoint: Checkpoint | str | Path,
    dataset_path: str | Path | None = None,
    model_seed: int | None = None,
    *,
    effective_seed: int | None = None,
    allow_download: bool = True,
) -> dict:
    """Evaluate one quantized checkpoint on its canonical labeled evaluation split."""
    if not isinstance(config, TargetConfig):
        config = load_target_config(config)
    if not isinstance(checkpoint, Checkpoint):
        path = Path(checkpoint)
        if model_seed is None:
            for model_index, seed in enumerate(config.model_seeds, start=1):
                if path.name == config.checkpoint_path(path.parent, model_index).name:
                    model_seed = seed
                    break
        checkpoint = load_checkpoint(path)

    shape = checkpoint.architecture
    if shape != config.architecture or checkpoint.q != config.q:
        raise ValueError("Checkpoint architecture or scale does not match target config")

    split = config.evaluation_split
    if config.dataset == "digits":
        if model_seed is None:
            raise ValueError("Digits evaluation requires a model seed")
        images, labels = digits.holdout(
            model_seed, training_profile=config.training_profile, effective_seed=effective_seed
        )
    elif config.dataset == "mnist":
        if dataset_path is None:
            raise ValueError("An MNIST dataset path is required")
        _, _, images, labels = mnist.load_data(dataset_path, allow_download=allow_download)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    sample_count = len(labels)
    if sample_count == 0:
        raise ValueError(f"Dataset split {split} contains no examples")
    quantized_correct = int(np.count_nonzero(_quantized_predictions(images, checkpoint) == labels))
    float_correct = int(np.count_nonzero(_float_predictions(images, checkpoint) == labels))
    return {
        "target": config.name,
        "dataset": config.dataset,
        "architecture": list(shape),
        "q": checkpoint.q,
        "model_seed": model_seed,
        "split": split,
        "samples": sample_count,
        "quantized_correct": quantized_correct,
        "quantized_accuracy_percent": 100.0 * quantized_correct / sample_count,
        "float_correct": float_correct,
        "float_accuracy_percent": 100.0 * float_correct / sample_count,
    }
