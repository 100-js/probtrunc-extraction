"""Target MLP training and quantized structural screening."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from .checkpoints import Checkpoint
from .config import TargetConfig


def prepare_training_data(
    features: NDArray, labels: NDArray, seed: int, profile: str
) -> tuple[NDArray, NDArray]:
    if profile == "original-digits":
        # Small Gaussian jitter prevents degenerate dead/zero weights on sparse pixels.
        noise = np.random.RandomState(seed).normal(0, 0.01, features.shape)
        return (features + noise).astype(np.float32), labels

    if profile == "rebuttal-narrow":
        if len(features) > 10_000:
            indices = np.random.RandomState(0).choice(len(features), 10_000, replace=False)
            features, labels = features[indices], labels[indices]
        noise = np.random.RandomState(seed).normal(0, 0.01, features.shape).astype(np.float32)
        return features + noise, labels

    if len(features) > 15_000:
        indices = np.random.RandomState(seed).permutation(len(features))[:15_000]
        return features[indices], labels[indices]
    return features, labels


def _has_one_signed_vector(weights: NDArray, axis: int) -> bool:
    return bool(np.any(np.all(weights >= 0, axis=axis)) or np.any(np.all(weights <= 0, axis=axis)))


def _passes_screen(weights: tuple[NDArray, ...], profile: str) -> bool:
    """Reject quantized targets containing dead or strictly one-signed neurons."""
    if any(_has_one_signed_vector(weight, axis=0) for weight in weights):
        return False
    # Narrow MNIST skips first-layer input rows (zero border pixels); wide checks columns only.
    if profile == "rebuttal-wide":
        return True
    row_layers = weights[1:] if profile == "rebuttal-narrow" else weights
    return not any(_has_one_signed_vector(weight, axis=1) for weight in row_layers)


def train_classifier(
    raw_features: NDArray, labels: NDArray, config: TargetConfig, seed: int
) -> tuple[Checkpoint, int]:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a 32-bit unsigned integer")
    if not np.array_equal(np.unique(labels), np.arange(config.architecture[-1])):
        raise ValueError("the training data must contain all ten class labels")
    features, labels = prepare_training_data(raw_features, labels, seed, config.training_profile)
    max_iter = 1000 if config.training_profile == "original-digits" else 300
    for effective_seed in range(seed, 2**32):
        training_x, _, training_y, _ = train_test_split(
            features, labels, test_size=0.2, random_state=effective_seed
        )
        model = MLPClassifier(
            hidden_layer_sizes=config.architecture[1:-1],
            activation="relu",
            max_iter=max_iter,
            random_state=effective_seed,
        )
        model.fit(training_x, training_y)
        weights = tuple(np.round(weight * config.q).astype(np.int64) for weight in model.coefs_)
        if _passes_screen(weights, config.training_profile):
            biases = tuple(
                np.round(bias * config.q).astype(np.int64) for bias in model.intercepts_
            )
            return Checkpoint(config.q, weights, biases), effective_seed
    raise ValueError("no accepted target before the seed range was exhausted")
