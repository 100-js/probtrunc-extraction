"""Scikit-learn 8x8 Digits dataset helpers and target training entry point."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from .checkpoints import Checkpoint
from .config import TargetConfig
from .training import prepare_training_data, train_classifier


def _load_data(profile: str) -> tuple[NDArray, NDArray]:
    digits = load_digits()
    # The original 64-20-20-10 benchmark kept raw 0-16 intensities.
    if profile == "original-digits":
        return digits.data, digits.target
    return digits.data.astype(np.float32) / 16.0, digits.target


def holdout(
    seed: int,
    *,
    training_profile: str = "original-digits",
    effective_seed: int | None = None,
) -> tuple[NDArray, NDArray]:
    features, labels = _load_data(training_profile)
    features, labels = prepare_training_data(features, labels, seed, training_profile)
    _, evaluation_x, _, evaluation_y = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=seed if effective_seed is None else effective_seed,
    )
    return evaluation_x, evaluation_y


def train(config: TargetConfig, seed: int) -> tuple[Checkpoint, int]:
    if config.dataset != "digits":
        raise ValueError("the Digits trainer requires dataset = 'digits'")
    features, labels = _load_data(config.training_profile)
    return train_classifier(features, labels, config, seed)
