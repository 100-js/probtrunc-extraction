"""MNIST dataset loader with on-demand download and SHA-256 verification."""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import NamedTuple
import urllib.request

import numpy as np
from numpy.typing import NDArray

from .checkpoints import Checkpoint
from .config import TargetConfig

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"


class MNISTData(NamedTuple):
    x_train: NDArray[np.float32]
    y_train: NDArray
    x_test: NDArray[np.float32]
    y_test: NDArray


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def ensure_mnist(dataset_path: str | Path, *, allow_download: bool = True) -> Path:
    """Return a SHA-256 verified path to mnist.npz, downloading on demand if absent."""
    path = Path(dataset_path)
    if not path.is_file():
        if not allow_download:
            raise FileNotFoundError(f"MNIST archive not found: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = NamedTemporaryFile(dir=path.parent, suffix=".npz.tmp", delete=False)
        temporary = Path(handle.name)
        handle.close()
        try:
            print(f"[*] Downloading MNIST dataset from {MNIST_URL} to {path}...", flush=True)
            urllib.request.urlretrieve(MNIST_URL, temporary)
            actual = _sha256(temporary)
            if actual != MNIST_SHA256:
                raise ValueError(
                    f"Downloaded MNIST SHA-256 mismatch: expected {MNIST_SHA256}, found {actual}"
                )
            temporary.replace(path)
            print(f"[+] Verified MNIST archive saved to {path}.", flush=True)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        actual = _sha256(path)
        if actual != MNIST_SHA256:
            raise ValueError(
                f"SHA-256 mismatch for {path}: expected {MNIST_SHA256}, found {actual}"
            )
    return path


def load_data(dataset_path: str | Path, *, allow_download: bool = True) -> MNISTData:
    verified = ensure_mnist(dataset_path, allow_download=allow_download)
    with np.load(verified, allow_pickle=False) as archive:
        splits = []
        for split in ("train", "test"):
            images, labels = archive[f"x_{split}"], archive[f"y_{split}"]
            if images.ndim != 3 or images.shape[1:] != (28, 28) or labels.shape != (len(images),):
                raise ValueError(f"MNIST has invalid image or label dimensions in split {split}")
            features = images.reshape(len(images), 784).astype(np.float32) / 255.0
            splits.append((features, labels))
    return MNISTData(*splits[0], *splits[1])


def train(config: TargetConfig, seed: int, dataset_path: str | Path) -> tuple[Checkpoint, int]:
    from .training import train_classifier

    if config.dataset != "mnist":
        raise ValueError("the MNIST trainer requires dataset = 'mnist'")
    data = load_data(dataset_path)
    return train_classifier(data.x_train, data.y_train, config, seed)
