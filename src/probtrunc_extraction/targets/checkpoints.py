"""Integer fixed-point checkpoint container and atomic NPZ I/O."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Checkpoint:
    q: int
    weights: tuple[NDArray[np.int64], ...]
    biases: tuple[NDArray[np.int64], ...]

    def __post_init__(self) -> None:
        if type(self.q) is not int or self.q < 1 or self.q & (self.q - 1):
            raise ValueError("checkpoint q must be a positive integer power of two")
        if not self.weights or len(self.weights) != len(self.biases):
            raise ValueError("checkpoint must have matching weight and bias arrays")
        input_width = None
        for weight, bias in zip(self.weights, self.biases, strict=True):
            if not _is_int64(weight) or not _is_int64(bias):
                raise ValueError("checkpoint weights and biases must be signed 64-bit integers")
            if weight.ndim != 2 or not weight.size or bias.shape != (weight.shape[1],):
                raise ValueError("checkpoint has inconsistent weight and bias shapes")
            if input_width is not None and weight.shape[0] != input_width:
                raise ValueError("checkpoint layers have inconsistent dimensions")
            input_width = weight.shape[1]

    @property
    def architecture(self) -> tuple[int, ...]:
        return (self.weights[0].shape[0], *(weight.shape[1] for weight in self.weights))


def _is_int64(value: NDArray) -> bool:
    return isinstance(value, np.ndarray) and value.dtype.kind == "i" and value.dtype.itemsize == 8


def _read_int(value: NDArray) -> int:
    if value.shape != () or value.dtype.kind not in "iu":
        raise ValueError("checkpoint metadata must be an integer scalar")
    return int(value.item())


def load_checkpoint(path: str | Path) -> Checkpoint:
    try:
        with np.load(path, allow_pickle=False) as archive:
            q = _read_int(archive["q"])
            layer_count = _read_int(archive["num_layers"])
            if not 1 <= layer_count <= (len(archive.files) - 2) // 2:
                raise ValueError("num_layers does not match the arrays in the checkpoint")
            weights = tuple(archive[f"W_{index}"] for index in range(layer_count))
            biases = tuple(archive[f"B_{index}"] for index in range(layer_count))
        return Checkpoint(q, weights, biases)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid checkpoint {path}: {error}") from error


def save_checkpoint(path: str | Path, checkpoint: Checkpoint) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"q": checkpoint.q, "num_layers": len(checkpoint.weights)}
    for index, (weight, bias) in enumerate(zip(checkpoint.weights, checkpoint.biases, strict=True)):
        arrays[f"W_{index}"] = weight
        arrays[f"B_{index}"] = bias
    handle = NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez(handle, **arrays)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
