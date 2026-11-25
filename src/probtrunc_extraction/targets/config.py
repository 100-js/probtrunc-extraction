"""Declarative TOML configuration for target training and extraction benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import tomllib

_INPUT_WIDTHS = {"digits": 64, "mnist": 784}
_EVALUATION_SPLITS = {
    "digits": ("digits-holdout",),
    "mnist": ("official-test",),
}
_TRAINING_PROFILES = {"original-digits", "rebuttal-narrow", "rebuttal-wide"}
_CHECKPOINT_GROUPS = {"digits", "mnist-narrow", "mnist-wide"}
_EXTRACTORS = {"original-digits", "mnist"}
_ORACLE_BACKENDS = {"int64", "float64"}
_WALK_MODES = {"dense", "sparsity_masked"}
_W2_INV_BACKENDS = {"sympy", "float64"}
_LAST_LAYER_SOLVERS = {"lll", "flooded_pinv"}
_CHECKPOINT_PATTERN = re.compile(
    r"[^/\\:{}]*\{model_index(?::02d)?\}[^/\\:{}]*\.npz"
)


@dataclass(frozen=True)
class TargetConfig:
    name: str
    dataset: str
    architecture: tuple[int, ...]
    precision: int
    model_seeds: tuple[int, ...]
    training_profile: str
    checkpoint_pattern: str
    evaluation_split: str
    checkpoint_group: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(r"[a-z][a-z0-9-]*", self.name):
            raise ValueError("name must use lowercase letters, numbers, or hyphens")
        if not isinstance(self.dataset, str) or self.dataset not in _INPUT_WIDTHS:
            raise ValueError("dataset must be digits or mnist")
        if (
            not isinstance(self.architecture, tuple)
            or len(self.architecture) < 3
            or any(type(width) is not int or width < 2 for width in self.architecture)
            or self.architecture[0] != _INPUT_WIDTHS[self.dataset]
            or self.architecture[-1] != 10
        ):
            raise ValueError("architecture must match the dataset input width and ten classes")
        if type(self.precision) is not int or not 1 <= self.precision <= 30:
            raise ValueError("precision must be an integer from 1 to 30")
        if (
            not isinstance(self.model_seeds, tuple)
            or not self.model_seeds
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.model_seeds)
            or len(set(self.model_seeds)) != len(self.model_seeds)
        ):
            raise ValueError("model_seeds must contain distinct 32-bit unsigned integers")
        if not isinstance(self.training_profile, str) or self.training_profile not in _TRAINING_PROFILES:
            raise ValueError(f"unknown training_profile: {self.training_profile}")
        if self.training_profile == "original-digits" and (
            self.dataset != "digits" or self.architecture != (64, 20, 20, 10) or self.precision != 9
        ):
            raise ValueError("the original Digits profile requires 64-20-20-10 at precision 9")
        if not isinstance(self.checkpoint_pattern, str) or not _CHECKPOINT_PATTERN.fullmatch(
            self.checkpoint_pattern
        ):
            raise ValueError("checkpoint_pattern must be an NPZ filename with a model_index field")
        if self.evaluation_split not in _EVALUATION_SPLITS[self.dataset]:
            raise ValueError(f"evaluation_split is invalid for {self.dataset}")
        if self.checkpoint_group is not None and (
            not isinstance(self.checkpoint_group, str) or self.checkpoint_group not in _CHECKPOINT_GROUPS
        ):
            raise ValueError(f"unknown checkpoint_group: {self.checkpoint_group}")

    @property
    def q(self) -> int:
        return 2**self.precision

    def checkpoint_path(self, checkpoint_dir: str | Path, model_index: int) -> Path:
        if type(model_index) is not int or not 1 <= model_index <= len(self.model_seeds):
            raise ValueError("model_index is outside the configured 1-based range")
        return Path(checkpoint_dir) / self.checkpoint_pattern.format(model_index=model_index)


@dataclass(frozen=True)
class ExtractionConfig:
    extractor: str
    oracle_backend: str
    trial_seeds: tuple[int, ...]
    w1_anchors: int
    w1_steps_per_anchor: int
    w1_queries: int
    b1_anchors: int
    b1_queries: int
    w2_max_anchors: int
    w2_steps_per_anchor: int
    w2_p1_queries: int
    w2_p2_queries: int
    w2_p3_queries: int
    b2_num_bases: int
    b2_test_queries: int
    last_layer_solver: str
    last_layer_repetitions: int
    track_bitwidth: bool = False
    rank1_consensus: bool = False
    b1_high_snr_filter: bool = False
    w2_walk_mode: str = "dense"
    w2_inv_backend: str = "sympy"
    bias_tolerances: tuple[int | None, int | None, int | None] | None = None

    def __post_init__(self) -> None:
        if self.extractor not in _EXTRACTORS:
            raise ValueError(f"unknown extractor: {self.extractor}")
        if self.oracle_backend not in _ORACLE_BACKENDS:
            raise ValueError(f"unknown oracle_backend: {self.oracle_backend}")
        if (
            not isinstance(self.trial_seeds, tuple)
            or not self.trial_seeds
            or any(type(seed) is not int or seed < 0 for seed in self.trial_seeds)
            or len(set(self.trial_seeds)) != len(self.trial_seeds)
        ):
            raise ValueError("trial_seeds must contain distinct nonnegative integers")
        for field_name in (
            "w1_anchors",
            "w1_steps_per_anchor",
            "w1_queries",
            "b1_anchors",
            "b1_queries",
            "w2_max_anchors",
            "w2_steps_per_anchor",
            "w2_p1_queries",
            "w2_p3_queries",
            "b2_num_bases",
            "b2_test_queries",
            "last_layer_repetitions",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if type(self.w2_p2_queries) is not int or self.w2_p2_queries < 0:
            raise ValueError("w2_p2_queries must be a nonnegative integer")
        if self.w2_walk_mode not in _WALK_MODES:
            raise ValueError(f"unknown w2_walk_mode: {self.w2_walk_mode}")
        if self.w2_inv_backend not in _W2_INV_BACKENDS:
            raise ValueError(f"unknown w2_inv_backend: {self.w2_inv_backend}")
        if self.last_layer_solver not in _LAST_LAYER_SOLVERS:
            raise ValueError(f"unknown last_layer_solver: {self.last_layer_solver}")
        if self.bias_tolerances is not None:
            if len(self.bias_tolerances) != 3 or any(
                tol is not None and (type(tol) is not int or tol < 0)
                for tol in self.bias_tolerances
            ):
                raise ValueError("bias_tolerances must be a 3-tuple of nonnegative ints or None")


@dataclass(frozen=True)
class BenchmarkConfig:
    target: TargetConfig
    extraction: ExtractionConfig


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    if "target" not in raw or "extraction" not in raw:
        raise ValueError(f"Configuration {config_path} must define [target] and [extraction] tables")

    target_fields = dict(raw["target"])
    for field in ("architecture", "model_seeds"):
        if isinstance(target_fields.get(field), list):
            target_fields[field] = tuple(target_fields[field])

    ext_fields = dict(raw["extraction"])
    for field in ("trial_seeds", "bias_tolerances"):
        if isinstance(ext_fields.get(field), list):
            ext_fields[field] = tuple(ext_fields[field])

    try:
        target = TargetConfig(**target_fields)
        extraction = ExtractionConfig(**ext_fields)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid configuration {config_path}: {error}") from error
    return BenchmarkConfig(target=target, extraction=extraction)


def load_target_config(path: str | Path) -> TargetConfig:
    return load_benchmark_config(path).target
