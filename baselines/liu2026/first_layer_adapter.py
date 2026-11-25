"""Isolate Liu's first-layer recovery from target-parameter diagnostics."""

from __future__ import annotations

import importlib
import sys
import time

import numpy as np

from oracle_bridge import AttackBudgetExceeded, build_liu_model


class UnsupportedFirstLayerFallback(RuntimeError):
    """The released fallback requires a true-sign decision and is excluded."""


def build_decoy_model(shape: tuple[int, ...], q: int, seed: int = 271828):
    """Give Liu an architecture carrier with non-target, reproducible weights."""
    rng = np.random.default_rng(seed)
    weights = [
        rng.integers(-q, q + 1, size=(shape[i], shape[i + 1]), dtype=np.int64)
        for i in range(len(shape) - 1)
    ]
    biases = [
        rng.integers(-q, q + 1, size=shape[i + 1], dtype=np.int64)
        for i in range(len(shape) - 1)
    ]
    return build_liu_model(weights, biases, q)


def install_first_layer_guards() -> list[str]:
    """Remove truth-dependent quality feedback from the first-layer path.

    The checker is labelled a cheating test upstream, but its returned
    zero-indices also filter precision points.  We retain every component
    point and use identity IDs strictly as bookkeeping.  The fallback using
    true signs is stopped if reached; it is not silently replaced by a
    different extraction algorithm.
    """
    quality_module = importlib.import_module(
        "signature_recovery.cluster_critical_points.merge_good_critical_points"
    )
    recovery_module = importlib.import_module("signature_recovery.signature_recovery")
    original_quality = quality_module.check_components_quality

    def oracle_only_quality(args, model, components, partial_rows, all_criticals, save_path):
        if args.layerID != 1 or args.real_attack != 1:
            raise ValueError("The oracle-only quality shim is limited to first-layer extraction")
        width = int(model.layers[1].output_shape[1])
        return list(range(width)), [[] for _ in components]

    patched: list[str] = []
    for name, module in tuple(sys.modules.items()):
        if name and name.startswith("signature_recovery") and module is not None:
            if getattr(module, "check_components_quality", None) is original_quality:
                setattr(module, "check_components_quality", oracle_only_quality)
                patched.append(name)

    def neutral_diagnostic_signs(args, model, recovered_weights, components, criticals):
        if args.layerID != 1 or args.real_attack != 1 or args.cheat_sign != 0:
            raise ValueError("True-sign diagnostics are disabled only in the oracle-only first-layer path")
        return np.ones(len(recovered_weights), dtype=np.int64)

    def stop_truth_sign_fallback(*args, **kwargs):
        raise UnsupportedFirstLayerFallback(
            "Liu's released component-rescue fallback uses true signs; this oracle-only run stops here"
        )

    recovery_module.cheat_on_sign = neutral_diagnostic_signs
    recovery_module.add_good_components_back = stop_truth_sign_fallback
    patched.extend([
        "signature_recovery.signature_recovery.cheat_on_sign",
        "signature_recovery.signature_recovery.add_good_components_back",
    ])
    return sorted(patched)


class ContinuousOracleBridge:
    """Exact float64 reference oracle for the same decoded integer weights."""

    def __init__(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        q: int,
        shape: tuple[int, ...],
        *,
        grid_input: bool = False,
        max_queries: int | None = None,
        max_seconds: float | None = None,
    ) -> None:
        self.weights = [np.asarray(w, dtype=np.float64) / q for w in weights]
        self.biases = [np.asarray(b, dtype=np.float64) / q for b in biases]
        self.q = q
        self.shape = shape
        self.grid_input = grid_input
        self.max_queries = max_queries
        self.deadline = time.monotonic() + max_seconds if max_seconds else None
        self.total_queries = 0
        self.logical_queries = 0
        self.max_abs_input_code = 0
        self._global_config = None

    def predict_manual_fast(self, x, weights, biases) -> np.ndarray:
        samples = np.asarray(x, dtype=np.float64)
        if samples.ndim == 1:
            samples = samples[None, :]
        if samples.ndim != 2 or samples.shape[1] != self.shape[0] or samples.shape[0] == 0:
            raise ValueError(f"Expected nonempty input batch with width {self.shape[0]}")
        if not np.all(np.isfinite(samples)):
            raise ValueError("Reference-oracle inputs must be finite")
        if self.max_queries is not None and self.total_queries + len(samples) > self.max_queries:
            raise AttackBudgetExceeded(f"Oracle query limit reached: {self.max_queries}")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise AttackBudgetExceeded("Wall-time limit reached before next oracle query")

        query_input = np.rint(samples * self.q) / self.q if self.grid_input else samples
        output = query_input
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            output = output @ weight + bias
            if i < len(self.weights) - 1:
                output = np.maximum(output, 0)

        self.total_queries += len(samples)
        self.logical_queries += len(samples)
        if self._global_config is not None:
            self._global_config.query_count += len(samples)
            if self._global_config.set_save_queries:
                self._global_config.SAVED_QUERIES.extend(zip(samples, output))
        return output
