"""Connect Liu et al.'s released attack code to this project's PT oracle.

Importing this module does not start an attack.  Call ``install_bridge`` after
importing Liu's ``model_extraction`` module and before invoking its ``main_``.
The bridge itself replaces the function used for external model queries.
Separate, disclosed compatibility fixes in the Liu checkout are documented
in INTEGRATION_REPORT.md.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC_ROOT = ROOT / "src"
LIU_ROOT = HERE / "third_party" / "liu_extraction"
for source_root in (SRC_ROOT, LIU_ROOT, HERE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from probtrunc_extraction.oracle import CPUOracleCounter  # noqa: E402


class AttackBudgetExceeded(RuntimeError):
    """The driver stopped an attack before exceeding a configured limit."""


def load_integer_target(path: str | Path) -> tuple[list[np.ndarray], list[np.ndarray], int, tuple[int, ...]]:
    """Read a saved integer target without creating or training a model."""
    with np.load(path, allow_pickle=False) as data:
        q = int(data["q"])
        count = int(data["num_layers"])
        weights = [np.asarray(data[f"W_{i}"], dtype=np.int64).copy() for i in range(count)]
        biases = [np.asarray(data[f"B_{i}"], dtype=np.int64).copy() for i in range(count)]
    if q <= 1 or len(weights) == 0:
        raise ValueError("Expected a positive fixed-point scale and at least one layer")
    shape = [weights[0].shape[0]]
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        if weight.ndim != 2 or bias.shape != (weight.shape[1],):
            raise ValueError(f"Invalid parameter dimensions at layer {i + 1}")
        if weight.shape[0] != shape[-1]:
            raise ValueError(f"Disconnected layer {i + 1}")
        shape.append(weight.shape[1])
    return weights, biases, q, tuple(shape)


def build_liu_model(weights: list[np.ndarray], biases: list[np.ndarray], q: int):
    """Make the parameter container expected by Liu's original attack.

    Liu uses the Keras model to retrieve weights and layer sizes; its query
    function is intercepted by ``install_bridge``.  This container represents
    the continuous network with the same quantized integer parameters.
    """
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(weights[0].shape[0],), dtype="float64")
    output = inputs
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        layer = tf.keras.layers.Dense(
            weight.shape[1],
            activation="relu" if i < len(weights) - 1 else "linear",
            dtype="float64",
            name=f"dense_{i + 1}",
        )
        output = layer(output)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        layer = model.layers[i + 1]
        layer.set_weights([weight.astype(np.float64) / q, bias.astype(np.float64) / q])
        # Liu reads Dense.output_shape; Keras 3 no longer provides it.
        if not hasattr(layer, "output_shape"):
            layer.output_shape = (None, weight.shape[1])
    return model


class ProbTruncBridge:
    """Map Liu's floating queries to integer queries, preserving oracle RNG."""

    def __init__(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        q: int,
        shape: tuple[int, ...],
        *,
        noise_seed: int,
        repeats: int = 1,
        max_queries: int | None = None,
        max_seconds: float | None = None,
    ) -> None:
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        if max_queries is not None and max_queries < 1:
            raise ValueError("max_queries must be positive")
        if max_seconds is not None and max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        self.q = q
        self.shape = shape
        self.oracle = CPUOracleCounter(weights, biases, q, shape)
        self._noise_state = np.random.RandomState(noise_seed).get_state()
        self._global_config = None
        self.max_abs_input_code = 0
        self.repeats = repeats
        self.logical_queries = 0
        self._weights = weights
        self._biases = biases
        self.max_queries = max_queries
        self.deadline = time.monotonic() + max_seconds if max_seconds is not None else None

    def validate_model(self, model) -> None:
        """Reject a Keras container whose parameters differ from the oracle."""
        from signature_recovery.utils import getAllWeightsAndBiases

        model_weights, model_biases = getAllWeightsAndBiases(model)
        if len(model_weights) != len(self._weights):
            raise ValueError("Liu model and PT oracle have different layer counts")
        for i, (weight, bias) in enumerate(zip(model_weights, model_biases)):
            if not np.array_equal(weight, self._weights[i].astype(np.float64) / self.q):
                raise ValueError(f"Liu model weight mismatch at layer {i + 1}")
            if not np.array_equal(bias, self._biases[i].astype(np.float64) / self.q):
                raise ValueError(f"Liu model bias mismatch at layer {i + 1}")

    def predict_manual_fast(self, x, weights, biases) -> np.ndarray:
        """Drop-in replacement for Liu's external-query function.

        ``weights`` and ``biases`` are Liu's existing call signature; target
        parameters are held by the integer oracle.  ``x`` is rounded to the
        target's q-grid.  Independent oracle responses for each encoded input
        are averaged before decoding to Liu's floating scale.  The supplied
        ``weights`` and ``biases`` are never used to produce an oracle output.
        """
        samples = np.asarray(x, dtype=np.float64)
        if samples.ndim == 1:
            samples = samples[None, :]
        if samples.ndim != 2 or samples.shape[1] != self.shape[0]:
            raise ValueError(f"Expected input shape (n, {self.shape[0]})")
        if samples.shape[0] == 0:
            raise ValueError("Oracle input batch must be nonempty")
        if not np.all(np.isfinite(samples)):
            raise ValueError("Oracle inputs must be finite")
        required_queries = samples.shape[0] * self.repeats
        if self.max_queries is not None and self.oracle.total_queries + required_queries > self.max_queries:
            raise AttackBudgetExceeded(f"Oracle query limit reached: {self.max_queries}")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise AttackBudgetExceeded("Wall-time limit reached before next oracle query")
        scaled = np.rint(samples * self.q)
        if np.any(np.abs(scaled) >= 2**63):
            raise OverflowError("Encoded oracle input exceeds signed int64")
        codes = scaled.astype(np.int64)
        if codes.size:
            self.max_abs_input_code = max(self.max_abs_input_code, int(np.max(np.abs(codes))))

        # CPUOracleCounter uses np.random.randint.  Give the oracle an
        # independent stream so its truncation draws do not alter Liu's
        # random search directions.
        attack_state = np.random.get_state()
        physical_before = self.oracle.total_queries
        np.random.set_state(self._noise_state)
        try:
            total = None
            for _ in range(self.repeats):
                if self.deadline is not None and time.monotonic() >= self.deadline:
                    raise AttackBudgetExceeded("Wall-time limit reached before next oracle query")
                response = np.asarray(self.oracle.prob_oracle(codes), dtype=np.float64)
                if total is None:
                    total = response
                else:
                    total += response
        finally:
            self._noise_state = np.random.get_state()
            np.random.set_state(attack_state)
            if self._global_config is not None:
                self._global_config.query_count += self.oracle.total_queries - physical_before

        decoded = total / (self.q * self.repeats)
        self.logical_queries += samples.shape[0]
        if self._global_config is not None:
            if self._global_config.set_save_queries:
                self._global_config.SAVED_QUERIES.extend(zip(samples, decoded))
        return decoded


def install_bridge(bridge: ProbTruncBridge) -> list[str]:
    """Patch all *loaded* Liu modules that imported the query function.

    Import ``model_extraction`` first; it loads the attack modules.  We only
    replace references identical to the upstream query function, and never
    alter the upstream files.  The returned module list is an audit record.
    """
    import global_vars
    from signature_recovery import utils

    original = utils.predict_manual_fast
    if getattr(original, "__module__", None) != "signature_recovery.utils":
        raise RuntimeError("Liu query function was already replaced")

    patched = []
    for name, module in tuple(sys.modules.items()):
        if not name or module is None:
            continue
        if name != "model_extraction" and not name.startswith((
            "signature_recovery", "sign_recovery", "blackbox_src"
        )):
            continue
        if getattr(module, "predict_manual_fast", None) is original:
            setattr(module, "predict_manual_fast", bridge.predict_manual_fast)
            patched.append(name)
    required = {"signature_recovery.utils", "signature_recovery.critical_point_search", "model_extraction"}
    if not required.issubset(patched):
        raise RuntimeError(f"Liu modules missing at bridge installation: {sorted(required - set(patched))}")
    bridge._global_config = global_vars.GlobalConfig
    return sorted(patched)
