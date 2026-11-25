"""Plaintext fixed-point neural network oracle with probabilistic truncation."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .fixedpoint import INT64_MAX, signed_bit_width


class CPUOracleCounter:
    """Simulated PPML inference oracle implementing layerwise probabilistic truncation.

    Each affine layer computes the raw fixed-point product ``M = Y @ W_i`` and applies
    probabilistic truncation by ``q = 2^S``:

        trunc(M) = floor(M / q) + 1[(M mod q) > r],   r ~ Uniform{0, ..., q - 1},

    followed by adding the quantized bias ``B_i`` and (on hidden layers) ReLU.
    This matches the output distribution of standard MPC truncation protocols (e.g.,
    SecureML / ABY3) without ring wraparound, enforced via a conservative int64 bound.

    Implementation note on matrix-multiplication backends (Digits vs. MNIST):
      - ``backend="int64"`` performs exact 64-bit integer matrix multiplication. This
        preserves exact low-order bits even when pre-truncation accumulators exceed the
        53-bit IEEE-754 float64 mantissa limit (in the original 64-dim Digits benchmark,
        pre-truncation products reach ~2^55.3, requiring 57 signed bits).
      - ``backend="float64"`` casts inputs and weights to float64 to invoke BLAS DGEMM,
        delivering an ~8x speedup on high-dimensional 784-input MNIST query batches
        where NumPy's int64 matmul falls back to a scalar C loop. When intermediate
        products exceed 2^53, float64 rounds the least-significant bits before casting
        back to int64; at the macro probing scales used in the MNIST experiments this
        error is negligible relative to the probing step, while dramatically reducing
        benchmark wall-clock time.
    """

    def __init__(
        self,
        W_list: Sequence[np.ndarray],
        B_list: Sequence[np.ndarray],
        q: int,
        model_shape: Sequence[int],
        *,
        backend: str = "float64",
        track_bitwidth: bool = False,
    ) -> None:
        if backend not in ("int64", "float64"):
            raise ValueError(f"Unsupported oracle backend: {backend!r} (expected 'int64' or 'float64')")
        self.W = [np.asarray(w, dtype=np.int64) for w in W_list]
        self.W_float = [w.astype(np.float64) for w in self.W]
        self.B = [np.asarray(b, dtype=np.int64) for b in B_list]
        self.q = int(q)
        self.model_shape = tuple(int(dim) for dim in model_shape)
        self.backend = backend
        self.track_bitwidth = bool(track_bitwidth)
        self.total_queries = 0

        self._max_column_sums = [
            max(sum(abs(int(value)) for value in w[:, col]) for col in range(w.shape[1]))
            for w in self.W
        ]
        self._max_biases = [max(abs(int(value)) for value in b) for b in self.B]

        num_layers = len(self.W)
        self.max_input_abs = 0
        self.max_pre_trunc_abs = [0] * num_layers
        self.max_post_trunc_abs = [0] * num_layers

    def prob_oracle(self, X_batch: np.ndarray, sf_leak: bool = False):
        """Evaluate a query or batch of integer-encoded inputs under probabilistic truncation."""
        arr = np.asarray(X_batch)
        if arr.ndim == 1:
            self.total_queries += 1
            arr_2d = arr.reshape(1, -1)
        else:
            self.total_queries += int(arr.shape[0])
            arr_2d = arr

        if self.backend == "int64":
            Y_int = np.rint(arr_2d).astype(np.int64) if arr_2d.dtype.kind == "f" else arr_2d.astype(np.int64)
            if self.track_bitwidth and Y_int.size:
                in_mag = max(abs(int(Y_int.min())), abs(int(Y_int.max())))
                if in_mag > self.max_input_abs:
                    self.max_input_abs = in_mag
            return self._forward_int64(Y_int, sf_leak=sf_leak)

        Y_float = arr_2d.astype(np.float64)
        if self.track_bitwidth and Y_float.size:
            in_mag = int(round(max(abs(float(np.min(Y_float))), abs(float(np.max(Y_float))))))
            if in_mag > self.max_input_abs:
                self.max_input_abs = in_mag
        return self._forward_float64(Y_float, sf_leak=sf_leak)

    def _forward_int64(self, Y: np.ndarray, *, sf_leak: bool):
        leaked_sf = []
        num_layers = len(self.W)
        for i in range(num_layers):
            max_input = max(abs(int(Y.min())), abs(int(Y.max())))
            raw_bound = max_input * self._max_column_sums[i]
            if raw_bound > INT64_MAX:
                raise OverflowError(f"Layer {i + 1} raw product may exceed signed int64.")
            if (raw_bound + self.q - 1) // self.q + 1 + self._max_biases[i] > INT64_MAX:
                raise OverflowError(f"Layer {i + 1} output may exceed signed int64.")

            M = np.dot(Y, self.W[i])
            if self.track_bitwidth and M.size:
                pre_mag = max(abs(int(M.min())), abs(int(M.max())))
                if pre_mag > self.max_pre_trunc_abs[i]:
                    self.max_pre_trunc_abs[i] = pre_mag

            sf = np.random.randint(0, self.q, size=M.shape)
            if sf_leak:
                leaked_sf.append(sf)

            Y = M // self.q + (M % self.q > sf).astype(np.int64) + self.B[i]
            if i < num_layers - 1:
                Y = np.maximum(0, Y)

            if self.track_bitwidth and Y.size:
                post_mag = max(abs(int(Y.min())), abs(int(Y.max())))
                if post_mag > self.max_post_trunc_abs[i]:
                    self.max_post_trunc_abs[i] = post_mag

        if sf_leak:
            return Y, leaked_sf
        return Y

    def _forward_float64(self, Y: np.ndarray, *, sf_leak: bool):
        leaked_sf = []
        num_layers = len(self.W)
        for i in range(num_layers):
            max_input = max(abs(float(np.min(Y))), abs(float(np.max(Y))))
            raw_bound = max_input * self._max_column_sums[i]
            if raw_bound > INT64_MAX:
                raise OverflowError(f"Layer {i + 1} raw product may exceed signed int64.")
            if (raw_bound + self.q - 1) // self.q + 1 + self._max_biases[i] > INT64_MAX:
                raise OverflowError(f"Layer {i + 1} output may exceed signed int64.")

            M_float = np.dot(Y, self.W_float[i])
            M = np.round(M_float).astype(np.int64)
            if self.track_bitwidth and M.size:
                pre_mag = max(abs(int(M.min())), abs(int(M.max())))
                if pre_mag > self.max_pre_trunc_abs[i]:
                    self.max_pre_trunc_abs[i] = pre_mag

            sf = np.random.randint(0, self.q, size=M.shape)
            if sf_leak:
                leaked_sf.append(sf)

            Y_int = M // self.q + (M % self.q > sf).astype(np.int64) + self.B[i]
            if i < num_layers - 1:
                Y_int = np.maximum(0, Y_int)

            if self.track_bitwidth and Y_int.size:
                post_mag = max(abs(int(Y_int.min())), abs(int(Y_int.max())))
                if post_mag > self.max_post_trunc_abs[i]:
                    self.max_post_trunc_abs[i] = post_mag

            Y = Y_int.astype(np.float64)

        Y_out = np.round(Y).astype(np.int64)
        if sf_leak:
            return Y_out, leaked_sf
        return Y_out

    def bitwidth_report(self) -> dict:
        """Return observed maximum magnitudes and required signed bit widths per layer."""
        layers = []
        for idx, (pre_mag, post_mag) in enumerate(
            zip(self.max_pre_trunc_abs, self.max_post_trunc_abs, strict=True), start=1
        ):
            pre_bits = signed_bit_width(pre_mag)
            post_bits = signed_bit_width(post_mag)
            layers.append({
                "layer": idx,
                "pre_trunc_max_abs": int(pre_mag),
                "pre_trunc_log2": float(math.log2(pre_mag)) if pre_mag > 0 else 0.0,
                "pre_trunc_signed_bits": int(pre_bits),
                "pre_trunc_int64_headroom": int(64 - pre_bits),
                "post_trunc_max_abs": int(post_mag),
                "post_trunc_log2": float(math.log2(post_mag)) if post_mag > 0 else 0.0,
                "post_trunc_signed_bits": int(post_bits),
            })
        return {
            "input_max_abs": int(self.max_input_abs),
            "input_signed_bits": int(signed_bit_width(self.max_input_abs)),
            "layers": layers,
        }
