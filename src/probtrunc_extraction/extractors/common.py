"""Shared algebraic primitives, submatrix basis selectors, and final-layer solvers."""

from __future__ import annotations

import math
import random
from typing import Callable, Sequence

import numpy as np
import scipy.stats as stats
import sympy as sp

try:
    from fpylll import IntegerMatrix, LLL
except ImportError:
    IntegerMatrix = None
    LLL = None


def set_seed(seed: int = 42) -> None:
    """Seed both NumPy legacy RNG and Python's stdlib RNG for reproducible trials."""
    np.random.seed(seed)
    random.seed(seed)


def primitive_vector(diff: np.ndarray) -> np.ndarray | None:
    """Reduce a non-zero integer vector by its GCD with canonical positive leading sign."""
    vec = np.asarray(diff)
    if np.all(vec == 0):
        return None

    if vec[np.argmax(vec != 0)] < 0:
        vec = -vec

    gcd_val = abs(int(vec[0]))
    for val in vec[1:]:
        gcd_val = math.gcd(gcd_val, abs(int(val)))
        if gcd_val == 1:
            break

    return vec // gcd_val


def find_rank_full_sub_indices(
    matrix: np.ndarray,
    target_size: int,
    *,
    method: str = "qr",
) -> list[int]:
    """Select ``target_size`` linearly independent row indices of ``matrix``.

    Paper vs. Implementation note (Square Submatrix Inverse vs. Pseudoinverse):
      While the theoretical paper formulation writes a Moore-Penrose pseudoinverse
      over all ``D_in`` input coordinates, inverting the full overdetermined matrix
      via SVD can amplify tiny singular values caused by nearly dependent rows.
      Instead, all extractors select a well-conditioned ``target_size x target_size``
      square submatrix ``W_sub = W[sub_indices, :]`` and invert ``W_sub`` directly,
      zeroing the remaining ``D_in - target_size`` input coordinates.

    Digits vs. MNIST note (Row-Selection Strategy):
      - ``method="greedy"`` reproduces the original 64-dim Digits W2/B2/L3 row
        selection, scanning rows ``0..D_in-1`` sequentially until full rank is met.
      - ``method="qr"`` upgrades selection to column-pivoted QR decomposition
        (``scipy.linalg.qr(matrix.T, pivoting=True)``) for 784-dim MNIST networks,
        where border pixels are nearly zero and sequential greedy selection would
        pick severely ill-conditioned rows that inflate inverse norms and cause
        int64 overflow.
    """
    if method not in ("qr", "greedy"):
        raise ValueError(f"Unsupported row-selection method: {method!r} (expected 'qr' or 'greedy')")

    if method == "qr":
        try:
            from scipy.linalg import qr

            _, _, piv = qr(matrix.T.astype(np.float64), pivoting=True)
            sub_candidate = sorted(int(idx) for idx in piv[:target_size])
            if np.linalg.matrix_rank(matrix[sub_candidate, :]) == target_size:
                return sub_candidate
        except Exception:
            pass

    sub_indices: list[int] = []
    for i in range(matrix.shape[0]):
        test_indices = sub_indices + [i]
        if np.linalg.matrix_rank(matrix[test_indices, :]) == len(test_indices):
            sub_indices.append(i)
        if len(sub_indices) == target_size:
            break

    if len(sub_indices) < target_size:
        raise ValueError("Insufficient matrix rank to construct a full-rank sub-matrix.")
    return sub_indices


def find_max_det_sub_indices(
    matrix: np.ndarray,
    target_size: int,
    *,
    search_attempts: int = 1000,
) -> list[int]:
    """Randomly sample ``search_attempts`` full-rank square submatrices and return max |det|.

    Digits vs. MNIST note:
      Used in ``original_digits.decrypt_L1`` to select a near-orthogonal 20x20 basis
      out of 64 rows. For 784-dim MNIST (especially 784x128), 1,000 random rank loops
      are prohibitively slow, so ``mnist.py`` replaces this with deterministic
      column-pivoted QR via ``find_rank_full_sub_indices(..., method="qr")``.
    """
    d_in = matrix.shape[0]
    best_logdet = -np.inf
    best_sub_indices: list[int] = []

    while True:
        for _ in range(search_attempts):
            perm = np.random.permutation(d_in)
            temp_indices: list[int] = []
            for i in perm:
                test_indices = temp_indices + [int(i)]
                if np.linalg.matrix_rank(matrix[test_indices, :]) == len(test_indices):
                    temp_indices.append(int(i))
                if len(temp_indices) == target_size:
                    break

            if len(temp_indices) == target_size:
                _, logdet = np.linalg.slogdet(matrix[temp_indices, :])
                if logdet > best_logdet:
                    best_logdet = logdet
                    best_sub_indices = sorted(temp_indices)

        if best_sub_indices:
            return best_sub_indices


def _simulate_prefix_activations(
    X_batch: np.ndarray,
    W_list: Sequence[np.ndarray],
    B_list: Sequence[np.ndarray],
    q: int,
) -> np.ndarray:
    """Forward-propagate ``X_batch`` through already recovered prefix layers."""
    activations = X_batch.copy()
    for weight, bias in zip(W_list, B_list, strict=True):
        raw = np.dot(activations, weight)
        activations = np.round(raw / q) + bias
        activations = np.maximum(0, activations)
    return activations


def solve_last_layer_lll(
    oracle_func: Callable,
    W_list: Sequence[np.ndarray],
    B_list: Sequence[np.ndarray],
    D_out: int,
    q: int,
    *,
    sub_indices_method: str = "qr",
    last_layer_repetitions: int = 1,
    scale_target_with_q: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover final-layer integer weights via LLL and biases via residual mode.

    Paper vs. Implementation note (Demand-Driven Routing & Two-Stage Recovery):
      1. Traditional linear inversion does not enforce ReLU non-negativity across
         preceding hidden layers, causing target activations to be clipped to zero.
         To construct a full-rank set of pre-output hidden activations ``A_prev``,
         we use 'Demand-Driven' positive-path routing: for each hidden neuron ``j``,
         we compute the activation demand needed to overcome negative biases,
         backpropagate strictly along positive-weight paths, and flood those paths
         with large activations. We combine these ``D_hidden`` adversarial inputs
         with 10 small random inputs (``m = D_hidden + 10`` total equations).
      2. Weight and bias extraction is two-stage: LLL lattice reduction solves the
         Bounded Distance Decoding (BDD) instance for exact integer weights
         ``W_last`` first; given ``W_last``, each bias ``B_last[j]`` is resolved by
         taking the coordinate-wise mode of output residuals ``Y_out - round(A_prev @ w / q)``.
      3. When ``fpylll`` is unavailable (e.g., on Windows without native GMP/MPFR),
         the solver transparently falls back to ``sympy.Matrix.lll()``.

    Digits vs. MNIST note (Last-Layer Repetition Policy):
      In 64-dim Digits, accumulated truncation variance is small enough that
      single-shot queries (``last_layer_repetitions = 1``, 30 total queries) keep
      residual bias modes within 1 LSB. In 784-dim MNIST, accumulated noise grows
      with ``sqrt(d_0)``, so averaging ``last_layer_repetitions = 16`` (480 queries)
      suppresses Bernoulli carry jitter in the residual mode while remaining <0.02%
      of total query cost.
    """
    d_in = W_list[0].shape[0]
    d_h1 = W_list[0].shape[1]
    d_hidden = W_list[-1].shape[1]

    W1 = W_list[0]
    B1 = B_list[0]
    sub_indices = find_rank_full_sub_indices(W1, d_h1, method=sub_indices_method)
    W1_inv_float = np.linalg.inv(W1[sub_indices, :])

    target_amp = 5000.0 * max(1.0, q / 512.0) if scale_target_with_q else 5000.0
    adv_X_list = []
    for j in range(d_hidden):
        target_H = np.zeros(d_hidden, dtype=np.float64)
        target_H[j] = target_amp

        for k in range(len(W_list) - 1, 0, -1):
            W_k = W_list[k]
            B_k = B_list[k]
            M_req = np.zeros(W_k.shape[1], dtype=np.float64)
            for n in range(len(target_H)):
                if target_H[n] > 0:
                    M_req[n] = max(100.0, target_H[n] - B_k[n])
            influence = np.dot(W_k, M_req)
            target_H = np.where(influence > 0, 10000.0, 0.0)

        M1_tgt = np.zeros(d_h1, dtype=np.float64)
        for i in range(d_h1):
            if target_H[i] > 0:
                M1_tgt[i] = (target_H[i] - B1[i]) * q
            else:
                M1_tgt[i] = -10000.0

        X_sub = np.dot(M1_tgt, W1_inv_float)
        X_adv = np.zeros(d_in, dtype=np.int64)
        X_adv[sub_indices] = np.round(X_sub).astype(np.int64)
        adv_X_list.append(X_adv)

    m_eqs_random = 10
    X_random = np.random.randint(-1000, 1000, size=(m_eqs_random, d_in)).astype(np.int64)
    X_batch = np.vstack([X_random, np.array(adv_X_list)])
    m_eqs_total = X_batch.shape[0]

    if last_layer_repetitions > 1:
        X_rep = np.repeat(X_batch, last_layer_repetitions, axis=0)
        Y_rep = oracle_func(X_rep, sf_leak=False)
        Y_out = np.round(
            np.mean(Y_rep.reshape(m_eqs_total, last_layer_repetitions, -1), axis=1)
        ).astype(np.int64)
    else:
        Y_out = oracle_func(X_batch, sf_leak=False)

    A_prev = _simulate_prefix_activations(X_batch, W_list, B_list, q)
    A_aug = np.hstack([A_prev, np.ones((m_eqs_total, 1), dtype=np.int64) * q])
    n_vars = A_aug.shape[1]

    W_ext = np.zeros((d_hidden, D_out), dtype=np.int64)
    B_ext = np.zeros(D_out, dtype=np.int64)
    S_weight = 1
    Wt = 10000

    print("Initiating last layer extraction (LLL)...")
    use_fpylll = IntegerMatrix is not None and LLL is not None
    for j in range(D_out):
        T = Y_out[:, j] * q
        if use_fpylll:
            mat = IntegerMatrix(n_vars + 1, n_vars + 1 + m_eqs_total)
        else:
            mat = sp.zeros(n_vars + 1, n_vars + 1 + m_eqs_total)

        for i in range(n_vars):
            mat[i, i] = 1
            for k in range(m_eqs_total):
                mat[i, n_vars + 1 + k] = S_weight * int(A_aug[k, i])

        mat[n_vars, n_vars] = Wt
        for k in range(m_eqs_total):
            mat[n_vars, n_vars + 1 + k] = S_weight * int(-T[k])

        if use_fpylll:
            LLL.reduction(mat)
            nrows = mat.nrows
        else:
            mat = mat.lll()
            nrows = mat.rows

        found = False
        sol = None
        for i in range(nrows):
            if mat[i, n_vars] == Wt:
                sol = [int(mat[i, k]) for k in range(n_vars)]
                found = True
                break
            if mat[i, n_vars] == -Wt:
                sol = [-int(mat[i, k]) for k in range(n_vars)]
                found = True
                break

        if found and sol is not None:
            W_ext[:, j] = sol[:-1]
            M_last = np.dot(A_prev, W_ext[:, j])
            Y_base = (M_last + (q // 2)) // q
            B_candidates = Y_out[:, j] - Y_base
            B_ext[j] = int(stats.mode(B_candidates, keepdims=False).mode)
        else:
            print(f"Error: Class {j:02d} LLL inversion failed.")

    print("Last layer extraction complete.")
    return W_ext, B_ext


def solve_last_layer_flooded_pinv(
    oracle_func: Callable,
    W_list: Sequence[np.ndarray],
    B_list: Sequence[np.ndarray],
    D_out: int,
    q: int,
    *,
    sub_indices_method: str = "qr",
    last_layer_repetitions: int = 64,
    scales: Sequence[float] = (1.0, 2.0, 3.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Recover final-layer weights and biases via Multi-Scale Flooded Linear Inversion.

    Digits vs. MNIST note (Why Flooded Linear Inversion Replaces LLL at S=12 & Wide S=6):
      For high-precision targets (``S=12, q=4096``) or wide penultimate layers
      (``784-128-64-10`` with ``D_hidden=64``), forming and reducing a high-dimensional
      LLL basis under amplified ``q``-scaled noise is slow and sensitive to lattice
      parameters. Instead, we construct an overdetermined system of ``3 * D_hidden``
      demand-driven inputs across three activation flood scales ``[1.0, 2.0, 3.0]``
      (``target_H = 50,000 * scale``), average ``last_layer_repetitions`` (default 64)
      oracle evaluations per equation to drive truncation variance well below 0.5 LSB,
      solve directly via ``np.linalg.pinv(A_aug) @ (Y_out * q)`` + rounding for exact
      integer ``W_last``, and resolve ``B_last`` via the coordinate-wise residual mode.
    """
    d_in = W_list[0].shape[0]
    d_h1 = W_list[0].shape[1]
    d_hidden = W_list[-1].shape[1]

    W1 = W_list[0]
    B1 = B_list[0]
    sub_indices = find_rank_full_sub_indices(W1, d_h1, method=sub_indices_method)
    W1_inv_float = np.linalg.inv(W1[sub_indices, :].astype(np.float64))

    print("Initiating last layer extraction via Flooded Linear Inversion...")
    adv_X_list = []
    for scale in scales:
        for j in range(d_hidden):
            target_H = np.zeros(d_hidden, dtype=np.float64)
            target_H[j] = 50000.0 * scale

            for k in range(len(W_list) - 1, 0, -1):
                W_k = W_list[k]
                B_k = B_list[k]
                M_req = np.zeros(W_k.shape[1], dtype=np.float64)
                for n in range(len(target_H)):
                    if target_H[n] > 0:
                        M_req[n] = max(100.0, target_H[n] - B_k[n])
                influence = np.dot(W_k, M_req)
                target_H = np.where(influence > 0, 100000.0 * scale, 0.0)

            M1_tgt = np.zeros(d_h1, dtype=np.float64)
            for i in range(d_h1):
                if target_H[i] > 0:
                    M1_tgt[i] = (target_H[i] - B1[i]) * q
                else:
                    M1_tgt[i] = -100000.0 * scale

            X_sub = np.dot(M1_tgt, W1_inv_float)
            X_adv = np.zeros(d_in, dtype=np.int64)
            X_adv[sub_indices] = np.round(X_sub).astype(np.int64)
            adv_X_list.append(X_adv)

    X_batch = np.array(adv_X_list)
    m_eqs_total = X_batch.shape[0]

    if last_layer_repetitions > 1:
        X_rep = np.repeat(X_batch, last_layer_repetitions, axis=0)
        Y_rep = oracle_func(X_rep, sf_leak=False)
        Y_out = np.round(
            np.mean(Y_rep.reshape(m_eqs_total, last_layer_repetitions, -1), axis=1)
        ).astype(np.int64)
    else:
        Y_out = oracle_func(X_batch, sf_leak=False)

    A_prev = _simulate_prefix_activations(X_batch, W_list, B_list, q)
    A_aug = np.hstack([A_prev, np.ones((m_eqs_total, 1), dtype=np.int64) * q])

    pinv = np.linalg.pinv(A_aug.astype(np.float64))
    sol_float = np.dot(pinv, (Y_out * q).astype(np.float64))
    W_ext = np.round(sol_float[:d_hidden, :]).astype(np.int64)

    B_ext = np.zeros(D_out, dtype=np.int64)
    for j in range(D_out):
        M_last = np.dot(A_prev, W_ext[:, j])
        B_candidates = Y_out[:, j] - np.round(M_last / q).astype(np.int64)
        B_ext[j] = int(stats.mode(B_candidates, keepdims=False).mode)

    print("Last layer extraction complete.")
    return W_ext, B_ext
