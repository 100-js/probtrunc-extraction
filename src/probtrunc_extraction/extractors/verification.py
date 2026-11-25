"""Permutation-aligned exact integer verification for extracted MLP parameters."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def unshuffle_and_verify_L1(
    W1_ext: np.ndarray,
    B1_ext: np.ndarray,
    W1_true: np.ndarray,
    B1_true: np.ndarray,
    W2_true: np.ndarray,
    *,
    tolerance: int | None = 1,
    weight_tolerance: int = 0,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[int, int] | None]:
    """Align extracted L1 neurons to ground truth, enforcing exact weights (tol=0).

    When ``tolerance`` is ``None``, alignment requires exact weight equality only
    while still returning aligned biases so the caller can report observed LSB errors.
    """
    _, d_h = W1_true.shape
    pi_1: dict[int, int] = {}
    has_forced_live_match = False

    for ext_idx in range(d_h):
        for true_idx in range(d_h):
            if true_idx in pi_1.values():
                continue

            diff_W = np.max(np.abs(W1_ext[:, ext_idx] - W1_true[:, true_idx]))
            diff_B = np.abs(B1_ext[ext_idx] - B1_true[true_idx])

            bias_ok = True if tolerance is None else (diff_B <= tolerance)
            if diff_W <= weight_tolerance and bias_ok:
                pi_1[ext_idx] = true_idx
                break

    unmatched_ext = [i for i in range(d_h) if i not in pi_1]
    unmatched_true = [i for i in range(d_h) if i not in pi_1.values()]

    tol_str = "unbounded" if tolerance is None else f"<={tolerance}"
    print(
        f"Equivalence check complete. (Exact matches within weight_tol<={weight_tolerance}, "
        f"bias_tol {tol_str}: {len(pi_1)}/{d_h})"
    )

    live_true_neurons = []
    dead_true_neurons = []
    for true_idx in unmatched_true:
        if np.all(W2_true[true_idx, :] == 0):
            dead_true_neurons.append(true_idx)
        else:
            live_true_neurons.append(true_idx)

    if live_true_neurons:
        has_forced_live_match = True
        cost_matrix = np.zeros((len(live_true_neurons), len(unmatched_ext)))
        for i, true_idx in enumerate(live_true_neurons):
            for j, ext_idx in enumerate(unmatched_ext):
                err_W = np.max(np.abs(W1_ext[:, ext_idx] - W1_true[:, true_idx]))
                err_B = np.abs(B1_ext[ext_idx] - B1_true[true_idx])
                cost_matrix[i, j] = err_W + err_B

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_ext_indices = []
        for i, j in zip(row_ind, col_ind):
            pi_1[unmatched_ext[j]] = live_true_neurons[i]
            matched_ext_indices.append(unmatched_ext[j])
        unmatched_ext = [x for x in unmatched_ext if x not in matched_ext_indices]

    if dead_true_neurons:
        print("Arbitrary assignment for unresolved inactive (zero-weight) neurons.")
        for true_idx in dead_true_neurons:
            if not unmatched_ext:
                break
            pi_1[unmatched_ext.pop(0)] = true_idx

    W1_recon = np.zeros_like(W1_true)
    B1_recon = np.zeros_like(B1_true)
    for ext_i, true_i in pi_1.items():
        W1_recon[:, true_i] = W1_ext[:, ext_i]
        B1_recon[true_i] = B1_ext[ext_i]

    if has_forced_live_match:
        print(
            "Error: Optimal assignment deviation exceeds tolerance threshold for active neurons. "
            "Extraction aborted."
        )
        return None, None, None

    print("Layer 1 structural recovery complete. Functional equivalence verified.")
    return W1_recon, B1_recon, pi_1


def unshuffle_and_verify_W2(
    W2_ext: np.ndarray,
    W2_true: np.ndarray,
    found_pi_1: dict[int, int],
) -> tuple[np.ndarray | None, dict[int, int] | None]:
    """Unshuffle W2 rows via ``found_pi_1`` and verify exact integer column recovery."""
    _, d_h2 = W2_true.shape
    has_forced_match = False
    W2_row_aligned = np.zeros_like(W2_ext)
    for ext_i, true_i in found_pi_1.items():
        W2_row_aligned[true_i, :] = W2_ext[ext_i, :]

    cost_matrix = np.zeros((d_h2, d_h2))
    for true_j in range(d_h2):
        for ext_j in range(d_h2):
            cost_matrix[true_j, ext_j] = np.max(np.abs(W2_row_aligned[:, ext_j] - W2_true[:, true_j]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    pi_2: dict[int, int] = {}
    for true_j, ext_j in zip(row_ind, col_ind):
        pi_2[int(ext_j)] = int(true_j)
        if cost_matrix[true_j, ext_j] != 0:
            has_forced_match = True

    W2_recon = np.zeros_like(W2_true)
    for ext_j, true_j in pi_2.items():
        W2_recon[:, true_j] = W2_row_aligned[:, ext_j]

    if has_forced_match or not np.array_equal(W2_recon, W2_true):
        print("Error: Sub-optimal assignment detected (Deviation > 0). W2 recovery aborted.")
        return None, None

    print("W2 functional recovery and verification complete.")
    return W2_recon, pi_2


def unshuffle_and_verify_B2(
    B2_ext: np.ndarray,
    B2_true: np.ndarray,
    found_pi_2: dict[int, int],
    *,
    tolerance: int | None = 1,
) -> np.ndarray | None:
    """Align extracted B2 biases via ``found_pi_2`` and check optional LSB tolerance."""
    d_h2 = len(B2_true)
    B2_recon = np.zeros(d_h2, dtype=np.int64)
    for ext_i, true_i in found_pi_2.items():
        B2_recon[true_i] = B2_ext[ext_i]

    max_error = int(np.max(np.abs(B2_recon - B2_true)))
    if tolerance is None or max_error <= tolerance:
        tol_str = "unbounded" if tolerance is None else str(tolerance)
        print(f"B2 functional recovery verified within tolerance {tol_str} (Maximum deviation: {max_error}).")
        return B2_recon

    print(f"Warning: B2 deviation exceeds tolerance ({max_error} > {tolerance}).")
    return None


def unshuffle_and_verify_L_last(
    W_last_ext: np.ndarray,
    B_last_ext: np.ndarray,
    W_last_true: np.ndarray,
    B_last_true: np.ndarray,
    found_pi_last: dict[int, int],
    *,
    tolerance: int | None = 1,
) -> tuple[bool, np.ndarray, np.ndarray]:
    """Verify exact final-layer weight recovery (0 error) and optional bias LSB tolerance."""
    W_last_recon = np.zeros_like(W_last_true)
    for ext_i, true_i in found_pi_last.items():
        W_last_recon[true_i, :] = W_last_ext[ext_i, :]

    w_match = bool(np.array_equal(W_last_recon, W_last_true))
    max_error = int(np.max(np.abs(B_last_ext - B_last_true)))
    b_match = True if tolerance is None else (max_error <= tolerance)

    if w_match and b_match:
        print(f"Final layer parameters successfully extracted. (Maximum bias deviation: {max_error})")
        return True, W_last_recon, B_last_ext

    print(
        f"Warning: Final layer parameter discrepancy detected "
        f"(weight_exact={w_match}, max_bias_err={max_error}, tolerance={tolerance})."
    )
    return False, W_last_recon, B_last_ext
