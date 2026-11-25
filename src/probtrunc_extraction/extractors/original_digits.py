"""Original 64-20-20-10 (S=9) Digits model extraction implementation.

This module faithfully preserves the extraction pipeline used for the original
50-trial 64-20-20-10 benchmark (Section 5 / Table 1, Table 2 SIR separation, and
Table 3 signed bit-width analysis), while delegating shared algebraic primitives
and cross-platform LLL reduction (with automatic SymPy fallback) to ``common.py``.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import scipy.stats as stats
import sympy as sp
from scipy.optimize import linprog

from .common import (
    find_max_det_sub_indices,
    find_rank_full_sub_indices,
    primitive_vector,
    solve_last_layer_lll,
)


def decrypt_L1(
    oracle_func: Callable,
    D_in: int,
    D_h: int,
    D_out: int,
    q: int,
    total_layers: int,
    *,
    W1_anchors: int = 1000,
    W1_steps_per_anchor: int = 10,
    W1_queries: int = 1,
    B1_anchors: int = 10,
    B1_queries: int = 500,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Recover first-layer integer weights W1 and biases B1 for the original 64-dim benchmark."""
    print("Initiating W1 extraction...")

    delta_x = 1e10
    N_queries = W1_queries
    extracted_W1_raw: list[np.ndarray] = []

    # Digits vs. MNIST note:
    # The original 64-dim extractor projects output logits onto the all-ones vector
    # and queries coordinate perturbations one by one. In 784-dim MNIST, mnist.py
    # instead sends all 1 + D_in queries in a single BLAS batch and inspects each
    # output logit coordinate separately.
    def get_macro_grad_X(X_anchor: np.ndarray, proj_vec: np.ndarray) -> np.ndarray:
        X_rep = np.tile(X_anchor, (N_queries, 1))
        mean_base = np.mean(np.dot(oracle_func(X_rep), proj_vec))

        grad = np.zeros(D_in, dtype=np.int64)
        for i in range(D_in):
            X_step = X_anchor.copy()
            X_step[i] += delta_x
            mean_step = np.mean(np.dot(oracle_func(np.tile(X_step, (N_queries, 1))), proj_vec))
            grad[i] = int(np.round(((mean_step - mean_base) / delta_x) * (q**total_layers)))
        return grad

    for _ in range(W1_anchors):
        if len(extracted_W1_raw) >= D_h:
            break

        X_walk = np.floor(np.random.randn(D_in) * 1e13).astype(float)
        V_dir = np.floor(np.random.randn(D_in) * 1e11).astype(float)
        PROJ_VEC = np.random.choice([1], size=D_out)
        prev_G = get_macro_grad_X(X_walk, PROJ_VEC)

        for _ in range(W1_steps_per_anchor):
            if len(extracted_W1_raw) >= D_h:
                break

            X_walk += V_dir
            current_G = get_macro_grad_X(X_walk, PROJ_VEC)
            diff = current_G - prev_G

            primitive = primitive_vector(diff)
            # Paper vs. Implementation note (Magnitude Plausibility Filter):
            # Rejects composite multi-neuron jumps or noisy GCD reductions whose
            # primitive entries fall outside plausible quantized weight magnitudes.
            if primitive is not None and 0.02 * q < np.max(np.abs(primitive)) < 1.5 * q:
                is_new = True
                for ex in extracted_W1_raw:
                    if np.array_equal(primitive, ex) or np.array_equal(primitive, -ex):
                        is_new = False
                        break
                if is_new:
                    extracted_W1_raw.append(primitive)

            prev_G = current_G

    if len(extracted_W1_raw) < D_h:
        print(f"Error: Insufficient W1 neurons extracted ({len(extracted_W1_raw)}/{D_h}). Rank deficient.")
        return None, None

    extracted_W1_raw_mat = np.array(extracted_W1_raw).T

    # Paper vs. Implementation note (Max-Det Square Submatrix vs. Pseudoinverse):
    # Instead of computing a full SVD pseudoinverse over all D_in rows (which risks
    # amplifying tiny singular values from nearly dependent rows), we search 1,000
    # random full-rank square submatrices and invert the maximum-volume (max |det|)
    # submatrix. For 784-dim MNIST, mnist.py replaces this random search with
    # deterministic column-pivoted QR for speed and conditioning.
    sub_indices = find_max_det_sub_indices(extracted_W1_raw_mat, D_h, search_attempts=1000)
    W_sub = extracted_W1_raw_mat[sub_indices, :]
    W_inv = np.linalg.inv(W_sub)

    print("Initiating W1 sign & B1 extraction...")
    extracted_W1_final = np.zeros_like(extracted_W1_raw_mat)
    extracted_B1_final = np.zeros(D_h, dtype=np.int64)

    step_unit = 1e6
    k_vals = [-2, -1, 1, 2]

    for i in range(D_h):
        raw_w1 = extracted_W1_raw_mat[:, i]
        candidate_B1s = []
        candidate_signs = []

        for _ in range(B1_anchors):
            M1_step_tgt = np.zeros(D_h)
            M1_step_tgt[i] = step_unit
            J_i_sub = np.dot(M1_step_tgt, W_inv)
            J_i = np.zeros(D_in, dtype=np.int64)
            J_i[sub_indices] = np.round(J_i_sub).astype(np.int64)

            M1_base_tgt = np.random.choice([-1e10, 1e10], size=D_h)
            M1_base_tgt[i] = 0.0
            X_base_sub = np.dot(M1_base_tgt, W_inv)
            X_base = np.zeros(D_in, dtype=np.int64)
            X_base[sub_indices] = np.round(X_base_sub).astype(np.int64)

            X_coords = [X_base + k * J_i for k in k_vals]
            M_actuals = [np.dot(X, raw_w1) for X in X_coords]

            Y_N2 = np.mean(oracle_func(np.tile(X_coords[0], (B1_queries, 1))), axis=0)
            Y_N1 = np.mean(oracle_func(np.tile(X_coords[1], (B1_queries, 1))), axis=0)
            Y_P1 = np.mean(oracle_func(np.tile(X_coords[2], (B1_queries, 1))), axis=0)
            Y_P2 = np.mean(oracle_func(np.tile(X_coords[3], (B1_queries, 1))), axis=0)

            diff_P = np.max(np.abs(Y_P2 - Y_P1))
            diff_N = np.max(np.abs(Y_N2 - Y_N1))

            if diff_P > diff_N:
                sign = 1
                Y_OFF1, Y_OFF2 = Y_N1, Y_N2
                M_OFF1, M_OFF2 = M_actuals[1], M_actuals[0]
                Y_ON1, Y_ON2 = Y_P1, Y_P2
                M_ON1, M_ON2 = M_actuals[2], M_actuals[3]
            else:
                sign = -1
                Y_OFF1, Y_OFF2 = Y_P1, Y_P2
                M_OFF1, M_OFF2 = M_actuals[2], M_actuals[3]
                Y_ON1, Y_ON2 = Y_N1, Y_N2
                M_ON1, M_ON2 = M_actuals[1], M_actuals[0]

            slope_OFF = (Y_OFF2 - Y_OFF1) / (M_OFF2 - M_OFF1)
            slope_ON = (Y_ON2 - Y_ON1) / (M_ON2 - M_ON1)

            c_out = np.argmax(np.abs(slope_ON - slope_OFF))
            s_OFF = slope_OFF[c_out]
            s_ON = slope_ON[c_out]
            b_OFF = Y_OFF1[c_out] - s_OFF * M_OFF1
            b_ON = Y_ON1[c_out] - s_ON * M_ON1

            M_star = (b_OFF - b_ON) / (s_ON - s_OFF)
            B1_val = -int(np.round((M_star * sign) / q))

            candidate_signs.append(sign)
            candidate_B1s.append(B1_val)

        final_sign = int(stats.mode(candidate_signs, keepdims=False).mode)
        final_B1 = int(stats.mode(candidate_B1s, keepdims=False).mode)
        extracted_W1_final[:, i] = raw_w1 * final_sign
        extracted_B1_final[i] = final_B1

    print("Layer 1 extraction complete.")
    return extracted_W1_final, extracted_B1_final


def decrypt_W2(
    oracle_func: Callable,
    W1: np.ndarray,
    D_in: int,
    D_h1: int,
    D_h2: int,
    D_out: int,
    q: int,
    total_layers: int = 3,
    *,
    max_anchors: int = 1000,
    steps_per_anchor: int = 10,
    p1_queries: int = 1,
    p2_queries: int = 10,
    p3_queries: int = 1,
    return_sir: bool = False,
):
    """Recover second-layer signed weights W2 for the original 64-dim benchmark."""
    print("Initiating W2 extraction...")

    # Digits vs. MNIST note:
    # Retain sequential greedy row selection here to preserve the exact original
    # 64-20-20-10 benchmark behavior; mnist.py uses column-pivoted QR instead.
    sub_indices = find_rank_full_sub_indices(W1, D_h1, method="greedy")
    W_sub = W1[sub_indices, :]
    W_inv_sp = sp.Matrix(W_sub).inv()
    W1_inv_float = np.array(W_inv_sp, dtype=np.float64)

    batch_size = 1 + D_h1

    def get_macro_grad_batched(
        H1_anchor: np.ndarray,
        delta_x: float,
        n_queries: int,
        proj_vec: np.ndarray,
    ) -> np.ndarray | None:
        k_base = np.round(np.dot(H1_anchor, W1_inv_float)).astype(np.int64)
        k_batch = [k_base]
        for i in range(D_h1):
            H_step_target = np.zeros(D_h1)
            H_step_target[i] = delta_x
            dk_int = np.round(np.dot(H_step_target, W1_inv_float)).astype(np.int64)
            k_batch.append(k_base + dk_int)

        k_full_batch = np.zeros((batch_size, D_in), dtype=np.int64)
        for idx, k_sub in enumerate(k_batch):
            k_full_batch[idx, sub_indices] = k_sub

        X_batch = k_full_batch * q
        X_rep = np.repeat(X_batch, n_queries, axis=0)
        out_rep = oracle_func(X_rep)
        proj_out = np.dot(out_rep, proj_vec)
        means = np.mean(proj_out.reshape(batch_size, n_queries), axis=1)

        M1_actual = np.dot(k_full_batch, W1)
        H1_actual = np.maximum(0, M1_actual)

        mean_base = means[0]
        H1_base_actual = H1_actual[0]
        dH_1x = np.zeros((D_h1, D_h1), dtype=np.float64)
        dY_1x = np.zeros(D_h1, dtype=np.float64)
        for i in range(D_h1):
            dH_1x[i] = H1_actual[1 + i] - H1_base_actual
            dY_1x[i] = means[1 + i] - mean_base

        try:
            inv_1x = np.linalg.inv(dH_1x)
        except np.linalg.LinAlgError:
            return None

        scale_factor = q ** (total_layers - 1)
        G_1x_float = np.dot(inv_1x, dY_1x) * scale_factor
        return np.round(G_1x_float).astype(np.int64)

    extracted_W2_list: list[np.ndarray] = []
    candidate_pool: list[np.ndarray] = []

    def process_gradients(current_G: np.ndarray | None, prev_G: np.ndarray | None) -> bool:
        if current_G is None or prev_G is None:
            return False

        diff = current_G - prev_G
        if np.all(diff == 0):
            return False

        primitive = primitive_vector(diff)
        if primitive is not None and 0.02 * q < np.max(np.abs(primitive)) < 1.5 * q:
            for ex in extracted_W2_list:
                if np.array_equal(primitive, ex) or np.array_equal(primitive, -ex):
                    return False

            # Paper vs. Implementation note (Two-Hit Candidate Confirmation) &
            # Digits vs. MNIST note:
            # To filter occasional composite/spurious gradient differences in Layer 2,
            # candidates must be observed twice before acceptance. The original Digits
            # code accepts any second hit in ``candidate_pool`` (even within the same
            # random walk); mnist.py strengthens this by requiring hits from two
            # distinct, independent warps/sweeps (``cand_warp != warp_id``).
            for idx_cand, cand in enumerate(candidate_pool):
                if np.array_equal(primitive, cand) or np.array_equal(primitive, -cand):
                    extracted_W2_list.append(primitive)
                    candidate_pool.pop(idx_cand)
                    return True

            candidate_pool.append(primitive)
        return False

    print("Phase 1: Signature extraction with random walks.")
    p1_delta_x = 1e7
    PROJ_VEC = np.ones(D_out, dtype=np.int64)

    for _ in range(max_anchors):
        if len(extracted_W2_list) >= D_h2:
            break

        H1_walk = np.floor((1 + np.random.rand(D_h1)) / 2 * 1e10).astype(float)
        V_dir = np.floor(np.random.randn(D_h1) * 1e9).astype(float)
        PROJ_VEC = np.random.randint(1, 3, size=D_out) * np.random.choice([-1, 1], size=D_out)
        prev_G = get_macro_grad_batched(H1_walk, p1_delta_x, p1_queries, PROJ_VEC)

        for _ in range(steps_per_anchor):
            if len(extracted_W2_list) >= D_h2:
                break
            H1_walk = np.abs(H1_walk + V_dir)
            current_G = get_macro_grad_batched(H1_walk, p1_delta_x, p1_queries, PROJ_VEC)
            if current_G is not None:
                process_gradients(current_G, prev_G)
                prev_G = current_G

    if len(extracted_W2_list) < D_h2:
        print("Phase 2: Signature extraction with 2D sweeps.")
        # Paper vs. Implementation note (Phase 2 Boundary-Scraping 2D Sweeps):
        # Although a random walk theoretically intersects all W2 hyperplanes, the
        # preceding ReLU layer restricts the feasible domain to the nonnegative
        # orthant (H1 >= 0), pushing certain W2 hyperplanes near extreme orthant
        # boundaries that high-dimensional random walks rarely cross. Phase 2
        # systematically sweeps coordinate pairs (i, j) along the boundary faces
        # to recover these hard-to-reach neurons.
        p2_delta_x = 1_000_000.0
        for i in range(D_h1):
            for j in range(D_h1):
                if i == j:
                    continue
                if len(extracted_W2_list) >= D_h2:
                    break

                H1_walk = np.full(D_h1, 1e7, dtype=float)
                H1_walk[i] = 1e10
                V_dir = np.zeros(D_h1, dtype=float)
                V_dir[j] = 1e8

                prev_G = get_macro_grad_batched(H1_walk, p2_delta_x, p2_queries, PROJ_VEC)
                for _ in range(200):
                    if len(extracted_W2_list) >= D_h2:
                        break
                    H1_walk += V_dir
                    current_G = get_macro_grad_batched(H1_walk, p2_delta_x, p2_queries, PROJ_VEC)
                    if current_G is not None:
                        process_gradients(current_G, prev_G)
                        prev_G = current_G

    if len(extracted_W2_list) < D_h2:
        print(f"Error: Insufficient W2 neurons extracted ({len(extracted_W2_list)}/{D_h2}). Rank deficient.")
        if return_sir:
            return None, []
        return None

    print("Phase 3: Sign extraction.")
    extracted_W2_raw = np.array(extracted_W2_list).T
    W2_inv = np.linalg.pinv(extracted_W2_raw)
    extracted_W2_final = np.zeros_like(extracted_W2_raw)

    def get_X_for_H1_no_B1(H1_tgt: np.ndarray) -> np.ndarray:
        M1_tgt = H1_tgt * q
        X_sub = np.dot(M1_tgt, W1_inv_float)
        X_out = np.zeros(D_in, dtype=np.int64)
        X_out[sub_indices] = np.round(X_sub).astype(np.int64)
        return X_out

    sir_values: list[float] = []
    sir_details: list[dict] = []

    # Paper vs. Implementation note (LP-Based Sign Recovery & SIR Diagnostic):
    # We solve a linear program (HiGHS) over H1 >= min_H1_req subject to
    # raw_w2 . H1 = target_bias so that +/- steps stay strictly inside the positive
    # L1 ReLU orthant while crossing only neuron i's hyperplane. Comparing +/- output
    # response amplitudes determines the sign; their ratio (SIR) empirically separates
    # exact W2 recoveries (~3,000) from corrupted recoveries (<10).
    for i in range(D_h2):
        raw_w2 = extracted_W2_raw[:, i]
        M2_step = np.zeros(D_h2)
        M2_step[i] = 5_000_000.0
        H1_step = np.dot(M2_step, W2_inv)

        min_H1_req = np.abs(2 * H1_step) + 15000
        bounds = [(min_H1_req[k], None) for k in range(D_h1)]

        success = False
        res = None
        for target_bias in (0, 500_000, -500_000, 1_500_000, -1_500_000, 2_500_000):
            res = linprog(c=np.ones(D_h1), A_eq=[raw_w2], b_eq=[target_bias], bounds=bounds, method="highs")
            if res.success:
                success = True
                break

        if not success or res is None:
            print(f"Warning: Sign extraction failed for L2 Neuron {i:02d}. Retaining original sign.")
            extracted_W2_final[:, i] = raw_w2
            sir_details.append({
                "neuron_idx": i,
                "linprog_success": False,
                "diff_pos": 0.0,
                "diff_neg": 0.0,
                "chosen_sign": 1,
                "sir": 1.0,
            })
            continue

        H1_base = np.round(res.x).astype(np.float64)
        X_pos_1 = get_X_for_H1_no_B1(H1_base + 1 * H1_step)
        X_pos_2 = get_X_for_H1_no_B1(H1_base + 2 * H1_step)
        X_neg_1 = get_X_for_H1_no_B1(H1_base - 1 * H1_step)
        X_neg_2 = get_X_for_H1_no_B1(H1_base - 2 * H1_step)

        Y1_p = np.mean(oracle_func(np.tile(X_pos_1, (p3_queries, 1))), axis=0)
        Y2_p = np.mean(oracle_func(np.tile(X_pos_2, (p3_queries, 1))), axis=0)
        diff_pos = float(np.sum(np.abs(Y2_p - Y1_p)))

        Y1_n = np.mean(oracle_func(np.tile(X_neg_1, (p3_queries, 1))), axis=0)
        Y2_n = np.mean(oracle_func(np.tile(X_neg_2, (p3_queries, 1))), axis=0)
        diff_neg = float(np.sum(np.abs(Y2_n - Y1_n)))

        if diff_pos > diff_neg:
            extracted_W2_final[:, i] = raw_w2
            sir_val = float(abs(diff_pos / (diff_neg + 1e-4)))
            chosen_sign = 1
        else:
            extracted_W2_final[:, i] = -raw_w2
            sir_val = float(abs(diff_neg / (diff_pos + 1e-4)))
            chosen_sign = -1

        sir_values.append(sir_val)
        sir_details.append({
            "neuron_idx": i,
            "linprog_success": True,
            "diff_pos": diff_pos,
            "diff_neg": diff_neg,
            "chosen_sign": chosen_sign,
            "sir": sir_val,
        })

    print("W2 extraction complete.")
    if sir_values:
        print(f"(Avg SIR: {sum(sir_values) / len(sir_values):.2f}).")
    if return_sir:
        return extracted_W2_final, sir_details
    return extracted_W2_final


def decrypt_B2(
    oracle_func: Callable,
    W1: np.ndarray,
    B1: np.ndarray,
    W2: np.ndarray,
    D_in: int,
    D_h1: int,
    D_h2: int,
    q: int,
    *,
    num_bases: int = 10,
    test_queries: int = 500,
) -> np.ndarray:
    """Recover second-layer biases B2 via multi-base piecewise-linear intersection."""
    print("Initiating B2 extraction...")

    sub_indices = find_rank_full_sub_indices(W1, D_h1, method="greedy")
    W1_inv_float = np.linalg.inv(W1[sub_indices, :])

    extracted_B2 = np.zeros(D_h2, dtype=np.int64)
    eff_W2 = np.dot(W1, W2)

    def get_intersect(x1, y1, x2, y2, x3, y3, x4, y4, c_idx) -> float:
        m_off = (y2[c_idx] - y1[c_idx]) / (x2 - x1)
        m_on = (y4[c_idx] - y3[c_idx]) / (x4 - x3)
        b_off = y1[c_idx] - m_off * x1
        b_on = y3[c_idx] - m_on * x3
        if m_on == m_off:
            return 0.0
        return float((b_off - b_on) / (m_on - m_off))

    T_OFF2, T_OFF1, T_ON1, T_ON2 = -2e6, -1e6, 1e6, 2e6

    for j in range(D_h2):
        c = int(np.argmax(np.abs(eff_W2[:, j])))
        w_eff = eff_W2[c, j] / q
        candidate_B2s = []

        for _ in range(num_bases):
            H1_base = np.random.uniform(1e8, 5e8, D_h1)
            M2_j = np.dot(H1_base, W2[:, j])
            if M2_j > 0:
                neg_idx = np.where(W2[:, j] < 0)[0]
                if len(neg_idx) > 0:
                    H1_base[neg_idx] += (-M2_j) / np.sum(W2[neg_idx, j])
            else:
                pos_idx = np.where(W2[:, j] > 0)[0]
                if len(pos_idx) > 0:
                    H1_base[pos_idx] += (-M2_j) / np.sum(W2[pos_idx, j])

            M1_tgt = (H1_base - B1) * q
            X_sub = np.dot(M1_tgt, W1_inv_float)
            X_base = np.zeros(D_in, dtype=np.float64)
            X_base[sub_indices] = X_sub
            X_base = np.round(X_base).astype(np.int64)

            M1_actual = np.dot(X_base, W1)
            H1_base_act = (M1_actual / q) + B1
            M2_base_act = np.dot(H1_base_act, W2)[j]

            k_OFF2 = int((T_OFF2 - M2_base_act) / w_eff)
            k_OFF1 = int((T_OFF1 - M2_base_act) / w_eff)
            k_ON1 = int((T_ON1 - M2_base_act) / w_eff)
            k_ON2 = int((T_ON2 - M2_base_act) / w_eff)

            def get_Y_B2(k_val: int) -> np.ndarray:
                X_test = X_base.copy()
                X_test[c] += k_val
                return np.mean(oracle_func(np.tile(X_test, (test_queries, 1))), axis=0)

            Y_OFF2 = get_Y_B2(k_OFF2)
            Y_OFF1 = get_Y_B2(k_OFF1)
            Y_ON1 = get_Y_B2(k_ON1)
            Y_ON2 = get_Y_B2(k_ON2)

            slope_OFF = (Y_OFF1 - Y_OFF2) / (k_OFF1 - k_OFF2)
            slope_ON = (Y_ON2 - Y_ON1) / (k_ON2 - k_ON1)
            out_c = int(np.argmax(np.abs(slope_ON - slope_OFF)))

            k_star = get_intersect(k_OFF2, Y_OFF2, k_OFF1, Y_OFF1, k_ON1, Y_ON1, k_ON2, Y_ON2, out_c)
            M2_star = M2_base_act + k_star * w_eff
            candidate_B2s.append(-int(round(M2_star / q)))

        extracted_B2[j] = int(stats.mode(candidate_B2s, keepdims=False).mode)

    print("B2 extraction complete.")
    return extracted_B2


def decrypt_L_last(
    oracle_func: Callable,
    W_list: Sequence[np.ndarray],
    B_list: Sequence[np.ndarray],
    D_out: int,
    q: int,
    *,
    last_layer_repetitions: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover final-layer weights and biases using single-shot Demand-Driven LLL."""
    return solve_last_layer_lll(
        oracle_func,
        W_list,
        B_list,
        D_out,
        q,
        sub_indices_method="greedy",
        last_layer_repetitions=last_layer_repetitions,
        scale_target_with_q=False,
    )
