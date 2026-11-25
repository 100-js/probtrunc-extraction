"""Unified MNIST model extraction engine for Narrow (784-20-20-10) and Wide (784-128-64-10).

This module consolidates the rebuttal MNIST extraction pipelines into a single
parameterized implementation across precisions S in {6, 9, 12} and hidden widths
(20-20 and 128-64). All algorithmic refinements over the original 64-dim Digits
extractor (``original_digits.py``) and over the theoretical paper formulation are
explicitly parameterized via ``configs/*.toml`` and documented inline below.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import scipy.stats as stats
import sympy as sp
from scipy.optimize import linprog

from .common import (
    find_rank_full_sub_indices,
    primitive_vector,
    solve_last_layer_flooded_pinv,
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
    rank1_consensus: bool = False,
    b1_high_snr_filter: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Recover first-layer integer weights W1 and biases B1 for 784-dim MNIST networks."""
    print("Initiating W1 extraction...")

    # Digits vs. MNIST note (q-Adaptive Probing Scale & Repetition Schedule):
    # Unlike the fixed S=9 (q=512) Digits implementation, MNIST benchmarks sweep
    # S in {6, 9, 12} (q in {64, 512, 4096}). Because the gradient scaling factor
    # q^(total_layers) grows rapidly while signed int64 bounds cap the walk radius,
    # high-precision targets (q > 1024) increase delta_x, shrink walk_scale, and
    # average W1_queries=32 responses per probe point.
    q_ratio = q / 512.0
    delta_x = 1e10 if q <= 512 else (1.0e11 if q > 1024 else 5e10)
    walk_scale = 1e13 / max(1.0, q_ratio) if q <= 1024 else 1.0e12
    dir_scale = 1e11 / max(1.0, q_ratio) if q <= 1024 else 1.0e11
    N_queries = W1_queries

    extracted_W1_raw: list[np.ndarray] = []

    # Digits vs. MNIST note (Batched Multi-Output Gradient Probing):
    # Instead of projecting onto a single scalar logit sum and querying one input
    # coordinate at a time (as in original_digits.py), we submit all 1 + D_in
    # perturbation queries in one BLAS-accelerated batch and retain all D_out
    # output coordinates separately (shape D_in x D_out).
    def get_macro_grads_all(X_anchor: np.ndarray) -> np.ndarray:
        batch_size = 1 + D_in
        X_batch = np.empty((batch_size, D_in), dtype=np.float64)
        X_batch[0] = X_anchor
        X_batch[1:] = X_anchor + np.eye(D_in) * delta_x

        if N_queries == 1:
            out = oracle_func(X_batch)
            return np.round(((out[1:] - out[0:1]) / delta_x) * (q**total_layers)).astype(np.int64)

        X_rep = np.repeat(X_batch, N_queries, axis=0)
        out = oracle_func(X_rep)
        means = np.mean(out.reshape(batch_size, N_queries, D_out), axis=1)
        return np.round(((means[1:] - means[0:1]) / delta_x) * (q**total_layers)).astype(np.int64)

    def _accept_if_valid(prim: np.ndarray | None) -> None:
        # Digits vs. MNIST note (Non-Zero Support Filter):
        # In addition to the magnitude plausibility window (0.02*q < max(|v|) < 1.5*q),
        # MNIST requires at least 10 non-zero entries to reject degenerate sparse
        # noise artifacts from near-zero border pixels.
        if (
            prim is not None
            and 0.02 * q < np.max(np.abs(prim)) < 1.5 * q
            and np.count_nonzero(prim) >= 10
        ):
            for ex in extracted_W1_raw:
                if np.array_equal(prim, ex) or np.array_equal(prim, -ex):
                    return
            extracted_W1_raw.append(prim)

    for _ in range(W1_anchors):
        if len(extracted_W1_raw) >= D_h:
            break

        X_walk = np.floor(np.random.randn(D_in) * walk_scale).astype(float)
        V_dir = np.floor(np.random.randn(D_in) * dir_scale).astype(float)
        # Digits vs. MNIST note (Overflow Guard):
        # At high precision (S=12) or wide input dimensions, random walk anchors
        # occasionally trigger the oracle's conservative int64 bound; we safely skip
        # such anchors rather than aborting extraction.
        try:
            prev_G = get_macro_grads_all(X_walk)
        except OverflowError:
            continue

        for _ in range(W1_steps_per_anchor):
            if len(extracted_W1_raw) >= D_h:
                break

            X_walk += V_dir
            try:
                current_G = get_macro_grads_all(X_walk)
            except OverflowError:
                break
            diff = current_G - prev_G

            if rank1_consensus:
                # Wide MNIST note (Strict Cross-Coordinate Rank-1 Consensus):
                # In dense 128-neuron hidden layers, a single random-walk step
                # frequently crosses multiple L1 ReLU boundaries simultaneously.
                # A true single L1 neuron flip produces a Rank-1 outer product
                # delta_G = w_1 (x) beta^T, so every active output column shares the
                # exact same primitive direction. Requiring unanimous collinearity
                # across all (>= 2) active output coordinates strictly eliminates
                # composite multi-neuron vectors.
                if np.count_nonzero(diff) > 0:
                    active_cols = [c for c in range(D_out) if np.count_nonzero(diff[:, c]) > 0]
                    if len(active_cols) >= 2:
                        prim0 = primitive_vector(diff[:, active_cols[0]])
                        if prim0 is not None:
                            is_rank1 = True
                            for c in active_cols[1:]:
                                p_c = primitive_vector(diff[:, c])
                                if p_c is None or not (
                                    np.array_equal(p_c, prim0) or np.array_equal(p_c, -prim0)
                                ):
                                    is_rank1 = False
                                    break
                            if is_rank1:
                                _accept_if_valid(prim0)
            else:
                for c in range(D_out):
                    d_c = diff[:, c]
                    if np.count_nonzero(d_c) > 0:
                        _accept_if_valid(primitive_vector(d_c))

            prev_G = current_G

    if len(extracted_W1_raw) < D_h:
        print(f"Error: Insufficient W1 neurons extracted ({len(extracted_W1_raw)}/{D_h}). Rank deficient.")
        return None, None

    extracted_W1_raw_mat = np.array(extracted_W1_raw).T

    # Paper vs. Implementation & Digits vs. MNIST note:
    # Replaces both the paper's full pseudoinverse and Digits' 1,000-trial random
    # max-det search with deterministic column-pivoted QR (method="qr") to pick a
    # well-conditioned D_h x D_h square submatrix despite zero-valued MNIST borders.
    try:
        sub_indices = find_rank_full_sub_indices(extracted_W1_raw_mat, D_h, method="qr")
    except Exception as error:
        print(f"Error finding rank-full submatrix: {error}")
        return None, None

    W_sub = extracted_W1_raw_mat[sub_indices, :]
    W_inv = np.linalg.inv(W_sub)

    print("Initiating W1 sign & B1 extraction...")
    extracted_W1_final = np.zeros_like(extracted_W1_raw_mat)
    extracted_B1_final = np.zeros(D_h, dtype=np.int64)

    step_unit = 1e6 * q_ratio
    k_vals = [-2, -1, 1, 2]

    for i in range(D_h):
        raw_w1 = extracted_W1_raw_mat[:, i]
        if b1_high_snr_filter:
            scored_candidates: list[tuple[float, int, int]] = []
        else:
            candidate_B1s: list[int] = []
            candidate_signs: list[int] = []

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

            if b1_high_snr_filter:
                sign = 1 if diff_P > diff_N else -1
                slope_OFF = (Y_N2 - Y_N1) / (M_actuals[0] - M_actuals[1])
                slope_ON = (Y_P2 - Y_P1) / (M_actuals[3] - M_actuals[2])
                c_out = int(np.argmax(np.abs(slope_ON - slope_OFF)))
                s_diff = float(np.abs(slope_ON[c_out] - slope_OFF[c_out]))
                if s_diff < 1e-12:
                    continue
                s_OFF = slope_OFF[c_out]
                s_ON = slope_ON[c_out]
                b_OFF = Y_N1[c_out] - s_OFF * M_actuals[1]
                b_ON = Y_P1[c_out] - s_ON * M_actuals[2]
                M_star = (b_OFF - b_ON) / (s_ON - s_OFF)
                B1_val = -int(np.round((M_star * sign) / q))
                scored_candidates.append((s_diff, sign, B1_val))
            else:
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
                c_out = int(np.argmax(np.abs(slope_ON - slope_OFF)))
                s_OFF = slope_OFF[c_out]
                s_ON = slope_ON[c_out]
                b_OFF = Y_OFF1[c_out] - s_OFF * M_OFF1
                b_ON = Y_ON1[c_out] - s_ON * M_ON1
                M_star = (b_OFF - b_ON) / (s_ON - s_OFF)
                B1_val = -int(np.round((M_star * sign) / q))
                candidate_signs.append(sign)
                candidate_B1s.append(B1_val)

        if b1_high_snr_filter:
            # Wide MNIST note (High-SNR Anchor Selection for B1):
            # In wider networks, some random anchors experience weak downstream
            # gain through ReLU layers. Ranking anchors by slope discontinuity
            # |slope_ON - slope_OFF| and voting over the top half filters out
            # low-SNR anchors before computing the sign and bias mode.
            if scored_candidates:
                scored_candidates.sort(key=lambda item: item[0], reverse=True)
                top = scored_candidates[: max(3, len(scored_candidates) // 2)]
                final_sign = int(stats.mode([c[1] for c in top], keepdims=False).mode)
                final_B1 = int(stats.mode([c[2] for c in top], keepdims=False).mode)
            else:
                final_sign, final_B1 = 1, 0
        else:
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
    max_anchors: int = 1500,
    steps_per_anchor: int = 15,
    p1_queries: int = 1,
    p2_queries: int = 10,
    p3_queries: int = 1,
    walk_mode: str = "dense",
    w2_inv_backend: str = "sympy",
    return_sir: bool = False,
):
    """Recover second-layer signed weights W2 for Narrow or Wide MNIST networks."""
    if walk_mode not in ("dense", "sparsity_masked"):
        raise ValueError(f"Unsupported walk_mode: {walk_mode!r}")
    if w2_inv_backend not in ("sympy", "float64"):
        raise ValueError(f"Unsupported w2_inv_backend: {w2_inv_backend!r}")

    print("Initiating W2 extraction...")

    # Digits vs. MNIST note:
    # Pivoted QR selects a well-conditioned square submatrix of W1. For Narrow MNIST
    # (D_h1=20) we retain exact rational inversion via SymPy; for Wide MNIST (D_h1=128)
    # exact symbolic inversion is intractable, so float64 inversion is used instead.
    sub_indices = find_rank_full_sub_indices(W1, D_h1, method="qr")
    W_sub = W1[sub_indices, :]
    if w2_inv_backend == "sympy":
        W1_inv_float = np.array(sp.Matrix(W_sub).inv(), dtype=np.float64)
    else:
        W1_inv_float = np.linalg.inv(W_sub.astype(np.float64))

    batch_size = 1 + D_h1
    p1_delta_x = 1e7 if q <= 512 else (1.0e8 if q > 1024 else 5e7)
    dk_p1 = np.round(p1_delta_x * W1_inv_float).astype(np.int64)

    def get_macro_grad_batched(
        H1_anchor: np.ndarray,
        delta_x: float,
        n_queries: int,
        proj_vec: np.ndarray,
        dk_cached: np.ndarray | None = None,
    ) -> np.ndarray | None:
        k_base = np.round(np.dot(H1_anchor, W1_inv_float)).astype(np.int64)
        k_full_batch = np.zeros((batch_size, D_in), dtype=np.int64)
        k_full_batch[0, sub_indices] = k_base
        if dk_cached is not None:
            k_full_batch[1:, sub_indices] = k_base + dk_cached
        else:
            dk_all = np.round(delta_x * W1_inv_float).astype(np.int64)
            k_full_batch[1:, sub_indices] = k_base + dk_all

        X_batch = k_full_batch * q
        if n_queries == 1:
            out_rep = oracle_func(X_batch)
            means = np.dot(out_rep, proj_vec)
        else:
            X_rep = np.repeat(X_batch, n_queries, axis=0)
            out_rep = oracle_func(X_rep)
            proj_out = np.dot(out_rep, proj_vec)
            means = np.mean(proj_out.reshape(batch_size, n_queries), axis=1)

        # k_full_batch is identically zero outside sub_indices; multiplying only the
        # active D_h1 columns against W_sub is bit-exact and ~6x faster on 784-dim MNIST.
        M1_actual = np.dot(k_full_batch[:, sub_indices], W_sub)
        H1_actual = np.maximum(0, M1_actual)

        dH_1x = (H1_actual[1:] - H1_actual[0]).astype(np.float64)
        dY_1x = (means[1:] - means[0]).astype(np.float64)
        try:
            inv_1x = np.linalg.inv(dH_1x)
        except np.linalg.LinAlgError:
            return None

        scale_factor = q ** (total_layers - 1)
        G_1x_float = np.dot(inv_1x, dY_1x) * scale_factor
        return np.round(G_1x_float).astype(np.int64)

    extracted_W2_list: list[np.ndarray] = []
    # Digits vs. MNIST note (Cross-Warp Independent Candidate Verification):
    # Whereas original_digits.py promotes a W2 signature on any second hit in
    # ``candidate_pool`` (even along the same walk), MNIST tags each candidate with
    # its warp/sweep identifier (``warp_id``) and promotes a signature only after
    # agreement across TWO INDEPENDENT warps/sweeps (``cand_warp != warp_id``).
    # We index the pool by canonical prefix for fast O(1) lookup at wide D_h1=128.
    candidate_pool: dict[tuple[int, ...], list[tuple[np.ndarray, str]]] = {}

    def process_gradients(
        current_G: np.ndarray | None,
        prev_G: np.ndarray | None,
        warp_id: str,
    ) -> bool:
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

            key = tuple(int(v) for v in primitive[:5])
            bucket = candidate_pool.setdefault(key, [])
            for idx_cand, (cand, cand_warp) in enumerate(bucket):
                if cand_warp != warp_id and (
                    np.array_equal(primitive, cand) or np.array_equal(primitive, -cand)
                ):
                    extracted_W2_list.append(primitive)
                    bucket.pop(idx_cand)
                    return True

            if not any(
                cw == warp_id and (np.array_equal(primitive, c) or np.array_equal(primitive, -c))
                for c, cw in bucket
            ):
                bucket.append((primitive, warp_id))
        return False

    print("Phase 1: Signature extraction with random walks.")
    for a_idx in range(max_anchors):
        if len(extracted_W2_list) >= D_h2:
            break

        if walk_mode == "sparsity_masked":
            # Wide MNIST note (Sparsity-Masked Random Walk for Wide H1 >= 0 Orthant):
            # In 128-wide hidden layers, all-active dense H1 vectors push pre-activations
            # z2 = H1 @ W2 strongly toward one sign due to column-sum skew, while
            # Phase 2 2D sweeps over all 128 x 127 coordinate pairs are prohibitively
            # expensive. Masking 15%-85% of H1 coordinates active per warp naturally
            # samples both interior and sparse boundary faces of H1 >= 0 so all 64
            # neurons cross zero in Phase 1 without requiring Phase 2.
            prob_active = np.random.uniform(0.15, 0.85)
            mask = (np.random.rand(D_h1) < prob_active).astype(float)
            scale = np.random.uniform(0.2, 1.5) * 1e10
            H1_walk = np.abs(np.random.randn(D_h1)) * scale * mask + 1e8
            V_dir = np.random.randn(D_h1) * (scale * 0.08)
        else:
            H1_walk = np.floor((1 + np.random.rand(D_h1)) / 2 * 1e10).astype(float)
            V_dir = np.floor(np.random.randn(D_h1) * 1e9).astype(float)

        PROJ_VEC = np.random.randint(1, 3, size=D_out) * np.random.choice([-1, 1], size=D_out)
        prev_G = get_macro_grad_batched(H1_walk, p1_delta_x, p1_queries, PROJ_VEC, dk_cached=dk_p1)

        for _ in range(steps_per_anchor):
            if len(extracted_W2_list) >= D_h2:
                break
            if walk_mode == "sparsity_masked":
                H1_walk = np.maximum(0.0, H1_walk + V_dir)
            else:
                H1_walk = np.abs(H1_walk + V_dir)
            current_G = get_macro_grad_batched(
                H1_walk, p1_delta_x, p1_queries, PROJ_VEC, dk_cached=dk_p1
            )
            if current_G is not None:
                process_gradients(current_G, prev_G, f"p1_{a_idx}")
                prev_G = current_G

    if walk_mode == "dense" and p2_queries > 0 and len(extracted_W2_list) < D_h2:
        print("Phase 2: Signature extraction with 2D sweeps.")
        # Paper vs. Implementation & Digits vs. MNIST note (Phase 2 Boundary Sweeps):
        # Scrapes extreme boundary faces of the ReLU-restricted H1 >= 0 orthant when
        # dense random walks miss narrow W2 hyperplanes. Compared to original_digits.py
        # (200 steps/pair reusing PROJ_VEC), Narrow MNIST uses 40 steps/pair with a
        # fresh random projection per pair, q-scaled delta_x, and unified p2_queries
        # (10 at S=6/9, 32 at S=12).
        p2_delta_x = 1_000_000.0 if q <= 512 else (1.0e8 if q > 1024 else 5e7)
        dk_p2 = np.round(p2_delta_x * W1_inv_float).astype(np.int64)
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

                PROJ_VEC = np.random.randint(1, 3, size=D_out) * np.random.choice([-1, 1], size=D_out)
                prev_G = get_macro_grad_batched(
                    H1_walk, p2_delta_x, p2_queries, PROJ_VEC, dk_cached=dk_p2
                )
                for _ in range(40):
                    if len(extracted_W2_list) >= D_h2:
                        break
                    H1_walk += V_dir
                    current_G = get_macro_grad_batched(
                        H1_walk, p2_delta_x, p2_queries, PROJ_VEC, dk_cached=dk_p2
                    )
                    if current_G is not None:
                        process_gradients(current_G, prev_G, f"p2_{i}_{j}")
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

    q_scale = max(0.1, q / 512.0)
    sir_details: list[dict] = []

    for i in range(D_h2):
        raw_w2 = extracted_W2_raw[:, i]
        M2_step = np.zeros(D_h2)
        M2_step[i] = 5_000_000.0 * q_scale
        H1_step = np.dot(M2_step, W2_inv)

        min_H1_req = np.abs(2 * H1_step) + 15000 * q_scale
        bounds = [(min_H1_req[k], None) for k in range(D_h1)]

        success = False
        res = None
        target_candidates = [val * q_scale for val in (0, 500_000, -500_000, 1_500_000, -1_500_000, 2_500_000)]
        for target_bias in target_candidates:
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

        sir_details.append({
            "neuron_idx": i,
            "linprog_success": True,
            "diff_pos": diff_pos,
            "diff_neg": diff_neg,
            "chosen_sign": chosen_sign,
            "sir": sir_val,
        })

    print("W2 extraction complete.")
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
    num_bases: int = 20,
    test_queries: int = 1000,
) -> np.ndarray:
    """Recover second-layer biases B2 via multi-base piecewise-linear intersection."""
    print("Initiating B2 extraction...")

    sub_indices = find_rank_full_sub_indices(W1, D_h1, method="qr")
    W1_inv_float = np.linalg.inv(W1[sub_indices, :].astype(np.float64))

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

    # Digits vs. MNIST note (q-Scaled B2 Probe Offsets):
    # Scales the ON/OFF line-probe offsets above q=512 so pre-activation displacements
    # remain well outside the quantized truncation transition band at S=12.
    q_ratio = q / 512.0
    T_mult = max(1.0, q_ratio * 1.5)
    T_OFF2 = -2e6 * T_mult
    T_OFF1 = -1e6 * T_mult
    T_ON1 = 1e6 * T_mult
    T_ON2 = 2e6 * T_mult

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
    solver: str = "lll",
    last_layer_repetitions: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch final-layer extraction to either repeated LLL or Flooded Linear Inversion."""
    if solver == "lll":
        return solve_last_layer_lll(
            oracle_func,
            W_list,
            B_list,
            D_out,
            q,
            sub_indices_method="qr",
            last_layer_repetitions=last_layer_repetitions,
            scale_target_with_q=True,
        )
    if solver == "flooded_pinv":
        return solve_last_layer_flooded_pinv(
            oracle_func,
            W_list,
            B_list,
            D_out,
            q,
            sub_indices_method="qr",
            last_layer_repetitions=last_layer_repetitions,
        )
    raise ValueError(f"Unsupported last_layer_solver: {solver!r} (expected 'lll' or 'flooded_pinv')")
