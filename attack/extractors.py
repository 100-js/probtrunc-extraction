import math
import numpy as np
import scipy.stats as stats
import sympy as sp
from scipy.optimize import linprog

from .common import find_rank_full_sub_indices, primitive_vector

try:
    from fpylll import IntegerMatrix, LLL
except ImportError:
    IntegerMatrix = None
    LLL = None

def decrypt_L1(oracle_func, D_in, D_h, D_out, q, total_layers, W1_anchors=500, W1_steps_per_anchor=10, W1_queries=1, B1_anchors=10, B1_queries=500):
    print("Initiating W1 extraction...")

    delta_x = 1e10
    N_queries = W1_queries
    extracted_W1_raw = []

    def get_macro_grad_X(X_anchor, proj_vec):
        X_rep = np.tile(X_anchor, (N_queries, 1))
        mean_base = np.mean(np.dot(oracle_func(X_rep), proj_vec))

        grad = np.zeros(D_in, dtype=np.int64)
        for i in range(D_in):
            X_step = X_anchor.copy()
            X_step[i] += delta_x
            mean_step = np.mean(np.dot(oracle_func(np.tile(X_step, (N_queries, 1))), proj_vec))
            grad[i] = int(np.round(((mean_step - mean_base) / delta_x) * (q ** total_layers)))
        return grad

    for a_idx in range(W1_anchors):
        if len(extracted_W1_raw) >= D_h:
            break

        X_walk = np.floor(np.random.randn(D_in) * 1e13).astype(float)
        V_dir = np.floor(np.random.randn(D_in) * 1e11).astype(float)
        PROJ_VEC = np.random.choice([1], size=D_out)
        prev_G = get_macro_grad_X(X_walk, PROJ_VEC)

        for s_idx in range(W1_steps_per_anchor):
            if len(extracted_W1_raw) >= D_h:
                break

            X_walk += V_dir
            current_G = get_macro_grad_X(X_walk, PROJ_VEC)
            diff = current_G - prev_G

            primitive = primitive_vector(diff)
            if primitive is not None and 0.02 * q < np.max(np.abs(primitive)) < 1.5 * q:
                is_new = True
                for ex in extracted_W1_raw:
                    if np.array_equal(primitive, ex) or np.array_equal(primitive, -ex):
                        is_new = False
                        break

                if is_new:
                    extracted_W1_raw.append(primitive)
                    # print(f"Target acquired: {len(extracted_W1_raw):02d}/{D_h} (Warp {a_idx}, Step {s_idx})")

            prev_G = current_G

    if len(extracted_W1_raw) < D_h:
        print(f"Error: Insufficient W1 neurons extracted ({len(extracted_W1_raw)}/{D_h}). Rank deficient.")
        return None, None

    extracted_W1_raw = np.array(extracted_W1_raw).T

    # Instead of using a pseudoinverse (which relies on SVD), we search for a square submatrix that maximizes the determinant.
    # Computing the pseudoinverse on the entire overdetermined matrix risks blowing up microscopic floating-point rounding errors due to the reciprocals of extremely small singular values (caused by nearly dependent rows).
    # By finding the submatrix with the maximum volume (determinant), we select the most orthogonal basis.
    # This minimizes the condition number and ensures a numerically stable, exact matrix inversion without floating-point error amplification.
    best_logdet = -np.inf
    best_sub_indices = []
    search_attempts = 1000

    while True:
        for _ in range(search_attempts):
            perm = np.random.permutation(D_in)
            temp_indices = []

            for i in perm:
                test_indices = temp_indices + [i]
                if np.linalg.matrix_rank(extracted_W1_raw[test_indices, :]) == len(test_indices):
                    temp_indices.append(i)
                if len(temp_indices) == extracted_W1_raw.shape[1]:
                    break

            if len(temp_indices) == extracted_W1_raw.shape[1]:
                W_temp = extracted_W1_raw[temp_indices, :]
                sign, logdet = np.linalg.slogdet(W_temp)

                if logdet > best_logdet:
                    best_logdet = logdet
                    best_sub_indices = sorted(temp_indices)

        if best_sub_indices:
            break

    sub_indices = best_sub_indices
    W_sub = extracted_W1_raw[sub_indices, :]
    W_inv = np.linalg.inv(W_sub)

    print("Initiating W1 sign & B1 extraction...")

    extracted_W1_final = np.zeros_like(extracted_W1_raw)
    extracted_B1_final = np.zeros(extracted_W1_raw.shape[1], dtype=np.int64)

    step_unit = 1e6
    k_vals = [-2, -1, 1, 2]

    for i in range(extracted_W1_raw.shape[1]):
        raw_w1 = extracted_W1_raw[:, i]

        candidate_B1s = []
        candidate_signs = []
        status = ""

        for _ in range(B1_anchors):
            M1_step_tgt = np.zeros(extracted_W1_raw.shape[1])
            M1_step_tgt[i] = step_unit
            J_i_sub = np.dot(M1_step_tgt, W_inv)
            J_i = np.zeros(D_in, dtype=np.int64)
            J_i[sub_indices] = np.round(J_i_sub).astype(np.int64)

            M1_base_tgt = np.random.choice([-1e10, 1e10], size=extracted_W1_raw.shape[1])
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
                status = "Sign: (+)"
                Y_OFF1, Y_OFF2 = Y_N1, Y_N2
                M_OFF1, M_OFF2 = M_actuals[1], M_actuals[0]
                Y_ON1, Y_ON2 = Y_P1, Y_P2
                M_ON1, M_ON2 = M_actuals[2], M_actuals[3]
            else:
                sign = -1
                status = "Sign: (-)"
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

        final_sign = stats.mode(candidate_signs, keepdims=False).mode
        final_B1 = stats.mode(candidate_B1s, keepdims=False).mode

        extracted_W1_final[:, i] = raw_w1 * final_sign
        extracted_B1_final[i] = final_B1

        # print(f"Neuron {i:02d} | {status} | Extracted B1: {final_B1}")

    print("Layer 1 extraction complete.")
    return extracted_W1_final, extracted_B1_final


def decrypt_W2(oracle_func, W1, D_in, D_h1, D_h2, D_out, q, total_layers=3, max_anchors=500, steps_per_anchor=10, p1_queries=1, p2_queries=10, p3_queries=1):
    print("Initiating W2 extraction...")

    sub_indices = find_rank_full_sub_indices(W1, D_h1)

    W_sub = W1[sub_indices, :]
    W_inv_sp = sp.Matrix(W_sub).inv()
    W1_inv_float = np.array(W_inv_sp, dtype=np.float64)

    batch_size = 1 + D_h1

    def get_macro_grad_batched(H1_anchor, delta_x, n_queries, proj_vec):
        k_base_float = np.dot(H1_anchor, W1_inv_float)
        k_base = np.round(k_base_float).astype(np.int64)

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

    extracted_W2_list = []
    candidate_pool = []

    def process_gradients(current_G, prev_G, phase_name, a_idx, s_idx):
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

            for idx_cand, cand in enumerate(candidate_pool):
                if np.array_equal(primitive, cand) or np.array_equal(primitive, -cand):
                    extracted_W2_list.append(primitive)
                    warp_str = f"{a_idx+1:03d}" if isinstance(a_idx, int) else str(a_idx)
                    step_str = f"{s_idx+1:02d}" if isinstance(s_idx, int) else str(s_idx)
                    # print(f"[{phase_name}] Target acquired: {len(extracted_W2_list):02d}/{D_h2} (Warp {warp_str}, Step {step_str})")
                    candidate_pool.pop(idx_cand)
                    return True

            candidate_pool.append(primitive)
        return False

    print("Phase 1: Signature extraction with random walks.")
    p1_delta_x = 1e7

    for a_idx in range(max_anchors):
        if len(extracted_W2_list) >= D_h2:
            break

        H1_walk = np.floor((1 + np.random.rand(D_h1)) / 2 * 1e10).astype(float)
        V_dir = np.floor(np.random.randn(D_h1) * 1e9).astype(float)

        PROJ_VEC = np.random.randint(1, 3, size=D_out) * np.random.choice([-1, 1], size=D_out)
        prev_G = get_macro_grad_batched(H1_walk, p1_delta_x, p1_queries, PROJ_VEC)

        for s_idx in range(steps_per_anchor):
            if len(extracted_W2_list) >= D_h2:
                break
            H1_walk = np.abs(H1_walk + V_dir)
            current_G = get_macro_grad_batched(H1_walk, p1_delta_x, p1_queries, PROJ_VEC)
            if current_G is not None:
                process_gradients(current_G, prev_G, "Phase 1", a_idx, s_idx)
                prev_G = current_G

    if len(extracted_W2_list) < D_h2:
        print("Phase 2: Signature extraction with 2D sweeps.")
        # While a random walk theoretically finds all W2 hyperplanes, the previous layer's ReLU restricts the domain (H1 >= 0), leaving some hyperplanes hidden near the extreme boundaries.
        # High-dimensional random walks are highly inefficient at hitting these narrow regions.
        # Thus, we use systematic 2D sweeps across dimension pairs (i, j) to intentionally scrape the domain boundaries and guarantee finding these "hard-to-reach" neurons.
        p2_delta_x = 1000000.0

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

                for s_idx in range(200):
                    if len(extracted_W2_list) >= D_h2:
                        break
                    H1_walk += V_dir
                    current_G = get_macro_grad_batched(H1_walk, p2_delta_x, p2_queries, PROJ_VEC)
                    if current_G is not None:
                        process_gradients(current_G, prev_G, "Phase 2", f"{i:02d}-{j:02d}", s_idx)
                        prev_G = current_G

    if len(extracted_W2_list) < D_h2:
        print(f"Error: Insufficient W2 neurons extracted ({len(extracted_W2_list)}/{D_h2}). Rank deficient.")
        return None

    print("Phase 3: Sign extraction.")

    extracted_W2_raw = np.array(extracted_W2_list).T
    W2_inv = np.linalg.pinv(extracted_W2_raw)
    extracted_W2_final = np.zeros_like(extracted_W2_raw)

    def get_X_for_H1_no_B1(H1_tgt):
        M1_tgt = H1_tgt * q
        X_sub = np.dot(M1_tgt, W1_inv_float)
        X_out = np.zeros(D_in, dtype=np.int64)
        X_out[sub_indices] = np.round(X_sub).astype(np.int64)
        return X_out

    SIR = []

    for i in range(extracted_W2_raw.shape[1]):
        raw_w2 = extracted_W2_raw[:, i]

        M2_step = np.zeros(D_h2)
        M2_step[i] = 5000000.0
        H1_step = np.dot(M2_step, W2_inv)

        min_H1_req = np.abs(2 * H1_step) + 15000
        bounds = [(min_H1_req[k], None) for k in range(D_h1)]

        success = False
        target_candidates = [0, 500000, -500000, 1500000, -1500000, 2500000]

        for target_bias in target_candidates:
            res = linprog(c=np.ones(D_h1), A_eq=[raw_w2], b_eq=[target_bias], bounds=bounds, method="highs")
            if res.success:
                success = True
                break

        if not success:
            print(f"Warning: Sign extraction failed for L2 Neuron {i:02d}. Retaining original sign.")
            extracted_W2_final[:, i] = raw_w2
            continue

        H1_base = np.round(res.x).astype(np.float64)

        X_pos_1 = get_X_for_H1_no_B1(H1_base + 1 * H1_step)
        X_pos_2 = get_X_for_H1_no_B1(H1_base + 2 * H1_step)
        X_neg_1 = get_X_for_H1_no_B1(H1_base - 1 * H1_step)
        X_neg_2 = get_X_for_H1_no_B1(H1_base - 2 * H1_step)

        Y1_p = np.mean(oracle_func(np.tile(X_pos_1, (p3_queries, 1))), axis=0)
        Y2_p = np.mean(oracle_func(np.tile(X_pos_2, (p3_queries, 1))), axis=0)
        diff_pos = np.sum(np.abs(Y2_p - Y1_p))

        Y1_n = np.mean(oracle_func(np.tile(X_neg_1, (p3_queries, 1))), axis=0)
        Y2_n = np.mean(oracle_func(np.tile(X_neg_2, (p3_queries, 1))), axis=0)
        diff_neg = np.sum(np.abs(Y2_n - Y1_n))

        status = ""
        if diff_pos > diff_neg:
            extracted_W2_final[:, i] = raw_w2
            status = f"Sign retained (+Amp: {diff_pos:.0f} > -Amp: {diff_neg:.0f})"
            SIR.append(abs(diff_pos / (diff_neg + 1e-4)))
        else:
            extracted_W2_final[:, i] = -raw_w2
            status = f"Sign corrected (-Amp: {diff_neg:.0f} > +Amp: {diff_pos:.0f})"
            SIR.append(abs(diff_neg / (diff_pos + 1e-4)))

        # print(f"L2 Neuron {i:02d} | {status}")

    print(f"W2 extraction complete.")
    # print(f"(Avg SIR: {sum(SIR) / len(SIR):.2f}).")
    return extracted_W2_final


def decrypt_B2(oracle_func, W1, B1, W2, D_in, D_h1, D_h2, q, num_bases=100, test_queries=100):
    print("Initiating B2 extraction...")

    sub_indices = find_rank_full_sub_indices(W1, D_h1)

    W1_sub = W1[sub_indices, :]
    W1_inv_float = np.linalg.inv(W1_sub)

    extracted_B2 = np.zeros(D_h2, dtype=np.int64)
    eff_W2 = np.dot(W1, W2)

    def get_intersect(x1, y1, x2, y2, x3, y3, x4, y4, c_idx):
        m_off = (y2[c_idx] - y1[c_idx]) / (x2 - x1)
        m_on = (y4[c_idx] - y3[c_idx]) / (x4 - x3)
        b_off = y1[c_idx] - m_off * x1
        b_on = y3[c_idx] - m_on * x3
        if m_on == m_off:
            return 0.0
        return (b_off - b_on) / (m_on - m_off)

    for j in range(D_h2):
        c = np.argmax(np.abs(eff_W2[:, j]))
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

            T_OFF2 = -2e6
            T_OFF1 = -1e6
            T_ON1 = 1e6
            T_ON2 = 2e6

            k_OFF2 = int((T_OFF2 - M2_base_act) / w_eff)
            k_OFF1 = int((T_OFF1 - M2_base_act) / w_eff)
            k_ON1 = int((T_ON1 - M2_base_act) / w_eff)
            k_ON2 = int((T_ON2 - M2_base_act) / w_eff)

            def get_Y_B2(k_val):
                X_test = X_base.copy()
                X_test[c] += k_val
                return np.mean(oracle_func(np.tile(X_test, (test_queries, 1))), axis=0)

            Y_OFF2 = get_Y_B2(k_OFF2)
            Y_OFF1 = get_Y_B2(k_OFF1)
            Y_ON1 = get_Y_B2(k_ON1)
            Y_ON2 = get_Y_B2(k_ON2)

            slope_OFF = (Y_OFF1 - Y_OFF2) / (k_OFF1 - k_OFF2)
            slope_ON = (Y_ON2 - Y_ON1) / (k_ON2 - k_ON1)
            out_c = np.argmax(np.abs(slope_ON - slope_OFF))

            k_star = get_intersect(k_OFF2, Y_OFF2, k_OFF1, Y_OFF1, k_ON1, Y_ON1, k_ON2, Y_ON2, out_c)
            M2_star = M2_base_act + k_star * w_eff

            candidate_B2s.append(-int(round(M2_star / q)))

        final_B2_val = stats.mode(candidate_B2s, keepdims=False).mode
        extracted_B2[j] = int(final_B2_val)

        # print(f"B2[{j:02d}] extracted: {extracted_B2[j]}")

    print("B2 extraction complete.")
    return extracted_B2


def decrypt_L_last(oracle_func, W_list, B_list, D_out, q):
    D_in = W_list[0].shape[0]
    D_h1 = W_list[0].shape[1]
    D_hidden = W_list[-1].shape[1]

    W1 = W_list[0]
    B1 = B_list[0]

    sub_indices = find_rank_full_sub_indices(W1, D_h1)
    W1_sub = W1[sub_indices, :]
    W1_inv_float = np.linalg.inv(W1_sub)

    # Traditional linear inversion (e.g., pseudoinverse) fails to account for the domain constraints imposed by ReLU layers.
    # Since it does not guarantee non-negative pre-activations, intermediate negative values are often clipped to zero.
    # This introduces a significant risk of inversion failure or makes exact backpropagation highly intractable across deep networks.
    # To robustly bypass this ReLU constraint, we implement a 'Demand-Driven' routing strategy:
    # 1. Calculate the exact "demand" required to safely overcome negative biases.
    # 2. Backpropagate this demand by tracing ONLY positive-weight paths.
    # 3. Flood these positive paths with massive activation values.
    # This guarantees target activation by strictly maintaining non-negative pre-activations, ensuring invariance to model depth and bias magnitudes.
    adv_X_list = []
    for j in range(D_hidden):
        target_H = np.zeros(D_hidden, dtype=np.float64)
        target_H[j] = 5000.0

        for k in range(len(W_list) - 1, 0, -1):
            W_k = W_list[k]
            B_k = B_list[k]

            M_req = np.zeros(W_k.shape[1], dtype=np.float64)
            for n in range(len(target_H)):
                if target_H[n] > 0:
                    M_req[n] = max(100.0, target_H[n] - B_k[n])

            influence = np.dot(W_k, M_req)
            target_H = np.where(influence > 0, 10000.0, 0.0)

        H1_target = target_H

        M1_tgt = np.zeros(D_h1, dtype=np.float64)
        for i in range(D_h1):
            if H1_target[i] > 0:
                M1_tgt[i] = (H1_target[i] - B1[i]) * q
            else:
                M1_tgt[i] = -10000.0

        X_sub = np.dot(M1_tgt, W1_inv_float)
        X_adv = np.zeros(D_in, dtype=np.int64)
        X_adv[sub_indices] = np.round(X_sub).astype(np.int64)
        adv_X_list.append(X_adv)

    m_eqs_random = 10
    X_random = np.random.randint(-1000, 1000, size=(m_eqs_random, D_in)).astype(np.int64)
    X_batch = np.vstack([X_random, np.array(adv_X_list)])
    m_eqs_total = X_batch.shape[0]

    Y_out = oracle_func(X_batch, sf_leak=False)

    A = X_batch.copy()
    for i in range(len(W_list)):
        M = np.dot(A, W_list[i])
        A = np.round(M / q) + B_list[i]
        A = np.maximum(0, A)

    A_prev = A

    if IntegerMatrix is None or LLL is None:
        raise ImportError("fpylll is not installed. Last layer extraction cannot proceed.")

    A_aug = np.hstack([A_prev, np.ones((m_eqs_total, 1), dtype=np.int64) * q])
    n_vars = A_aug.shape[1]
    W_ext = np.zeros((D_hidden, D_out), dtype=np.int64)
    B_ext = np.zeros(D_out, dtype=np.int64)
    S = 1
    Wt = 10000

    print(f"Initiating last layer extraction...")

    for j in range(D_out):
        T = Y_out[:, j] * q
        mat = IntegerMatrix(n_vars + 1, n_vars + 1 + m_eqs_total)

        for i in range(n_vars):
            mat[i, i] = 1
            for k in range(m_eqs_total):
                mat[i, n_vars + 1 + k] = S * int(A_aug[k, i])

        mat[n_vars, n_vars] = Wt
        for k in range(m_eqs_total):
            mat[n_vars, n_vars + 1 + k] = S * int(-T[k])

        LLL.reduction(mat)

        found = False
        for i in range(mat.nrows):
            if mat[i, n_vars] == Wt:
                sol = [mat[i, k] for k in range(n_vars)]
                found = True
                break
            elif mat[i, n_vars] == -Wt:
                sol = [-mat[i, k] for k in range(n_vars)]
                found = True
                break

        if found:
            W_ext[:, j] = sol[:-1]

            M_last = np.dot(A_prev, W_ext[:, j])
            Y_base = (M_last + (q // 2)) // q
            B_candidates = Y_out[:, j] - Y_base
            final_B = int(stats.mode(B_candidates, keepdims=False).mode)
            B_ext[j] = final_B
        else:
            print(f"Error: Class {j:02d} LLL inversion failed.")

    print("Last layer extraction complete.")
    return W_ext, B_ext