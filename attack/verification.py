import numpy as np
from scipy.optimize import linear_sum_assignment


def unshuffle_and_verify_L1(W1_ext, B1_ext, W1_true, B1_true, W2_true, tolerance=0):
    D_x, D_h = W1_true.shape
    pi_1 = {}
    has_forced_live_match = False

    for ext_idx in range(D_h):
        for true_idx in range(D_h):
            if true_idx in pi_1.values():
                continue

            diff_W = np.max(np.abs(W1_ext[:, ext_idx] - W1_true[:, true_idx]))
            diff_B = np.abs(B1_ext[ext_idx] - B1_true[true_idx])

            if diff_W <= tolerance and diff_B <= tolerance:
                pi_1[ext_idx] = true_idx
                break

    unmatched_ext = [i for i in range(D_h) if i not in pi_1.keys()]
    unmatched_true = [i for i in range(D_h) if i not in pi_1.values()]

    print(f"Equivalence check complete. (Exact matches within tolerance <= {tolerance}: {len(pi_1)}/{D_h})")

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
            true_idx = live_true_neurons[i]
            ext_idx = unmatched_ext[j]
            pi_1[ext_idx] = true_idx
            matched_ext_indices.append(ext_idx)
            # print(f"Forced assignment: Target(Active) {true_idx:02d} -> Extracted {ext_idx:02d} (Maximum deviation sum: {cost_matrix[i, j]})")

        unmatched_ext = [x for x in unmatched_ext if x not in matched_ext_indices]

    if dead_true_neurons:
        print("Arbitrary assignment for unresolved inactive (zero-weight) neurons.")
        for true_idx in dead_true_neurons:
            if not unmatched_ext:
                break
            ext_idx = unmatched_ext.pop(0)
            pi_1[ext_idx] = true_idx
            print(f"Arbitrary assignment: Target(Inactive) {true_idx:02d} -> Extracted {ext_idx:02d}")

    W1_recon = np.zeros_like(W1_true)
    B1_recon = np.zeros_like(B1_true)

    for ext_i, true_i in pi_1.items():
        W1_recon[:, true_i] = W1_ext[:, ext_i]
        B1_recon[true_i] = B1_ext[ext_i]

    # print("Recovered permutation mapping (Extracted -> Target):")
    for ext_i in range(D_h):
        if ext_i in pi_1:
            true_i = pi_1[ext_i]
            is_dead = np.all(W2_true[true_i, :] == 0)

            if is_dead:
                pass
                # print(f"  Extracted {ext_i:02d} -> Target {true_i:02d} [Inactive neuron]")
            else:
                err_W = np.max(np.abs(W1_ext[:, ext_i] - W1_true[:, true_i]))
                err_B = np.abs(B1_ext[ext_i] - B1_true[true_i])
                # print(f"  Extracted {ext_i:02d} -> Target {true_i:02d} [Maximum deviation | W: {err_W}, B: {err_B}]")
        else:
            pass
            # print(f"  Extracted {ext_i:02d} -> Orphaned")

    if has_forced_live_match:
        print("Error: Optimal assignment deviation exceeds tolerance threshold for active neurons. Extraction aborted.")
        return None, None, None

    print("Layer 1 structural recovery complete. Functional equivalence verified.")
    return W1_recon, B1_recon, pi_1


def unshuffle_and_verify_W2(W2_ext, W2_true, found_pi_1):
    D_h1, D_h2 = W2_true.shape
    has_forced_match = False
    W2_row_aligned = np.zeros_like(W2_ext)

    for ext_i, true_i in found_pi_1.items():
        W2_row_aligned[true_i, :] = W2_ext[ext_i, :]

    cost_matrix = np.zeros((D_h2, D_h2))
    for true_j in range(D_h2):
        for ext_j in range(D_h2):
            cost_matrix[true_j, ext_j] = np.max(np.abs(W2_row_aligned[:, ext_j] - W2_true[:, true_j]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    pi_2 = {}

    # print("Layer 2 permutation mapping (Extracted -> Target):")
    for true_j, ext_j in zip(row_ind, col_ind):
        pi_2[ext_j] = true_j
        err_W = cost_matrix[true_j, ext_j]

        if err_W == 0:
            pass
            # print(f"  Extracted {ext_j:02d} -> Target {true_j:02d} [Maximum deviation: {err_W}]")
        else:
            has_forced_match = True
            # print(f"  Sub-optimal alignment: Extracted {ext_j:02d} -> Target {true_j:02d} (Maximum deviation: {err_W} > 0)")

    W2_recon = np.zeros_like(W2_true)
    for ext_j, true_j in pi_2.items():
        W2_recon[:, true_j] = W2_row_aligned[:, ext_j]

    if has_forced_match:
        print("Error: Sub-optimal assignment detected (Deviation > 0). W2 recovery aborted.")
        return None, None

    res_w2 = np.array_equal(W2_recon, W2_true)

    if res_w2:
        print("W2 functional recovery and verification complete.")
        return W2_recon, pi_2

    print("Warning: W2 verification failed. Discrepancy detected in row alignment or column permutation logic.")
    return None, None


def unshuffle_and_verify_B2(B2_ext, B2_true, found_pi_2, tolerance=0):
    D_h2 = len(B2_true)
    B2_recon = np.zeros(D_h2, dtype=np.int64)

    for ext_i, true_i in found_pi_2.items():
        B2_recon[true_i] = B2_ext[ext_i]

    errors = np.abs(B2_recon - B2_true)
    max_error = np.max(errors)
    is_match = max_error <= tolerance

    if is_match:
        print(f"B2 functional recovery verified within tolerance {tolerance} (Maximum deviation: {max_error}).")
        return B2_recon

    print("Warning: B2 deviation exceeds tolerance.")
    # Debugging information:
    for i in range(D_h2):
        err = errors[i]
        if err > tolerance:
            pass
            # print(f"Tolerance exceeded at Target {i:02d}: Extracted {B2_recon[i]} vs Target {B2_true[i]} (Deviation: {err})")
    return None


def unshuffle_and_verify_L_last(W_last_ext, B_last_ext, W_last_true, B_last_true, found_pi_last, tolerance=0):
    D_last_hidden, D_out = W_last_true.shape
    W_last_recon = np.zeros_like(W_last_true)

    for ext_i, true_i in found_pi_last.items():
        W_last_recon[true_i, :] = W_last_ext[ext_i, :]

    w_match = np.array_equal(W_last_recon, W_last_true)

    errors = np.abs(B_last_ext - B_last_true)
    max_error = np.max(errors)
    b_match = max_error <= tolerance

    is_overall_match = w_match and b_match

    if is_overall_match:
        print(f"Final layer parameters successfully extracted. (Maximum bias deviation: {max_error})")
        return True, W_last_recon, B_last_ext

    print("Warning: Parameter discrepancy detected.")
    
    # Detailed failure analysis:
    if not b_match:
        pass
        # print("B_last discrepancy analysis:")
        err_b = np.max(np.abs(B_last_ext - B_last_true))
        # print(f"  Bias extraction deviation (Maximum: {err_b})")
        # print(f"  Extracted: {B_last_ext}")
        # print(f"  Target: {B_last_true}")

    if not w_match:
        pass
        # print("W_last discrepancy analysis:")
        for i in range(D_last_hidden):
            row_recon = W_last_recon[i, :]
            row_true = W_last_true[i, :]

            if not np.array_equal(row_recon, row_true):
                is_dead = np.all(row_true == 0)
                err_max = np.max(np.abs(row_recon - row_true))

                if is_dead:
                    pass
                    # print(f"  Target {i:02d} row: [Inactive neuron] Extracted functionally ignorable values.")
                elif np.all(row_recon == 0):
                    pass
                    # print(f"  Target {i:02d} row: [Critical truncation] Extracted values vanished to 0. (Maximum deviation: {err_max})")
                else:
                    pass
                    # print(f"  Target {i:02d} row: [Mismatch] Quantization/regression error detected. (Maximum deviation: {err_max})")

                # print(f"  Extracted: {str(list(row_recon)}")
                # print(f"  Target: {str(list(row_true)}")

    return False, W_last_recon, B_last_ext