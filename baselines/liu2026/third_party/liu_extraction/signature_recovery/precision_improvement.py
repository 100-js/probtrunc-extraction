import gc
import os
import numpy as np
import jax
import jax.numpy as jnp
import optax
import itertools
from .utils import (
    forward,
    matmul,
    get_hidden_layers,
    on_which_hidden_layer,
    getAllWeightsAndBiases,
    compute_weight_difference,
    AcceptableFailure,
)
from tqdm import tqdm
from .critical_point_search import do_better_sweep
from whitebox import getWeightsAndBiases
from scipy.optimize import minimize, least_squares
from scipy.linalg import null_space
import sympy as sp
from itertools import combinations
import math


# ==========
#  Functions for Precision Improvement
# ==========
def find_diverse_subsets(data, n):
    """
    data: List of binary vectors (list of lists or np.array), shape: N x m
    n: target subset size
    return: list of index-tuples of subsets whose bitwise OR is all-1
    """
    N = len(data)
    m = len(data[0])
    data = np.array(data)

    result = []
    for indices in itertools.combinations(range(N), n):
        subset = data[list(indices)]
        bitwise_or = np.bitwise_or.reduce(subset, axis=0)
        if np.all(bitwise_or == 1):
            result.append(indices)
        if len(result) > 1000:
            break
    return result


def find_good_rank_matrices(stacked_input_full, n):
    """
    stacked_input_full: the whole matrix of stacked inputs, shape: N x m
    n: target rank of the submatrix
    """

    # from top to bottom, find the largest rank n submatrix
    N, m = stacked_input_full.shape
    upper_index = 0
    lower_index = 0
    print("The target rank is", n)

    # diverse_sets = []
    for i in range(1, N):
        submatrix = stacked_input_full[:i, :]
        if np.linalg.matrix_rank(submatrix) == n:
            upper_index = i - 1
            break
    for i in range(N - 1, 0, -1):
        submatrix = stacked_input_full[i:, :]
        if np.linalg.matrix_rank(submatrix) == n:
            lower_index = i + 1
            break
    # for i in range(0, N - n + 1):
    #     submatrix = stacked_input_full[i : i + n - 1, :]
    #     diverse_sets.append(submatrix)
    return [stacked_input_full[:upper_index, :], stacked_input_full[lower_index:, :]]
    # return diverse_sets


def brute_force_max_row_subset_with_rank(stacked_input_full, target_rank):
    n_rows = stacked_input_full.shape[0]
    max_subset = []

    for subset_size in reversed(range(0, n_rows + 1)):
        print("Trying subset size:", subset_size)
        count = math.comb(n_rows, subset_size)
        print("Number of combinations:", count)
        # If the number
        if count > 1e7:
            break
        for row_indices in combinations(range(n_rows), subset_size):
            submatrix = stacked_input_full[list(row_indices), :]
            rank = np.linalg.matrix_rank(submatrix)
            if rank == target_rank:
                return submatrix, list(row_indices)
    return None, []


# Find rows not linearly dependent on the others
def find_linearly_independent_rows(args, stacked_input_full):
    """
    stacked_input_full: the whole matrix of stacked inputs, shape: N x m
    return: indices of linearly independent rows
    """
    # We use the first m rows as the initial set of independent rows
    independent_rows = []
    n = stacked_input_full.shape[0]
    m = stacked_input_full.shape[1]

    # Find a good starting point
    random_indices = np.random.choice(np.arange(0, n), size=m, replace=False)
    basic_submatrix = stacked_input_full[random_indices, :]
    U, S, Vh = np.linalg.svd(basic_submatrix)
    print("Singular values of basic_submatrix is", S)

    try_count = 0
    while S[-1] > 1e-10 or S[-2] < 1e-4:
        if try_count > 100:
            return independent_rows
        print(
            "The smallest singular value is bad. We need to find a new starting point"
        )
        # Choose a new matrix
        random_indices = np.random.choice(np.arange(0, n), size=m, replace=False)
        basic_submatrix = stacked_input_full[random_indices, :]
        U, S, Vh = np.linalg.svd(basic_submatrix)
        print("Singular values of basic_submatrix is", S)
        try_count += 1

    if args.layerID == 1:
        if args.dataset == "mnist":
            step = 20
        if args.dataset == "cifar10":
            step = 100
    else:
        step = 1

    submatrix = basic_submatrix.copy()
    step_num = 0
    for i in range(0, stacked_input_full.shape[0], step):
        print("Start index is", i)
        # for i in tqdm(range(m - 1, stacked_input_full.shape[0])):
        submatrix = basic_submatrix.copy()
        if args.layerID == 1:
            rank = np.linalg.matrix_rank(basic_submatrix)
        else:
            rank = np.linalg.matrix_rank(basic_submatrix, tol=1e-10)
        print("The shape of the submatrix is", submatrix.shape)
        print("The initial rank is", rank)

        upper_bound = min(i + step, stacked_input_full.shape[0])
        for k in range(i, upper_bound):
            if k not in random_indices:
                submatrix = np.vstack((submatrix, stacked_input_full[k, :]))
        if args.layerID == 1:
            new_rank = np.linalg.matrix_rank(submatrix)
        else:
            new_rank = np.linalg.matrix_rank(submatrix, tol=1e-10)
        print("After stacking, the new rank is", new_rank)

        if new_rank > rank:
            print("The rank is not good within this step")
            if args.dataset == "mnist":
                for j in range(i, upper_bound):
                    if j not in random_indices:
                        subsubmatrix = np.vstack(
                            (basic_submatrix, stacked_input_full[j, :])
                        )
                        if args.layerID == 1:
                            subsubmatrix_rank = np.linalg.matrix_rank(subsubmatrix)
                        else:
                            subsubmatrix_rank = np.linalg.matrix_rank(
                                subsubmatrix, tol=1e-10
                            )
                        if subsubmatrix_rank > rank:
                            print("Adding row", j, "new rank:", subsubmatrix_rank)
                            independent_rows.append(j)
                        else:
                            basic_submatrix = subsubmatrix.copy()
            if args.dataset == "cifar10":
                for j in range(i, upper_bound):
                    if j not in random_indices:
                        independent_rows.append(j)
        else:
            basic_submatrix = submatrix.copy()
    print("Found", len(independent_rows), "linearly independent rows")
    print("Indices of independent rows:", independent_rows)
    return independent_rows


def improve_precision(
    args,
    A,
    B,
    known_A,
    known_B,
    extracted_sequence,
    model,
    dimInput,
    dimOfPrevLayer,
    dimOfLayer,
    critical_groups,
):
    dead_neuron = 0
    for i in range(dimOfLayer):
        if np.sum(np.abs(known_A[:, i])) < 1e-8:
            dead_neuron += 1
    print("We need to improve the precision of", dimOfLayer - dead_neuron, "rows")

    weights_all, biases_all = getAllWeightsAndBiases(model)

    savePath = os.path.join(
        args.output_path,
        "stacked matrix",
        f"Seed{args.seed}-Count{args.Count}",
    )
    os.makedirs(savePath, exist_ok=True)
    print(f"stacked matrix will be saved to \n\t {savePath}")

    if args.real_attack == 1 and args.layerID > 1:
        if args.layerID - 1 == 1:
            recoveryPath = os.path.join(
                args.output_path,
                "recovered results",
                f"Seed{args.seed}-Count500-RealAttack{args.real_attack}",
                # f"Seed{args.seed}-Count1000-RealAttack{args.real_attack}",
            )
        else:
            recoveryPath = os.path.join(
                args.output_path,
                "recovered results",
                f"Seed{args.seed}-Count{args.Count}-RealAttack{args.real_attack}",
            )
        recovered_parameters = np.load(
            os.path.join(recoveryPath, f"recovered_{args.layerID - 1}.npz")
        )
        prev_extracted_sequence = recovered_parameters["extracted_sequence"]
        prev_extracted_bias = recovered_parameters["extracted_bias"]

        prev_bias_cheat = biases_all[args.layerID - 2]
        prev_bias_cheat = np.array(prev_bias_cheat)
        prev_bias_cheat = prev_bias_cheat[prev_extracted_sequence]
        prev_alpha = prev_extracted_bias / prev_bias_cheat

        print("prev_extracted_bias", prev_extracted_bias)
        print("prev_bias_cheat", prev_bias_cheat)
        print("prev_alpha", prev_alpha)
    else:
        prev_extracted_sequence = np.arange(known_A.shape[0])
        prev_alpha = np.ones(known_A.shape[0])

        print("prev_extracted_sequence", prev_extracted_sequence)
        print("prev_alpha", prev_alpha)

    last_layer_outputs = []
    stacked_last_layer_outputs = []
    weight_improved = known_A.copy()
    bias_improved = known_B.copy()

    for i in range(len(critical_groups)):
        print("-------------------------------------------")
        extracted_weight = known_A[:, i].flatten()
        real_weight = weights_all[args.layerID - 1][:, extracted_sequence[i]]
        real_weight = real_weight.flatten()
        real_weight = real_weight[prev_extracted_sequence]
        for j in range(len(real_weight)):
            real_weight[j] = real_weight[j] / prev_alpha[j]

        if args.layerID == 1:
            non_zero_index = np.where(extracted_weight != 0)[0]

            extracted_weight = extracted_weight[non_zero_index]
            real_weight = real_weight[non_zero_index]
            if len(non_zero_index) < weight_improved.shape[0]:
                weight_improved = weight_improved[non_zero_index]

        last_layer_output = []
        critical_group = critical_groups[i]
        for critical_point in critical_group:
            # Get the output of the target layer
            hidden_layer = forward(critical_point, A, B, with_relu=True)
            last_layer_output.append(np.array(hidden_layer))
            # output = matmul(hidden_layer, known_A[:, i], known_B[i])
            # print("[Recovered] Output of the target layer:", output)

            # output_real = forward(
            #     critical_point,
            #     weights_all[: args.layerID],
            #     biases_all[: args.layerID],
            # )
            # print("[Real] Output of the target layer:", output_real[neuron_id])

        last_layer_outputs.append(np.array(last_layer_output))
        stacked_last_layer_output = np.vstack(last_layer_outputs[-1])
        if args.layerID == 1:
            stacked_last_layer_output = stacked_last_layer_output[:, non_zero_index]
        stacked_last_layer_outputs.append(np.array(stacked_last_layer_output))

        # improve the precision of each neuron in known_A
        print("===========================================")
        stacked_input = stacked_last_layer_outputs[i].copy()

        # Method 1: directly solve the linear systems
        # add 1 to each row of the last layer output
        print("==================[Method 1]===========================")
        stacked_input_full = np.hstack(
            (stacked_input, np.ones((stacked_input.shape[0], 1)))
        )

        np.savez(
            os.path.join(savePath, f"recovered_{args.layerID}.npz"),
            matrix=stacked_input_full,
        )

        # if args.real_attack == 1 and args.layerID == 1:
        # Remove some ''bad'' critical points
        # The published experiments use widths 8 and 16.  Keep the width-16
        # tolerance as the default for our width-20 first-layer comparison.
        tol = 1e-10
        if dimOfLayer == 8:
            tol = 1e-5

        if args.layerID == 1:
            rank = np.linalg.matrix_rank(stacked_input_full)
        else:
            rank = np.linalg.matrix_rank(stacked_input_full, tol=tol)
        print("rank of stacked_input_full", rank)
        if rank == stacked_input_full.shape[1]:
            print(
                "The stacked_input_full is full rank, we cannot find a null space vector."
            )
            independent_rows = find_linearly_independent_rows(args, stacked_input_full)
            # remove the bad rows independent_rows
            stacked_input_full = np.delete(stacked_input_full, independent_rows, axis=0)

        bad_indices = []
        if args.layerID == 1:
            null_space_vector = null_space(stacked_input_full)
        else:
            null_space_vector = null_space(stacked_input_full, rcond=tol)

        print("null_space_vector full", null_space_vector)
        print("rank of stacked_input_full", np.linalg.matrix_rank(stacked_input_full))
        print(
            "rank of stacked_input_full(tol)",
            np.linalg.matrix_rank(stacked_input_full, tol=tol),
        )
        U, S, Vh = np.linalg.svd(stacked_input_full)
        print("Singular values of stacked_input_full", S)

        find_soln = False
        rank = np.linalg.matrix_rank(stacked_input_full, tol=tol)
        if args.layerID == 1 and rank < stacked_input_full.shape[1] - 1:
            print(
                "The rank of stacked_input_full is less than the number of columns - 1, we cannot find a unique solution."
            )
            continue

        if null_space_vector.shape[1] >= 1 and find_soln == False:
            print("We find the null space vector from the full matrix")
            print("The shape of stacked_input_full is", stacked_input_full.shape)

            for k in range(null_space_vector.shape[1]):
                print("The index of null_space_vector", k)
                factor = 0.0
                diff = 0.0

                current_vector = null_space_vector[:, k].flatten()
                zero_index = np.where(extracted_weight == 0)[0]
                vector_valid = True
                for l in zero_index:
                    if np.abs(current_vector[l]) > 1e-2:
                        print(
                            "[fatal error] current_vector",
                            current_vector[l],
                            "at index",
                            l,
                        )
                        vector_valid = False
                        break
                    current_vector[l] = 0
                if not vector_valid:
                    continue

                # print("current_vector", current_vector)

                for l in range(len(extracted_weight)):
                    if extracted_weight[l] == 0 or current_vector[l] == 0:
                        continue
                    if factor == 0:
                        factor = current_vector[l] / extracted_weight[l]
                        break
                for l in range(len(extracted_weight)):
                    if extracted_weight[l] == 0 or current_vector[l] == 0:
                        continue
                    diff += abs(current_vector[l] - factor * extracted_weight[l])

                print("diff", diff)
                if diff < 1e-2:
                    find_soln = True
                    null_space_vector = current_vector
                    # print("null_space_vector", null_space_vector)

                    if null_space_vector[0] < 0:
                        null_space_vector = -null_space_vector
                    break

        if find_soln == False:
            print("We don't find the null space vector from the full matrix")
            if args.layerID > 1:
                continue

            null_space_vectors = []
            binary_stacked_input = np.where(stacked_input >= 0, 1, 0)
            active_weights = np.count_nonzero(extracted_weight)
            print("number of active_weights", active_weights)

            if args.layerID == 1:
                max_submatrix, selected_indicies = brute_force_max_row_subset_with_rank(
                    stacked_input_full, active_weights
                )
                diverse_sets = [max_submatrix]

                if max_submatrix is None:
                    diverse_sets = find_good_rank_matrices(
                        stacked_input_full, active_weights + 1
                    )
                    rank_1 = np.linalg.matrix_rank(diverse_sets[0])
                    rank_2 = np.linalg.matrix_rank(diverse_sets[1])
                    print("rank of diverse_sets[0]", rank_1)
                    print("rank of diverse_sets[1]", rank_2)

                    if rank_1 < rank_2:
                        diverse_sets = [diverse_sets[1]]
                    else:
                        diverse_sets = [diverse_sets[0]]
                print("diverse_sets", diverse_sets)
            else:
                diverse_sets = find_diverse_subsets(
                    binary_stacked_input, active_weights
                )

            if len(diverse_sets) == 0:
                print(
                    "No diverse sets found, using all rows as a single set for null space calculation."
                )
                diverse_sets = [np.arange(stacked_input.shape[0])]

            for j in range(len(diverse_sets)):
                print("Diverse set", j)
                print("Diverse set", diverse_sets[j])
                # print("The shape of diverse_sets[j] is", diverse_sets[j].shape)

                # stacked_input_partial = stacked_input_full[diverse_sets[j], :]
                stacked_input_partial = diverse_sets[j]
                # print("stacked_input_partial", stacked_input_partial)
                print(
                    "rank of stacked_input_partial",
                    np.linalg.matrix_rank(stacked_input_partial),
                )
                print("shape of stacked_input_partial", stacked_input_partial.shape)

                if args.layerID == 1:
                    print(
                        "output of stacked_input_partial @ known_A[:, i] + known_B[i]"
                    )
                    print(stacked_input_partial.shape)
                    print(extracted_weight.shape)
                    print(
                        stacked_input_partial @ np.append(extracted_weight, known_B[i])
                    )
                    print(
                        "output of stacked_input_partial @ known_A[:, i] + known_B[i] (real)"
                    )
                    print(stacked_input_partial @ np.append(real_weight, known_B[i]))

                null_space_vector = null_space(stacked_input_partial)
                # print("null_space_vector all", null_space_vector)

                for k in range(null_space_vector.shape[1]):
                    print("The index of null_space_vector", k)
                    factor = 0.0
                    diff = 0.0

                    current_vector = null_space_vector[:, k].flatten()
                    zero_index = np.where(extracted_weight == 0)[0]
                    # print("zero_index", zero_index)

                    vector_valid = True
                    for l in zero_index:
                        if np.abs(current_vector[l]) > 1e-2:
                            print(
                                "[fatal error] current_vector",
                                current_vector[l],
                                "at index",
                                l,
                            )
                            vector_valid = False
                            break
                        current_vector[l] = 0
                    if not vector_valid:
                        continue

                    for l in range(len(extracted_weight)):
                        if extracted_weight[l] == 0 or current_vector[l] == 0:
                            continue
                        if factor == 0:
                            factor = current_vector[l] / extracted_weight[l]
                            break
                    for l in range(len(extracted_weight)):
                        if extracted_weight[l] == 0 or current_vector[l] == 0:
                            continue
                        diff += abs(current_vector[l] - factor * extracted_weight[l])

                    print("diff", diff)

                    if diff < 1e-2:
                        find_soln = True
                        null_space_vector = current_vector
                        # print("null_space_vector", null_space_vector)

                        if null_space_vector[0] < 0:
                            null_space_vector = -null_space_vector

                        null_space_vectors.append(null_space_vector)

                        diff_improved = compute_weight_difference(
                            null_space_vector[:-1],
                            real_weight,
                        )
                        print(
                            "[Partial]Difference between improved and real weights:",
                            diff_improved,
                        )
                        break

            null_space_vector = np.mean(null_space_vectors, axis=0)
        # print("final null_space_vector", null_space_vector)

        if np.all(np.isnan(null_space_vector)):
            print(
                "Warning: The null space vector is NaN, using the original extracted weight."
            )
            null_space_vector = extracted_weight.copy()
            null_space_vector = np.append(
                null_space_vector, known_B[i]
            )  # append the bias as well

        x_opt = null_space_vector[:-1]
        c_opt = null_space_vector[-1]

        diff_original = compute_weight_difference(
            extracted_weight,
            real_weight,
        )
        diff_improved = compute_weight_difference(
            x_opt,
            real_weight,
        )
        print("************************************************************")
        print("This is the original", extracted_sequence[i], "th neuron")
        print(
            "[Precision Improvement]Difference between original and real weights:",
            diff_original,
        )
        print(
            "[Precision Improvement]Difference between improved and real weights:",
            diff_improved,
        )

        # keep the improved sign as the original sign
        sign = np.sign(extracted_weight[0] / x_opt[0])
        x_opt = x_opt * sign
        c_opt = c_opt * sign

        print("The index of the neuron", i)
        weight_improved[:, i] = x_opt
        bias_improved[i] = c_opt
        print(weight_improved[:, i].flatten())

    return weight_improved, bias_improved
