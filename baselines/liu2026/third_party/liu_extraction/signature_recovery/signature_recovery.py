import numpy as np
import time
import os
import heapq
import itertools
import scipy
from tqdm import tqdm

from .critical_point_search import (
    sweep_for_critical_points,
    do_better_sweep,
)
from .utils import (
    forward,
    matmul,
    get_hidden_layers,
    getAllWeightsAndBiases,
    on_which_hidden_layer,
    get_query_counts,
    check_activation,
    get_second_grad_unsigned,
    compute_weight_difference,
    predict_manual_fast,
    AcceptableFailure,
    GatherMoreData,
)
from .solve_noise_issue.visualize_noise_issue import visualize_filter_performance
from .solve_noise_issue.filter_out_noise_critical_points import (
    is_on_target_layer,
    check_filtered_critical_points_quality,
)
from .compute_critical_point_values import (
    get_ratios_lstsq,
    gather_ratios,
    gather_ratios_without_filter,
)
from .cluster_critical_points.classify_critical_points import (
    identify_good_critical_points,
)
from .cluster_critical_points.cluster_critical_points import (
    cluster_critical_points,
    check_clustered_component_quality,
)
from .follow_critical_hyperplane import get_more_crit_pts
from .cluster_critical_points.merge_good_critical_points import (
    merge_good_critical_points_into_a_specific_component,
    check_components_quality,
)
from .cluster_critical_points.merge_bad_critical_points import (
    merge_bad_critical_points_into_a_specific_component,
    merge_bad_critical_points_with_each_other,
)
from .cluster_critical_points.operations_on_components import (
    add_good_components_back,
)
from .cluster_critical_points.merge_extra_critical_points import (
    merge_extra_cps_into_a_specific_component,
)
from .precision_improvement import improve_precision
from scipy.linalg import null_space
import sympy as sp
import gurobipy as gp
from gurobipy import GRB
from sign_recovery.find_mergeable_critical_points import (
    find_mergeable_critical_points,
    get_mergeable_critical_points,
    SOE_for_signs_merged,
)
from sign_recovery.sign_recovery import recoverSign
from sign_recovery.neuron_wiggle import (
    neuron_wiggle_to_find_removable_columns,
    neuron_wiggle_to_get_all_signs,
)
from sign_recovery.get_nw_results import get_mnist_nw_results, get_cifar_nw_results

from typing import Union, Tuple
import math


def seconds_to_h_m_s(s: Union[int, float]) -> Tuple[int, int, int]:
    total_seconds = int(s)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return hours, minutes, seconds


def approximate_power_of_two(n: int) -> str:
    if n == 0:
        return -1
    m = math.log2(n)
    m_rounded = round(m, 2)
    return m_rounded


# The class of each critical point
class CriticalPoint:
    def __init__(self, index, ratio, point, h_matrix):
        self.index = index
        self.ratio = ratio
        self.point = point
        self.h_matrix = h_matrix


# ==========
#  Functions for Signature Recovery
def row_echelon_form(A):
    A = A.copy()
    m, n = A.shape
    row = 0

    for col in range(n):
        if row >= m:
            break

        pivot = np.argmax(np.abs(A[row:, col])) + row
        if A[pivot, col] == 0:
            continue

        A[[row, pivot]] = A[[pivot, row]]

        A[row] = A[row] / A[row, col]

        for r in range(row + 1, m):
            A[r] -= A[r, col] * A[row]

        row += 1

    return A


def find_subarrays_less_than_target(arr, target):
    result = []
    result_index = []
    current = []
    index = []

    for i, num in enumerate(arr):
        if num < target:
            current.append(num)
            index.append(i)
        else:
            if current:
                result.append(current)
                result_index.append(index)

                current = []
                index = []
    if current:
        result.append(current)
        result_index.append(index)

    return result, result_index


def get_rank_of_point(point, A, B):
    rank = 1e8
    # For layer 1
    output = np.matmul(point, A[0]) + B[0]
    output = output * (output > 0)

    layer_rank = np.sum(output != 0)
    if layer_rank < rank:
        rank = layer_rank
    # print("The rank of the point in layer 1 is", layer_rank)
    for i in range(1, len(A)):
        output = np.matmul(output, A[i]) + B[i]
        output = output * (output > 0)

        layer_rank = np.sum(output != 0)
        # print("The rank of the point in layer", i + 1, "is", layer_rank)
        if layer_rank < rank:
            rank = layer_rank
    # print("The rank of the point is", rank)

    return rank


def get_activation_pattern(args, model, point):
    weights_all, biases_all = getAllWeightsAndBiases(model)

    output = np.matmul(point, weights_all[0]) + biases_all[0]
    output = output * (output > 0)
    print("The active neurons in layer 1 are", np.where(output != 0)[0])

    for i in range(1, len(weights_all) - 1):
        output = np.matmul(output, weights_all[i]) + biases_all[i]
        output = output * (output > 0)
        print("The active neurons in layer", i + 1, "are", np.where(output != 0)[0])


# ==========
# Main function for signature recovery
def recoverCritPts_Signature(
    args,
    model,
    weights,
    biases,
    layerId,
    special,
    dataset=None,
    savePath=None,
):
    """
    This is the function for signature recovery adapted from Carlini.
    special: args.quantized. Whether quantized or not.
    If float16 then specify this as 1 and if float32 then specify this as 2 if float64 then specify this as 0.
    Beware that with float64 this will run the precision improvement function which is really not great for mnist models.
    Preferably choose float32.
    """
    if args.dataset == "cifar10":
        recovery_file_path = os.path.join(
            savePath,
            f"output_index_{args.output_index}",
            f"recovery_results_layer_{args.layerID}.txt",
        )
    else:
        recovery_file_path = os.path.join(
            savePath, f"recovery_results_layer_{args.layerID}.txt"
        )
    # Ensure the directory exists before writing to the file
    os.makedirs(os.path.dirname(recovery_file_path), exist_ok=True)

    # Return parameters:
    PARAM_SEARCH_AT_LOCATION = 1e2
    if len(model.layers) == 3:
        PARAM_SEARCH_AT_LOCATION = 1e4
    dimOfInput, dimOfPrevLayer, dimOfLayer = (
        weights[0].shape[0],
        weights[layerId - 1].shape[0],
        weights[layerId - 1].shape[1],
    )

    # A and B contain the real weights and biases of layers [0, LAYER-1]
    # We just assume we have recovered the previous layers
    # Note: if we want to run the end-to-end recovery, we need to use the extracted weights and biases
    if args.real_attack == 0:
        A, B = weights[:-1], biases[:-1]
    if args.real_attack == 1:
        A, B = [], []
        dead_neurons = []
        extracted_sequence = []

        # load the extracted weights and biases
        for i in range(1, args.layerID):
            print("Get recovered weights and biases for layer", i)
            if i == 1:
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
                os.path.join(recoveryPath, f"recovered_{i}.npz")
            )
            for key in recovered_parameters.files:
                print(key, recovered_parameters[key].dtype)

            A.append(recovered_parameters["extracted_normal"])
            B.append(recovered_parameters["extracted_bias"])
            extracted_sequence.append(recovered_parameters["extracted_sequence"])
            print(
                "The sequence of columns in A is",
                recovered_parameters["extracted_sequence"],
            )

            dead_neuron = []
            for index, b in enumerate(recovered_parameters["extracted_bias"]):
                if abs(b) == 0:
                    dead_neuron.append(index)
            dead_neurons.append(dead_neuron)
        print("The dead neurons from the previous layer are", dead_neurons)

    # Special branch for neuron wiggle
    if args.real_attack == 4:
        target_layer = args.layerID - 1
        weights_all, biases_all = getAllWeightsAndBiases(model)
        previous_weights = weights_all[:target_layer]
        previous_biases = biases_all[:target_layer]
        target_weights = weights_all[target_layer]
        target_biases = biases_all[target_layer]
        dead_neurons = []

        # Geerate random signs and apply them to the real weights and biases
        # random_signs = np.random.choice([-1, 1], size=target_weights.shape[1])
        # print("The random signs are", random_signs)

        # for neuron_id in range(target_weights.shape[1]):
        #     target_weights[:, neuron_id] = (
        #         target_weights[:, neuron_id] * random_signs[neuron_id]
        #     )
        #     target_biases[neuron_id] = (
        #         target_biases[neuron_id] * random_signs[neuron_id]
        #     )

        sign_confidence, guessed_sign_neuron_wiggle, dead_neurons = (
            neuron_wiggle_to_get_all_signs(
                args,
                model,
                target_weights,
                target_biases,
                target_layer,
                previous_weights,
                previous_biases,
                dead_neurons,
            )
        )
        # print("The random signs are", random_signs)
        print("The sign confidence levels are", sign_confidence)
        print("The guessed signs from neuron_wiggle are", guessed_sign_neuron_wiggle)
        print("The dead neurons are", dead_neurons)

        real_signs = np.ones_like(guessed_sign_neuron_wiggle)
        for dead_index in dead_neurons:
            real_signs[dead_index] = 0
        wrong_index = np.where(
            np.array(guessed_sign_neuron_wiggle) != np.array(real_signs)
        )[0]
        print("The wrong recovered signs are at indices", wrong_index)
        print("The number of wrong recovered signs is", len(wrong_index))
        print("Their confidence levels are", np.array(sign_confidence)[wrong_index])
        return

    # Special branch for our sign recovery method, SOE+Wiggle
    # confidence strategy and extension strategy
    # We want to scale the sign recovery to larger layers
    if args.real_attack == 3:
        target_layer = args.layerID - 1
        weights_all, biases_all = getAllWeightsAndBiases(model)
        previous_weights = weights_all[:target_layer]
        previous_biases = biases_all[:target_layer]
        target_weights = weights_all[target_layer]
        target_biases = biases_all[target_layer]

        # Step 1. Do the neuron wiggle and list the results based on the confidence level
        # For the large networks, the neuron wiggle takes a long time, so we just ignore it here
        # sign_confidence, guessed_sign_neuron_wiggle = neuron_wiggle_to_get_all_signs(
        #     args,
        #     model,
        #     target_weights,
        #     target_biases,
        #     target_layer,
        #     previous_weights,
        #     previous_biases,
        #     dead_neurons,
        # )
        # # print("The random signs are", random_signs)
        # print("The sign confidence levels are", sign_confidence)
        # print("The guessed signs from neuron_wiggle are", guessed_sign_neuron_wiggle)

        # get_mnist_nw_results
        (
            dead_neurons,
            almost_dead_neurons,
            sign_confidence,
            guessed_sign_neuron_wiggle,
            error_neurons,
        ) = get_cifar_nw_results(args.layerID)
        dead_neurons += almost_dead_neurons
        dead_neurons = sorted(list(set(dead_neurons)))
        for neuron_id in dead_neurons:
            guessed_sign_neuron_wiggle[neuron_id] = 0
            sign_confidence[neuron_id] = 0

        ############################
        # Confidence strategy
        ############################
        # """
        random_point_num = 1000
        candidate_point_num = 50
        selected_point_num = 10

        # Sort signs based on the confidence levels
        sorted_sign_indices = np.argsort(sign_confidence)[::-1]
        print("The sorted indices based on confidence levels are", sorted_sign_indices)
        for i in sorted_sign_indices:
            print(
                "Neuron index",
                i,
                "has confidence level",
                sign_confidence[i],
                "and guessed sign",
                guessed_sign_neuron_wiggle[i],
            )
        print("********************************")

        # Step 2. Find a large set of points and compute the least confidence level required by each point
        # Finally, find a subset of points that can use the highest least confidence level
        shape = model.input_shape[1:]
        counter = itertools.count()

        # lowest highest confidence level, rank of the point, unique counter, point, inactive_indexes
        cl_per_point = [(-1, -1, next(counter), None, None)] * candidate_point_num
        heapq.heapify(cl_per_point)

        def heap_add(score, rank, info_1, info_2):
            item = (score, rank, next(counter), info_1, info_2)
            if score > cl_per_point[0][0]:
                heapq.heapreplace(cl_per_point, item)
            if score == cl_per_point[0][0] and rank > cl_per_point[0][1]:
                heapq.heapreplace(cl_per_point, item)

        for point_index in tqdm(range(random_point_num)):
            point = np.random.normal(0, 5, size=shape).flatten()
            point_rank = get_rank_of_point(point, previous_weights, previous_biases)
            if point_rank == 0:
                continue

            rank_flag = False
            inactive_indexes = []
            layer_input = forward(point, previous_weights, previous_biases, True)
            for neuron_index in sorted_sign_indices:
                if neuron_index in dead_neurons:
                    continue

                output = (
                    np.matmul(layer_input, target_weights[:, neuron_index])
                    + target_biases[neuron_index]
                )
                output_at_neuron = output * guessed_sign_neuron_wiggle[neuron_index]
                if output_at_neuron < 0:
                    inactive_indexes.append(neuron_index)

                if (
                    point_rank
                    >= dimOfLayer - len(dead_neurons) - len(inactive_indexes) + 2
                ):
                    # +2 for an overdetermined system
                    # No need to continue, already reach the requirement
                    rank_flag = True
                    break

            # Find the position to insert this point, if possible
            if rank_flag:
                least_CL_neuron_id = inactive_indexes[-1]
                heap_add(
                    sign_confidence[least_CL_neuron_id],
                    point_rank,
                    point,
                    inactive_indexes,
                )
            else:
                heap_add(
                    0,
                    point_rank + len(inactive_indexes),
                    # point_rank,
                    point,
                    inactive_indexes,
                )
        # Sort points based on the confidence levels
        # first by confidence level and then by rank
        cl_per_point = sorted(cl_per_point, key=lambda x: (x[0], x[1]), reverse=True)
        for point_info in cl_per_point:
            print(
                "Point with least confidence level",
                point_info[0],
                "and rank",
                point_info[1],
                " has inactive indexes ",
                len(point_info[4]),
            )

        # Step 3. Iterate through the candidate points and try to recover the signs
        final_signs = []
        largest_cl_original = 0
        largest_cl_extension = 0
        valid_point_counter = 0
        for i in tqdm(range(candidate_point_num)):
            print("\n*************************************")
            print("Processing candidate point", i + 1)

            least_cl = cl_per_point[i][0]
            if least_cl == -1:
                print("This is an empty point, skipping")
                continue

            start_point = cl_per_point[i][3]
            inactive_indexes = cl_per_point[i][4]
            inactive_count = len(inactive_indexes)
            point_rank = get_rank_of_point(
                start_point, previous_weights, previous_biases
            )
            print("The rank of the point is", point_rank)
            print("inactive_indexes are", inactive_indexes)
            print("The last inactive index is", inactive_indexes[-1])
            print("The point uses inactive count", inactive_count)
            print(
                "[confidence Level] The point has least confidence level of its inactive neurons as",
                least_cl,
            )

            # The output of the target layer, without signs
            output = forward(start_point, previous_weights, previous_biases, True)
            output = np.matmul(output, target_weights) + target_biases
            original_output_signs = np.sign(output)

            # mnist
            # Layer 2
            # threshold = 1e-4
            # Layer 3
            # threshold = 8e-5
            # Layer 4
            # threshold = 1e-2

            # cifar
            threshold = 1e-4
            # Layer 2,3,4,7
            if args.layerID in [2, 3, 4, 7]:
                threshold = 1e-4
            # layer 5
            if args.layerID == 5:
                threshold = 1e-5
            # layer 6
            if args.layerID == 6:
                threshold = 1e-3
            print("The proper threshold for the first point is", threshold)

            if point_rank >= dimOfLayer - len(dead_neurons) - inactive_count + 2:
                # This process can directly solve the signs
                # and check the quality if the point
                print("[Special] This point can satisfy the SOE requirement")
                mergeable_critical_points = [start_point]
                all_remove_columns = []
                error_flag = False
                for neuron_id in range(dimOfLayer):
                    if neuron_id in inactive_indexes or neuron_id in dead_neurons:
                        all_remove_columns.append(neuron_id)
                    if (
                        neuron_id in inactive_indexes
                        and neuron_id in error_neurons
                        and neuron_id not in dead_neurons
                    ):
                        print("[error neuron in inactive neurons]", neuron_id)
                        error_flag = True
                        break
                if error_flag:
                    continue

                print("length of all_remove_columns is now", len(all_remove_columns))
                rank_Y, matrix_Y, vector_Z = SOE_for_signs_merged(
                    args,
                    model,
                    mergeable_critical_points,
                    target_layer,
                    previous_weights,
                    previous_biases,
                    target_weights,
                    target_biases,
                    all_remove_columns,
                )
                U, S, Vt = np.linalg.svd(matrix_Y)
                print("The singular values of matrix_Y are", S)
                print("[Final Result] The rank of the matrix Y is", rank_Y)
                print("The target rank is", dimOfLayer - len(all_remove_columns))

                if np.sum(vector_Z) == 0:
                    print("vector_Z is all zeros, skipping")
                    continue

                # Sign recovery process
                # remove the columns in remove_columns from the matrix_Y
                # matrix_Y_without_columns = np.delete(matrix_Y, remove_columns, axis=1)
                matrix_Y_without_columns = matrix_Y.copy()
                soln, *rest = np.linalg.lstsq(
                    np.array(
                        matrix_Y_without_columns, dtype=np.float64
                    ),  # float64 #float32
                    np.array(vector_Z, dtype=np.float64),
                    # 1e-5,
                    # tol for small networks
                )  # float64 #float32
                # print("The soln from the SOE (Y, 1e-5) is", soln)
                print("The soln from the SOE (Y) is", soln)
                # print(
                #     "Residual is",
                #     vector_Z - np.matmul(np.array(matrix_Y_without_columns), soln),
                # )

                soln_index = 0
                recovered_signs = []
                residual = vector_Z - np.matmul(
                    np.array(matrix_Y_without_columns), soln
                )

                max_res_item = max(abs(residual))
                print("The max_res_item is", max_res_item)

                active_neurons = []
                for neuron_id in range(dimOfLayer):
                    if neuron_id in all_remove_columns:
                        recovered_signs.append(guessed_sign_neuron_wiggle[neuron_id])
                    else:
                        # Get the output sign of this neuron
                        output_sign = original_output_signs[neuron_id]
                        # 1e-5 3
                        if abs(soln[soln_index]) < threshold:
                            # This neuron is inactive
                            recovered_signs.append(-1 * output_sign)
                        else:
                            print(
                                "neuron_id",
                                neuron_id,
                                "is active",
                                "soln value is",
                                soln[soln_index],
                            )
                            # This neuron is active
                            recovered_signs.append(1 * output_sign)
                            active_neurons.append(neuron_id)
                        soln_index += 1
                print("Active neurons are", active_neurons)
                # print("The original output signs are", original_output_signs)
                print("[Direct] The recovered signs are", recovered_signs)

                print("[Direct] The rank of the point is", point_rank)
                print("[Direct] The least confidence level is", least_cl)
                print(
                    "[Direct] The total number of inactive neurons is",
                    np.sum(original_output_signs == -1),
                )
                print("[Direct] Required inactive count is", len(inactive_indexes))
                print("[Direct] The number of dead neurons is", len(dead_neurons))
                # breakpoint()
            # continue
            if least_cl > largest_cl_original:
                largest_cl_original = least_cl

            # Do the extension process to try to increase the rank
            mergeable_critical_points, matrix_Y, vector_Z = (
                get_mergeable_critical_points(
                    args,
                    model,
                    start_point,
                    target_layer,
                    previous_weights,
                    previous_biases,
                    target_weights,
                    target_biases,
                    dead_neurons,
                )
            )
            U, S, Vt = np.linalg.svd(matrix_Y)
            print("The singular values of matrix_Y are", S)

            # rank_Y = np.linalg.matrix_rank(matrix_Y, tol=1e-5)
            rank_Y = np.linalg.matrix_rank(matrix_Y)
            print("The rank of the point is", point_rank)
            print(
                "The point uses inactive count",
                inactive_count,
            )
            print("[Final Result] The rank of the matrix Y is", rank_Y)
            print("The target rank is", dimOfLayer - len(dead_neurons))

            if rank_Y >= dimOfLayer - len(dead_neurons) - inactive_count + 2:
                print("[Special] SOE requirement is satisfied after neuron wiggle")

                # Find which columns can be removed using results from neuron wiggle
                remove_columns = []
                # compute the new index after removing dead neurons
                for inactive_id in inactive_indexes:
                    # get the position of inactive_id in dead_neurons
                    relative_dead_index = 0
                    while (
                        relative_dead_index < len(dead_neurons)
                        and dead_neurons[relative_dead_index] < inactive_id
                    ):
                        relative_dead_index += 1
                    adjusted_index = inactive_id - relative_dead_index
                    remove_columns.append(adjusted_index)

                    if (
                        rank_Y
                        >= dimOfLayer - len(dead_neurons) - len(remove_columns) + 2
                    ):
                        break

                print("The inactive indexes are", inactive_indexes)
                print("Dead neurons are", dead_neurons)
                print("The columns to be removed are", remove_columns)

                # Sign recovery process
                # remove the columns in remove_columns from the matrix_Y
                matrix_Y_without_columns = np.delete(matrix_Y, remove_columns, axis=1)
                rank_Y_without_columns = np.linalg.matrix_rank(matrix_Y_without_columns)
                # rank_Y_without_columns = np.linalg.matrix_rank(
                #     matrix_Y_without_columns, tol=1e-5
                # )
                print(
                    "The shape of matrix_Y_without_columns is",
                    matrix_Y_without_columns.shape,
                )
                print("The rank of matrix_Y_without_columns is", rank_Y_without_columns)

                soln, *rest = np.linalg.lstsq(
                    np.array(
                        matrix_Y_without_columns, dtype=np.float64
                    ),  # float64 #float32
                    np.array(vector_Z, dtype=np.float64),
                    # 1e-5,
                    # tol for small networks
                )  # float64 #float32
                # print("The soln from the SOE (Y, 1e-5) is", soln)
                print("The soln from the SOE (Y) is", soln)
                # print(
                #     "Residual is",
                #     vector_Z - np.matmul(np.array(matrix_Y_without_columns), soln),
                # )

                residual = vector_Z - np.matmul(
                    np.array(matrix_Y_without_columns), soln
                )
                max_res_item = max(abs(residual))
                print("The max_res_item is", max_res_item)

                recovered_signs = []
                soln_index = 0
                active_neurons = []
                for neuron_id in range(dimOfLayer):
                    relative_dead_index = 0
                    while (
                        relative_dead_index < len(dead_neurons)
                        and dead_neurons[relative_dead_index] < neuron_id
                    ):
                        relative_dead_index += 1
                    adjusted_index = neuron_id - relative_dead_index

                    if adjusted_index in remove_columns or neuron_id in dead_neurons:
                        recovered_signs.append(guessed_sign_neuron_wiggle[neuron_id])
                    else:
                        # Get the output sign of this neuron
                        output_sign = original_output_signs[neuron_id]
                        # 1e-5
                        if abs(soln[soln_index]) < threshold:
                            # This neuron is inactive
                            recovered_signs.append(-1 * output_sign)
                        else:
                            # This neuron is active
                            print(
                                "neuron_id",
                                neuron_id,
                                "is active",
                                "soln value is",
                                soln[soln_index],
                            )
                            recovered_signs.append(1 * output_sign)
                            active_neurons.append(neuron_id)
                        soln_index += 1
                print("Active neurons are", active_neurons)
                print("[Extended] The recovered signs are", recovered_signs)

                print(
                    "The last inactive index is",
                    inactive_indexes[len(remove_columns) - 1],
                )
                least_cl_extension = sign_confidence[
                    inactive_indexes[len(remove_columns) - 1]
                ]
                print("[Extension] The rank of the point is", point_rank)
                print("[Extension] The rank of the extended point is", rank_Y)
                print("[Extension] The least confidence level is", least_cl_extension)
                print(
                    "[confidence Level] The point has least confidence level after extension of its inactive neurons as",
                    least_cl_extension,
                )
                print(
                    "[Extension] The total number of inactive neurons is",
                    np.sum(original_output_signs == -1),
                )
                print("[Extension] Required inactive count is", len(remove_columns))
                print("[Extension] The number of dead neurons is", len(dead_neurons))

                if least_cl_extension > largest_cl_extension:
                    largest_cl_extension = least_cl_extension
                    # final_signs = recovered_signs
            else:
                print(
                    "[Extension] SOE requirement is NOT satisfied after neuron wiggle"
                )

            valid_point_counter += 1
            print(
                "[selected_point_counter] Find selected point number",
                valid_point_counter,
            )
            if valid_point_counter >= selected_point_num:
                break

            # If we only want to verify the correctness of the sign recovery, select a proper threshold and stop here
            if valid_point_counter >= 1:
                break
        print("\n=====================================")
        print(
            "The highest least confidence level from original points is",
            largest_cl_original,
        )
        print(
            "The highest least confidence level from extended points is",
            largest_cl_extension,
        )
        print("final_signs are", final_signs)
        # print("The maximum SOE rank among all clusters is", largest_rank)
        # print("The number of dead neurons is", len(dead_neurons))
        # print("The inactive count for that point is", inactive_count)
        # print("Required rank is", dimOfLayer - len(dead_neurons) - largest_rank)

        return
        # """

        ############################
        # extension strategy
        ############################
        # Increase the rank of SOE method
        mergeable_critical_points_list, matrix_Y_list, vector_Z_list = (
            find_mergeable_critical_points(
                args,
                model,
                target_layer,
                previous_weights,
                previous_biases,
                target_weights,
                target_biases,
                dead_neurons,
            )
        )

        # Sort signs based on the confidence levels
        sorted_sign_indices = np.argsort(sign_confidence)[::-1]
        print("The sorted indices based on confidence levels are", sorted_sign_indices)
        for i in sorted_sign_indices:
            print(
                "Neuron index",
                i,
                "has confidence level",
                sign_confidence[i],
                "and guessed sign",
                guessed_sign_neuron_wiggle[i],
            )
        print("********************************")

        largest_rank = 0
        largest_cl_extension = 0
        for cluster_index in range(len(mergeable_critical_points_list)):
            matrix_Y = np.array(matrix_Y_list[cluster_index])
            vector_Z = np.array(vector_Z_list[cluster_index])

            input_point = mergeable_critical_points_list[cluster_index][0]
            output = forward(input_point, previous_weights, previous_biases, True)
            output = np.matmul(output, target_weights) + target_biases
            original_output_signs = np.sign(output)
            recovered_signs = []

            rank_input_point = get_rank_of_point(
                input_point, previous_weights, previous_biases
            )
            print("[Extension] The rank of the input point is", rank_input_point)

            rank_Y = np.linalg.matrix_rank(matrix_Y, tol=1e-5)
            print("The rank_Y for the largest region is (tol=1e-5)", rank_Y)
            rank_Y = np.linalg.matrix_rank(matrix_Y)
            print("The rank_Y for the largest region is", rank_Y)

            print("The target rank is", dimOfLayer - len(dead_neurons))
            # print("The matrix Y is\n", matrix_Y)
            # print("The vector Z is", vector_Z)
            # Get singular values of matrix_Y
            U, S, Vt = np.linalg.svd(matrix_Y)
            print("The singular values of matrix_Y are", S)

            # mnist
            # Layer 5
            # threshold = 1e-6
            # Layer 7
            # threshold = 1e-1
            # Layer 8
            # threshold = 1e-3

            # cifar
            # layer 2
            threshold = 1e-4
            if args.layerID == 7:
                threshold = 1.1e-4
            print("The proper threshold for the first point is", threshold)

            if rank_Y > largest_rank:
                largest_rank = rank_Y

            if rank_Y >= dimOfLayer - len(dead_neurons):
                print(
                    "[Special] This point can satisfy the SOE requirement, use it directly"
                )

                largest_cl_extension = 1

                soln, *rest = np.linalg.lstsq(
                    np.array(matrix_Y, dtype=np.float64),  # float64 #float32
                    np.array(vector_Z, dtype=np.float64),
                    # 1e-5,
                )  # float64 #float32
                # print("The soln from the SOE (Y, 1e-5) is", soln)
                print("The soln from the SOE (Y) is", soln)
                # print(
                #     "Residual is",
                #     vector_Z - np.matmul(np.array(matrix_Y), soln),
                # )

                soln_index = 0
                for neuron_id in range(dimOfLayer):
                    if neuron_id in dead_neurons:
                        recovered_signs.append(guessed_sign_neuron_wiggle[neuron_id])
                    else:
                        # Get the output sign of this neuron
                        output_sign = original_output_signs[neuron_id]
                        # 1e-5
                        if abs(soln[soln_index]) < threshold:
                            # This neuron is inactive
                            recovered_signs.append(-1 * output_sign)
                        else:
                            # This neuron is active
                            recovered_signs.append(1 * output_sign)
                        soln_index += 1

                print("[Direct] The recovered signs are", recovered_signs)

                print("[Direct] The rank of the extended point is", rank_Y)
                print("[Direct] The least confidence level is", 1)
                print(
                    "[Direct] The total number of inactive neurons is",
                    np.sum(original_output_signs == -1),
                )
                print("[Direct] Required inactive count is", 0)
                print("[Direct] The number of dead neurons is", len(dead_neurons))
                continue

            # Combine SOE and neuron wiggle
            # Find which columns can be removed using neuron wiggle
            inactive_indexes = []
            rank_flag = False
            layer_input = forward(input_point, previous_weights, previous_biases, True)
            for neuron_index in sorted_sign_indices:
                if neuron_index in dead_neurons:
                    continue

                output = (
                    np.matmul(layer_input, target_weights[:, neuron_index])
                    + target_biases[neuron_index]
                )
                output_at_neuron = output * guessed_sign_neuron_wiggle[neuron_index]
                if output_at_neuron <= 0:
                    inactive_indexes.append(neuron_index)

                if rank_Y >= dimOfLayer - len(dead_neurons) - len(inactive_indexes) + 2:
                    # +2 for an overdetermined system
                    # No need to continue, already reach the requirement
                    rank_flag = True
                    break

            print("Satisfy the rank requirement?", rank_flag)
            print("The input point uses inactive_indexes", len(inactive_indexes))
            if not rank_flag:
                print(
                    "Cannot satisfy the rank requirement with neuron wiggle, skip this point"
                )
                continue

            least_CL_neuron_id = inactive_indexes[-1]
            print(
                "[confidence Level] The least confidence level of removed neurons is",
                sign_confidence[least_CL_neuron_id],
            )
            if sign_confidence[least_CL_neuron_id] > largest_cl_extension:
                largest_cl_extension = sign_confidence[least_CL_neuron_id]

            remove_columns = []
            error_flag = False
            # compute the new index after removing dead neurons
            for inactive_id in inactive_indexes:
                # get the position of inactive_id in dead_neurons
                if (
                    inactive_id in inactive_indexes
                    and inactive_id in error_neurons
                    and inactive_id not in dead_neurons
                ):
                    error_flag = True
                    print("[error neuron in inactive neurons]", inactive_id)
                    break

                relative_dead_index = 0
                while (
                    relative_dead_index < len(dead_neurons)
                    and dead_neurons[relative_dead_index] < inactive_id
                ):
                    relative_dead_index += 1
                adjusted_index = inactive_id - relative_dead_index
                remove_columns.append(adjusted_index)
            if error_flag:
                continue

            print("The inactive indexes are", inactive_indexes)
            print("Dead neurons are", dead_neurons)
            print("The columns to be removed are", remove_columns)

            if rank_Y >= dimOfLayer - len(dead_neurons) - len(remove_columns) + 2:
                # remove the columns in remove_columns from the matrix_Y
                matrix_Y_without_columns = np.delete(matrix_Y, remove_columns, axis=1)
                soln, *rest = np.linalg.lstsq(
                    np.array(
                        matrix_Y_without_columns, dtype=np.float64
                    ),  # float64 #float32
                    np.array(vector_Z, dtype=np.float64),
                    # 1e-7,
                    # layer 8
                )  # float64 #float32
                # print("The soln from the SOE (Y, 1e-7) is", soln)
                print("The soln from the SOE (Y) is", soln)
                # print(
                #     "Residual is",
                #     vector_Z - np.matmul(np.array(matrix_Y_without_columns), soln),
                # )

                soln_index = 0
                active_neurons = []
                for neuron_id in range(dimOfLayer):
                    if neuron_id in inactive_indexes or neuron_id in dead_neurons:
                        recovered_signs.append(guessed_sign_neuron_wiggle[neuron_id])
                    else:
                        # Get the output sign of this neuron
                        output_sign = original_output_signs[neuron_id]
                        # 1e-5
                        if abs(soln[soln_index]) < threshold:
                            # This neuron is inactive
                            recovered_signs.append(-1 * output_sign)
                        else:
                            # This neuron is active
                            print(
                                "neuron_id",
                                neuron_id,
                                "is active",
                                "soln value is",
                                soln[soln_index],
                            )
                            recovered_signs.append(1 * output_sign)
                            active_neurons.append(neuron_id)
                        soln_index += 1
                print("Active neurons are", active_neurons)
                rank_input_point = get_rank_of_point(
                    input_point, previous_weights, previous_biases
                )
                print("[Extension] The rank of the input point is", rank_input_point)
                print("[Extension] The recovered signs are", recovered_signs)
                print("[Extension] The rank of the extended point is", rank_Y)
                print(
                    "[Extension] The least confidence level is",
                    sign_confidence[least_CL_neuron_id],
                )
                print(
                    "[Extension] The total number of inactive neurons is",
                    np.sum(original_output_signs == -1),
                )
                print("[Extension] Required inactive count is", len(remove_columns))
                print("[Extension] The number of dead neurons is", len(dead_neurons))
                # print("The real signs are", random_signs)
                break
            else:
                print("The neuron wiggle can not decrease the rank to SOE requirement")
                continue

        print("\n=====================================")
        print(
            "The highest least confidence level from extended points is",
            largest_cl_extension,
        )
        print("The highest SOE rank from all clusters is", largest_rank)

        # print("The maximum SOE rank among all clusters is", largest_rank)
        # print("The number of dead neurons is", len(dead_neurons))
        # print("The inactive count for that point is", inactive_count)
        # print("Required rank is", dimOfLayer - len(dead_neurons) - largest_rank)

        return

        ############################
        # The simplest combination of SOE and neuron wiggle
        ############################

        # Increase the rank of SOE method
        # mergeable_critical_points, matrix_Y, vector_Z = find_mergeable_critical_points(
        #     args,
        #     model,
        #     target_layer,
        #     previous_weights,
        #     previous_biases,
        #     target_weights,
        #     target_biases,
        #     dead_neurons,
        # )
        mergeable_critical_points_list, matrix_Y_list, vector_Z_list = (
            find_mergeable_critical_points(
                args,
                model,
                target_layer,
                previous_weights,
                previous_biases,
                target_weights,
                target_biases,
                dead_neurons,
            )
        )
        mergeable_critical_points = mergeable_critical_points_list[0]
        matrix_Y = matrix_Y_list[0]
        vector_Z = vector_Z_list[0]

        matrix_Y = np.array(matrix_Y)

        input_point = mergeable_critical_points[0]
        output = forward(input_point, previous_weights, previous_biases, True)
        output = np.matmul(output, target_weights) + target_biases
        original_output_signs = np.sign(output)

        rank_Y = np.linalg.matrix_rank(matrix_Y, tol=1e-5)
        print("The rank_Y for the largest region is", rank_Y)
        print("The target rank is", dimOfLayer - len(dead_neurons))
        print("The matrix Y is\n", matrix_Y)
        print("The vector Z is", vector_Z)
        # Get singular values of matrix_Y
        U, S, Vt = np.linalg.svd(matrix_Y)
        print("The singular values of matrix_Y are", S)

        # Find remoable columns using neuron wiggle method
        if rank_Y < dimOfLayer - len(dead_neurons):
            remove_columns, guessed_sign_neuron_wiggle = (
                neuron_wiggle_to_find_removable_columns(
                    args,
                    model,
                    mergeable_critical_points,
                    rank_Y,
                    target_weights.shape[1] - len(dead_neurons),
                    target_weights,
                    target_biases,
                    target_layer,
                    previous_weights,
                    previous_biases,
                    dead_neurons,
                )
            )
        else:
            guessed_sign_neuron_wiggle = [0] * target_weights.shape[1]
            remove_columns = []

        # remove the columns in remove_columns from the matrix_Y
        matrix_Y_without_columns = np.delete(matrix_Y, remove_columns, axis=1)
        soln, *rest = np.linalg.lstsq(
            np.array(matrix_Y_without_columns, dtype=np.float64),  # float64 #float32
            np.array(vector_Z, dtype=np.float64),
            1e-6,
        )  # float64 #float32
        print("The soln from the SOE (Y, 1e-6) is", soln)
        print(
            "Residual is",
            vector_Z - np.matmul(np.array(matrix_Y_without_columns), soln),
        )

        recovered_signs = []
        if len(remove_columns) == 0 and guessed_sign_neuron_wiggle[0] != 0:
            print("The neuron wiggle can not decrease the rank to SOE requirement")
            print("We use the result recovered by neuron wiggle solely")
            recovered_signs = guessed_sign_neuron_wiggle
        else:
            # remove the columns in remove_columns from the matrix_Y
            matrix_Y_without_columns = np.delete(matrix_Y, remove_columns, axis=1)
            soln, *rest = np.linalg.lstsq(
                np.array(
                    matrix_Y_without_columns, dtype=np.float64
                ),  # float64 #float32
                np.array(vector_Z, dtype=np.float64),
                1e-5,
            )  # float64 #float32
            print("The soln from the SOE (Y, 1e-5) is", soln)
            print(
                "Residual is",
                vector_Z - np.matmul(np.array(matrix_Y_without_columns), soln),
            )

            soln_index = 0
            for neuron_id in range(dimOfLayer):
                if neuron_id in remove_columns or neuron_id in dead_neurons:
                    recovered_signs.append(guessed_sign_neuron_wiggle[neuron_id])
                else:
                    # Get the output sign of this neuron
                    output_sign = original_output_signs[neuron_id]
                    # 1e-5
                    if abs(soln[soln_index]) < 1e-3:
                        # This neuron is inactive
                        recovered_signs.append(-1 * output_sign)
                    else:
                        # This neuron is active
                        recovered_signs.append(1 * output_sign)
                    soln_index += 1
        print("The final signs are", recovered_signs)
        breakpoint()

    # We start to recover the signatures
    if args.real_attack == 0:
        # First setup the critical points generator
        critical_points_yielder = sweep_for_critical_points(
            args, model, PARAM_SEARCH_AT_LOCATION, dataset
        )

        # Extract weights corresponding to those critical points
        # We don't want to waste any queries for critical points
        # So we can reuse the critical points that we have already filtered out in the previous layers
        remaining_critical_points = []
        COUNT = args.Count
        print("\nStart to search for", COUNT, " critical points")

        t_cp_searching_start = time.time()
        while len(remaining_critical_points) < COUNT:
            point = next(critical_points_yielder)

            if any(np.any(np.abs(x) < 1e-5) for x in get_hidden_layers(point, A, B)):
                if args.debug:
                    print(
                        "This critical point is caused by a recovered neuron before the target layer"
                    )
                continue

            for cp in remaining_critical_points:
                if np.sum((cp - point) ** 2) ** 0.5 < 1e-4:
                    continue

            remaining_critical_points.append(point)
        t_cp_searching_stop = time.time()
        t_cp_searching = t_cp_searching_stop - t_cp_searching_start
        print("Time for searching critical points: ", t_cp_searching)
        print("After filtering out recovered critical points")
        print("We have", len(remaining_critical_points), "remaining points")

        (
            extracted_normal,
            extracted_bias,
            extracted_sign,
            extracted_sequence,
            still_remaining_critical_points,
            T,
            T_follow_hyperplane,
            clustered_components,
        ) = compute_layer_values(
            args,
            remaining_critical_points,
            critical_points_yielder,
            model,
            weights,
            biases,
            layerId,
            dimOfInput,
            dimOfPrevLayer,
            dimOfLayer,
            special,
            savePath,
        )

        remaining_critical_points = still_remaining_critical_points
        dead_neurons = []
        for i in range(len(extracted_bias)):
            if extracted_bias[i] == 0:
                dead_neurons.append(i)
        print("The dead neurons are", dead_neurons)
        print("[Step 1] Finishes. We have recovered neurons' signatures")
        print("The extracted sequence is", extracted_sequence)

    if args.layerID != 1 and args.real_attack == 1:
        # First setup the critical points generator
        critical_points_yielder = sweep_for_critical_points(
            args, model, PARAM_SEARCH_AT_LOCATION, dataset
        )

        # Extract weights corresponding to those critical points
        # We don't want to waste any queries for critical points
        # So we can reuse the critical points that we have already filtered out in the previous layers
        remaining_critical_points = []
        COUNT = args.Count
        print("\nStart to search for", COUNT, " critical points")

        t_cp_searching_start = time.time()
        weights_all, biases_all = getAllWeightsAndBiases(model)
        while len(remaining_critical_points) < COUNT:
            point = next(critical_points_yielder)
            if any(
                np.any(np.abs(x) < 1e-5)
                for x in get_hidden_layers(
                    point,
                    weights_all[: args.layerID - 1],
                    biases_all[: args.layerID - 1],
                )
            ):
                if args.debug:
                    print(
                        "This critical point is caused by a recovered neuron before the target layer"
                    )
                continue

            if check_activation(args, point, A, B, dead_neurons) == False:
                print("This critical point is not valid")
                continue

            for cp in remaining_critical_points:
                if np.sum((cp - point) ** 2) ** 0.5 < 1e-4:
                    continue

            remaining_critical_points.append(point)
        t_cp_searching_stop = time.time()
        t_cp_searching = t_cp_searching_stop - t_cp_searching_start
        print("Time for searching critical points: ", t_cp_searching)
        print("After filtering out recovered critical points")
        print("We have", len(remaining_critical_points), "remaining points")

        (
            extracted_normal,
            extracted_bias,
            extracted_sign,
            extracted_sequence,
            still_remaining_critical_points,
            T,
            T_follow_hyperplane,
            clustered_components,
        ) = compute_layer_values(
            args,
            remaining_critical_points,
            critical_points_yielder,
            model,
            weights,
            biases,
            layerId,
            dimOfInput,
            dimOfPrevLayer,
            dimOfLayer,
            special,
            savePath,
        )

        remaining_critical_points = still_remaining_critical_points
        dead_neurons = []
        for i in range(len(extracted_bias)):
            if extracted_bias[i] == 0:
                dead_neurons.append(i)
        print("The dead neurons are", dead_neurons)
        print("[Step 1] Finishes. We have recovered neurons' signatures")
        print("The extracted sequence is", extracted_sequence)

        query_count_sig, crit_query_count_sig = get_query_counts()
        print("Critical query count for signature recovery: ", crit_query_count_sig)
        print("Query count for signature recovery: ", query_count_sig)

        # Find more critical points
        T_precision_improve_start = time.time()
        target_length = dimOfLayer * 5
        min_length = min(len(component) for component in clustered_components)
        print(
            "At start, the length of components is",
            [len(component) for component in clustered_components],
        )
        print("The target length is", target_length)
        print("At start, the minimum length of components is", min_length)

        if args.dataset == "mnist":
            dimOfLayer = np.array(weights[0]).shape[1]
            if dimOfLayer == 8:
                flat_threshold = 1e-4
            if dimOfLayer == 16:
                flat_threshold = 1e-5
            flat_num = 2
        if args.dataset == "cifar10":
            flat_threshold = 1e-5
            flat_num = 2

        # Just for comparison, set a very small flat_threshold
        # flat_threshold = 1e-8

        if args.layerID <= 2:
            point_threshold = 1e-4
        else:
            point_threshold = 1e-5

        point_num = 0
        while min_length < target_length and dimOfLayer == 16 and point_num < 2e5:
            point = next(critical_points_yielder)
            point_num += 1
            output = forward(point, A, B, True)
            output = np.matmul(output, extracted_normal) + extracted_bias

            for element_index in range(len(output)):
                if element_index in dead_neurons:
                    # This neuron is dead, skip it
                    continue
                if np.abs(output[element_index]) < point_threshold:
                    if len(clustered_components[element_index]) > target_length:
                        # This component already has enough critical points, skip it
                        continue

                    filtered = False
                    bad_quality_points_count = 0
                    # Check the quality of the new critical point
                    for i in range(50):
                        d = np.sign(np.random.normal(0, 1, point.shape))
                        eps = 1e-4
                        # See what the second partial derivitive at this value is
                        ratio_val = get_second_grad_unsigned(
                            args, point, d, weights_all, biases_all, eps, eps / 3
                        )
                        if abs(ratio_val) < flat_threshold:
                            print(
                                "The bad second grad is",
                                abs(ratio_val),
                                "at direction",
                                i,
                            )
                            # If the ratio value is too small, we can skip this point
                            # and not add it to the ys list.
                            bad_quality_points_count += 1
                            if bad_quality_points_count >= flat_num:
                                print(
                                    "Too many bad quality directions, stopping the search for this critical point"
                                )
                                filtered = True
                                break
                            # filtered = True
                            # break
                    if filtered:
                        continue

                    clustered_components[element_index].append(point)
                    print("Add a new critical point to the components")
                    print("The output is", np.abs(output[element_index]))
                    layer_id, neuron_id = on_which_hidden_layer(point, model)
                    print(
                        "The critical point is on layer",
                        layer_id,
                        "and neuron",
                        neuron_id,
                    )

                    min_length = min(
                        len(component) for component in clustered_components
                    )
                    print(
                        "Now, the length of components is",
                        [len(component) for component in clustered_components],
                    )
                    # print("Now, the minimum length of components is", min_length)

                    continue

        # Improve the precision of the extracted normal
        print("\n[Step 2] Starts. Improving precision of extracted normal")

        # """
        if args.real_attack == 1 and args.layerID > 1:
            weight_improved, bias_improved = improve_precision(
                args,
                A,
                B,
                extracted_normal,
                extracted_bias,
                extracted_sequence,
                model,
                dimOfInput,
                dimOfPrevLayer,
                dimOfLayer,
                clustered_components,
            )
            extracted_normal, extracted_bias = weight_improved, bias_improved
        # """

        T_precision_improve_stop = time.time()
        T_precision_improve = T_precision_improve_stop - T_precision_improve_start
        print(
            "Time for improving precision of extracted normal: ",
            T_precision_improve,
        )

        query_count_sig_imp, crit_query_count_sig_imp = get_query_counts()
        print(
            "Critical query count for signature improvement: ",
            crit_query_count_sig_imp - crit_query_count_sig,
        )
        print(
            "Query count for signature recovery improvement: ",
            query_count_sig_imp - query_count_sig,
        )
        print(
            "All critical query count until signature recovery: ",
            crit_query_count_sig_imp,
        )
        print("All query count until signature recovery: ", query_count_sig_imp)

        # Recover signs
        T_sign_recovery_start = time.time()
        if args.cheat_sign == 0:
            print("\n[Step 2] Starts. Recovering signs")
            # Increase the rank of SOE method
            mergeable_critical_points, matrix_Y, vector_Z = (
                find_mergeable_critical_points(
                    args,
                    model,
                    args.layerID - 1,
                    A,
                    B,
                    extracted_normal,
                    extracted_bias,
                    dead_neurons,
                )
            )

            if len(mergeable_critical_points) == 0:
                print("No mergeable critical points found")
                rank_Y = 0
            else:
                matrix_Y = np.array(matrix_Y)

                input_point = mergeable_critical_points[0]
                output = forward(input_point, A, B, True)
                output = np.matmul(output, extracted_normal) + extracted_bias
                original_output_signs = np.sign(output)

                rank_Y = np.linalg.matrix_rank(matrix_Y, tol=1e-5)
                print("The rank_Y for the largest region is", rank_Y)
                print("The target rank is", dimOfLayer - len(dead_neurons))
                print("The matrix Y is\n", matrix_Y)
                print("The vector Z is", vector_Z)
                # Get singular values of matrix_Y
                U, S, Vt = np.linalg.svd(matrix_Y)
                print("The singular values of matrix_Y are", S)

            if rank_Y < dimOfLayer - len(dead_neurons):
                # Find removable columns using neuron wiggle method
                remove_columns, guessed_sign_neuron_wiggle = (
                    neuron_wiggle_to_find_removable_columns(
                        args,
                        model,
                        mergeable_critical_points,
                        rank_Y,
                        dimOfLayer - len(dead_neurons),
                        extracted_normal,
                        extracted_bias,
                        args.layerID - 1,
                        A,
                        B,
                        dead_neurons,
                    )
                )
            else:
                guessed_sign_neuron_wiggle = [0] * extracted_normal.shape[1]
                remove_columns = []

            print("The columns to remove are", remove_columns)
            print(
                "The guessed signs from neuron_wiggle are", guessed_sign_neuron_wiggle
            )

            recovered_signs = []
            if len(remove_columns) == 0 and guessed_sign_neuron_wiggle[0] != 0:
                print("The neuron wiggle can not decrease the rank to SOE requirement")
                print("We use the result recovered by neuron wiggle solely")
                recovered_signs = guessed_sign_neuron_wiggle
            else:
                # remove the columns in remove_columns from the matrix_Y
                matrix_Y_without_columns = np.delete(matrix_Y, remove_columns, axis=1)
                soln, *rest = np.linalg.lstsq(
                    np.array(
                        matrix_Y_without_columns, dtype=np.float64
                    ),  # float64 #float32
                    np.array(vector_Z, dtype=np.float64),
                    1e-5,
                )  # float64 #float32
                print("The soln from the SOE (Y, 1e-5) is", soln)
                print(
                    "Residual is",
                    vector_Z - np.matmul(np.array(matrix_Y_without_columns), soln),
                )

                soln_index = 0
                for neuron_id in range(dimOfLayer):
                    if neuron_id in remove_columns or neuron_id in dead_neurons:
                        recovered_signs.append(guessed_sign_neuron_wiggle[neuron_id])
                    else:
                        # Get the output sign of this neuron
                        output_sign = original_output_signs[neuron_id]
                        # 1e-5
                        if abs(soln[soln_index]) < 1e-3:
                            # This neuron is inactive
                            recovered_signs.append(-1 * output_sign)
                        else:
                            # This neuron is active
                            recovered_signs.append(1 * output_sign)
                        soln_index += 1
            print("The final signs are", recovered_signs)
            extracted_normal = extracted_normal * recovered_signs
            extracted_bias = extracted_bias * recovered_signs
            extracted_sign = np.array(recovered_signs, dtype=np.int_)
        T_sign_recovery_stop = time.time()
        T_sign_recovery = T_sign_recovery_stop - T_sign_recovery_start
        print("Time for sign recovery: ", T_sign_recovery)

        query_count_sign, crit_query_count_sign = get_query_counts()
        print(
            "Critical query count for sign recovery: ",
            crit_query_count_sign - query_count_sig_imp,
        )
        print("Query count for sign recovery: ", query_count_sign - query_count_sig_imp)
        print("All critical query count until sign recovery: ", crit_query_count_sign)
        print("All query count until sign recovery: ", query_count_sign)

    # Optimize the precision of the first layer with a large size
    if args.layerID == 1 and args.real_attack == 1:
        # Step 1: Use a small number of critical points to recover approximate signatures
        print("\n[Step 1]Recover approximate signatures for the first layer")
        weights_all, biases_all = getAllWeightsAndBiases(model)

        # First setup the critical points generator
        critical_points_yielder = sweep_for_critical_points(
            args, model, PARAM_SEARCH_AT_LOCATION, dataset
        )

        # Extract weights corresponding to those critical points
        # We don't want to waste any queries for critical points
        # So we reuse the critical points that we have already filtered out in the previous layers
        remaining_critical_points = []
        COUNT = args.Count
        print("\nStart to search for", COUNT, " critical points")

        t_cp_searching_start = time.time()
        while len(remaining_critical_points) < COUNT:
            point = next(critical_points_yielder)

            for cp in remaining_critical_points:
                if np.sum((cp - point) ** 2) ** 0.5 < 1e-4:
                    continue

            remaining_critical_points.append(point)
        t_cp_searching_stop = time.time()
        t_cp_searching = t_cp_searching_stop - t_cp_searching_start
        print("Time for searching critical points: ", t_cp_searching)
        print("After filtering out recovered critical points")
        print("We have", len(remaining_critical_points), "remaining points")

        (
            extracted_normal,
            extracted_bias,
            extracted_sign,
            extracted_sequence,
            still_remaining_critical_points,
            T,
            T_follow_hyperplane,
            clustered_components,
        ) = compute_layer_values(
            args,
            remaining_critical_points,
            critical_points_yielder,
            model,
            weights,
            biases,
            layerId,
            dimOfInput,
            dimOfPrevLayer,
            dimOfLayer,
            special,
            savePath,
        )

        # Stupid check for clustered_components
        for index, component in enumerate(clustered_components):
            print("Component", index, "contains", len(component), "points")
            for point_index in range(len(component)):
                print("Index", point_index)
                output = (
                    np.matmul(component[point_index], extracted_normal) + extracted_bias
                )
                print("Output of the critical point is", output[index])

        query_count_sig, crit_query_count_sig = get_query_counts()
        print("Critical query count for signature recovery: ", crit_query_count_sig)
        print("Query count for signature recovery: ", query_count_sig)

        # Find more critical points
        if args.dataset == "mnist":
            target_length = dimOfInput * 2
        if args.dataset == "cifar10":
            target_length = int(dimOfInput * 1.5)
        min_length = min(len(component) for component in clustered_components)
        print(
            "At start, the length of components is",
            [len(component) for component in clustered_components],
        )
        print("At start, the minimum length of components is", min_length)

        T_precision_improve_start = time.time()
        if args.dataset == "mnist":
            dimOfLayer = np.array(weights[0]).shape[1]
            if dimOfLayer == 8:
                flat_threshold = 1e-4
            if dimOfLayer == 16:
                flat_threshold = 1e-5
            flat_num = 2
        if args.dataset == "cifar10":
            flat_threshold = 1e-5
            flat_num = 2

        # Just for comparison, set a very small flat_threshold
        # flat_threshold = 1e-8

        # """
        while min_length < target_length:
            point = next(critical_points_yielder)
            output = np.matmul(point, extracted_normal) + extracted_bias

            for element_index in range(len(output)):
                if np.abs(output[element_index]) < 1e-5:
                    if len(clustered_components[element_index]) > target_length:
                        # This component already has enough critical points, skip it
                        continue

                    filtered = False
                    bad_quality_points_count = 0
                    # Check the quality of the new critical point
                    for i in range(50):
                        d = np.sign(np.random.normal(0, 1, point.shape))
                        eps = 1e-4
                        # See what the second partial derivitive at this value is
                        ratio_val = get_second_grad_unsigned(
                            args, point, d, weights_all, biases_all, eps, eps / 3
                        )
                        if abs(ratio_val) < flat_threshold:
                            print(
                                "The bad second grad is",
                                abs(ratio_val),
                                "at direction",
                                i,
                            )
                            # If the ratio value is too small, we can skip this point
                            # and not add it to the ys list.
                            bad_quality_points_count += 1
                            if bad_quality_points_count >= flat_num:
                                print(
                                    "Too many bad quality directions, stopping the search for this critical point"
                                )
                                filtered = True
                                break
                            # filtered = True
                            # break
                    if filtered:
                        continue

                    clustered_components[element_index].append(point)
                    print("Add a new critical point to the components")
                    layer_id, neuron_id = on_which_hidden_layer(point, model)
                    print(
                        "The critical point is on layer",
                        layer_id,
                        "and neuron",
                        neuron_id,
                    )

                    min_length = min(
                        len(component) for component in clustered_components
                    )
                    print(
                        "Now, the length of components is",
                        [len(component) for component in clustered_components],
                    )
                    # print("Now, the minimum length of components is", min_length)

                    continue

        #  Improve the precision of the first layer
        weight_improved, bias_improved = improve_precision(
            args,
            A,
            B,
            extracted_normal,
            extracted_bias,
            extracted_sequence,
            model,
            dimOfInput,
            dimOfPrevLayer,
            dimOfLayer,
            clustered_components,
        )
        extracted_normal, extracted_bias = weight_improved, bias_improved
        # """

        T_precision_improve_stop = time.time()
        T_precision_improve = T_precision_improve_stop - T_precision_improve_start
        print(
            "Time for improving precision of extracted normal: ",
            T_precision_improve,
        )

        query_count_sig_imp, crit_query_count_sig_imp = get_query_counts()
        print(
            "Critical query count for signature improvement: ",
            crit_query_count_sig_imp - crit_query_count_sig,
        )
        print(
            "Query count for signature recovery improvement: ",
            query_count_sig_imp - query_count_sig,
        )
        print(
            "All critical query count until signature recovery: ",
            crit_query_count_sig_imp,
        )
        print("All query count until signature recovery: ", query_count_sig_imp)

        T_sign_recovery_start = time.time()
        if args.cheat_sign == 0:
            # We need to recover the sign for the first layer
            eps = 1e-3
            print("The dimension of the first layer is", dimOfInput)

            # randomly select a point for the first layer
            matrix_Y = []
            vector_Z = []
            continuous_input_point = np.random.uniform(-1e1, 1e1, dimOfInput)
            for i in range(dimOfInput):
                print("Handling the", i, "th direction in the first layer")
                random_direction = np.sign(
                    np.random.normal(0, 1, continuous_input_point.shape)
                )

                point = continuous_input_point
                output_point = np.matmul(point, extracted_normal) + extracted_bias
                print("output of the previous layers", output_point)
                # output_point = np.abs(output_point)
                # print("output_point after abs", output_point)
                final_output_point = predict_manual_fast(
                    continuous_input_point, weights_all, biases_all
                )

                point_eps = continuous_input_point + eps * random_direction
                output_point_eps = (
                    np.matmul(point_eps, extracted_normal) + extracted_bias
                )
                print("output of the previous layers for eps", output_point_eps)
                # output_point_eps = np.abs(output_point_eps)
                # print("output_point_eps after abs", output_point_eps)
                final_output_point_eps = predict_manual_fast(
                    point_eps, weights_all, biases_all
                )

                output_delta = output_point_eps - output_point
                final_output_delta = final_output_point_eps - final_output_point
                matrix_Y.append(output_delta)
                vector_Z.append(final_output_delta)

            matrix_Y = np.array(matrix_Y)
            vector_Z = np.array(vector_Z)
            if args.dataset == "cifar10":
                # A query returns shape (1, outputs), so stacking directions
                # adds a singleton batch axis. Select the output coordinate,
                # not that batch axis, before solving for one coefficient per
                # recovered neuron.
                vector_Z = vector_Z.reshape(len(vector_Z), -1)[:, args.output_index]
            print("The vector_Z is", vector_Z)
            U, S, Vt = np.linalg.svd(matrix_Y)
            print("The shape of matrix_Y is", matrix_Y.shape)
            print("The singular values of matrix_Y are", S)
            print(
                "The rank of the matrix Y is", np.linalg.matrix_rank(matrix_Y, tol=1e-6)
            )

            # Solve the system of equations
            soln, *rest = np.linalg.lstsq(
                np.array(matrix_Y, dtype=np.float64),  # float64 #float32
                np.array(vector_Z, dtype=np.float64),
                1e-6,
            )  # float64 #float32
            print("The soln from the SOE is", soln)

            recovered_signs = []
            output = (
                np.matmul(continuous_input_point, extracted_normal) + extracted_bias
            )
            original_output_signs = np.sign(output)

            for neuron_id in range(dimOfLayer):
                # Get the output sign of this neuron
                output_sign = original_output_signs[neuron_id]
                if abs(soln[neuron_id]) < 1e-5:
                    # This neuron is inactive
                    recovered_signs.append(-1 * output_sign)
                else:
                    # This neuron is active
                    recovered_signs.append(1 * output_sign)
            print("The final signs are", recovered_signs)
            extracted_normal = extracted_normal * recovered_signs
            extracted_bias = extracted_bias * recovered_signs
            extracted_sign = np.array(recovered_signs, dtype=np.int_)
        T_sign_recovery_stop = time.time()
        T_sign_recovery = T_sign_recovery_stop - T_sign_recovery_start
        print("Time for sign recovery: ", T_sign_recovery)

        query_count_sign, crit_query_count_sign = get_query_counts()
        print(
            "Critical query count for sign recovery: ",
            crit_query_count_sign - crit_query_count_sig_imp,
        )
        print("Query count for sign recovery: ", query_count_sign - query_count_sig_imp)
        print("All critical query count until sign recovery: ", crit_query_count_sign)
        print("All query count until sign recovery: ", query_count_sign)

        # remaining_critical_points = still_remaining_critical_points

    # """
    print("-----------------------------------------------")
    print("Analysis of the results")
    print("-----------------------------------------------")

    print("After recovering the signature of layer", layerId)
    print("We have", len(remaining_critical_points), "remaining points")

    if args.layerID == 1:
        print("Time to merge good critical points:", T[0])
        print("For layer 1, time for filtering critical points is", T[1])
        print("Time for filtering critical points and computing ratios: ", T[2])
        print(
            "Time for identifying good CPs with full h rank and computing kernel space of bad CPs: ",
            T[3],
        )
    if args.layerID > 1:
        print("Time to merge good critical points:", T[0])
        print(
            "Time to merge bad critical points into components:",
            T[1],
        )
        print(
            "Time to merge bad critical points with each other:",
            T[2],
        )
        print(
            "Time to merge good and bad critical points with components:",
            T[3],
        )
        print("Time to merge components:", T[4])
        print("Time to filter noise components:", T[5])
        print("Time to merge components again:", T[6])
        print("Time for filtering critical points and computing ratios: ", T[7])
        print(
            "Time for identifying good CPs with full h rank and computing kernel space of bad CPs: ",
            T[8],
        )

    time_signature_recovery = sum(T)
    if T_follow_hyperplane is not None:
        time_follow_hyperplane = sum(T_follow_hyperplane)
        time_signature_recovery += time_follow_hyperplane
        print("Time to follow hyperplane:", T_follow_hyperplane)
        print("Total time to follow hyperplane:", time_follow_hyperplane)
    print("[*]Total time for signature recovery:", time_signature_recovery)
    h, m, sec = seconds_to_h_m_s(time_signature_recovery)
    print(f"[*]Which is {h} hours, {m} minutes, {sec} seconds")

    # if T_precision_improve is not None:
    if args.real_attack == 1:
        print(
            "[*]Time for improving precision of extracted normal: ", T_precision_improve
        )
        h, m, sec = seconds_to_h_m_s(T_precision_improve)
        print(f"[*]Which is {h} hours, {m} minutes, {sec} seconds")
        print(
            "[*]Total time including precision improvement: ",
            time_signature_recovery + T_precision_improve,
        )
        h, m, sec = seconds_to_h_m_s(time_signature_recovery + T_precision_improve)
        print(f"[*]Which is {h} hours, {m} minutes, {sec} seconds")
    # if T_sign_recovery is not None:
    if args.real_attack == 1 and args.cheat_sign == 0:
        print("[*]Time for sign recovery: ", T_sign_recovery)
        h, m, sec = seconds_to_h_m_s(T_sign_recovery)
        print(f"[*]Which is {h} hours, {m} minutes, {sec} seconds")
        print(
            "[*]Total time including sign recovery: ",
            time_signature_recovery + T_precision_improve + T_sign_recovery,
        )
        h, m, sec = seconds_to_h_m_s(
            time_signature_recovery + T_precision_improve + T_sign_recovery
        )
        print(f"[*]Which is {h} hours, {m} minutes, {sec} seconds")

    query_count, crit_query_count = get_query_counts()
    print("[*]Critical query count: ", crit_query_count)
    print("[*]Query count: ", query_count)

    if args.real_attack == 1:
        print("[*]Critical query count for signature recovery: ", crit_query_count_sig)
        pow = approximate_power_of_two(crit_query_count_sig)
        print(f"[*]Which is approximately 2^{pow}")
        print("[*]Query count for signature recovery: ", query_count_sig)
        pow = approximate_power_of_two(query_count_sig)
        print(f"[*]Which is approximately 2^{pow}")

        print(
            "[*]Critical query count for signature improvement: ",
            crit_query_count_sig_imp - crit_query_count_sig,
        )
        pow = approximate_power_of_two(crit_query_count_sig_imp - crit_query_count_sig)
        print(f"[*]Which is approximately 2^{pow}")
        print(
            "[*]Query count for signature recovery improvement: ",
            query_count_sig_imp - query_count_sig,
        )
        pow = approximate_power_of_two(query_count_sig_imp - query_count_sig)
        print(f"[*]Which is approximately 2^{pow}")
        print(
            "All critical query count until signature recovery: ",
            crit_query_count_sig_imp,
        )
        pow = approximate_power_of_two(crit_query_count_sig_imp)
        print(f"[*]Which is approximately 2^{pow}")
        print("All query count until signature recovery: ", query_count_sig_imp)
        pow = approximate_power_of_two(query_count_sig_imp)
        print(f"[*]Which is approximately 2^{pow}")
        print(
            "[*]Critical query count for sign recovery: ",
            crit_query_count_sign - crit_query_count_sig_imp,
        )
        pow = approximate_power_of_two(crit_query_count_sign - crit_query_count_sig_imp)
        print(f"[*]Which is approximately 2^{pow}")
        print(
            "[*]Query count for sign recovery: ", query_count_sign - query_count_sig_imp
        )
        pow = approximate_power_of_two(query_count_sign - query_count_sig_imp)
        print(f"[*]Which is approximately 2^{pow}")
        print(
            "[*]All critical query count until sign recovery: ", crit_query_count_sign
        )
        pow = approximate_power_of_two(crit_query_count_sign)
        print(f"[*]Which is approximately 2^{pow}")
        print("[*]All query count until sign recovery: ", query_count_sign)
        pow = approximate_power_of_two(query_count_sign)
        print(f"[*]Which is approximately 2^{pow}")

    """
    with open(recovery_file_path, "a") as recovery_file:
        recovery_file.write("---------------------[Step 6]---------------------\n")
        recovery_file.write("Data Analysis\n")
        recovery_file.write(
            f"We have {len(remaining_critical_points)} points not merged\n"
        )
        recovery_file.write(
            f"Time for filtering critical points and computing ratios: {T[7]}\n"
        )
        recovery_file.write(
            f"Time for identifying good CPs with full h rank and computing kernel space of bad CPs: {T[8]}\n"
        )
        recovery_file.write(f"Time to merge good critical points: {T[0]}\n")
        recovery_file.write(
            f"Time to merge bad critical points into components: {T[1]}\n"
        )
        recovery_file.write(
            f"Time to merge bad critical points with each other: {T[2]}\n"
        )
        recovery_file.write(
            f"Time to merge good and bad critical points with components: {T[3]}\n"
        )
        recovery_file.write(f"Time to merge components: {T[4]}\n")
        recovery_file.write(f"Time to filter noise components: {T[5]}\n")
        recovery_file.write(f"Time to merge components again: {T[6]}\n")
        if T_follow_hyperplane is not None:
            recovery_file.write(f"Time to follow hyperplane: {T_follow_hyperplane}\n")

        recovery_file.write(f"Critical query count: {crit_query_count}\n")
        recovery_file.write(f"Query count: {query_count}\n")
    """

    return (
        extracted_normal,
        extracted_bias,
        extracted_sign,
        extracted_sequence,
    )


# Main function of Signature Recovery that merges every part together.
# It outputs the full signature for all neurons in a layer in the end if it completes succesfully.
def compute_layer_values(
    args,
    original_critical_points,
    critical_points_yielder,
    model,
    weights,
    biases,
    layerId,
    dimInput,
    dimOfPrevLayer,
    dimOfLayer,
    special,
    savePath,
):
    """
    Main function of Signature Recovery that merges every part together.
    It outputs the full signature for all neurons in a layer in the end if it completes succesfully.
    """
    print("--------------------------------")
    print("Start to recover the signature of layer", layerId)
    print("--------------------------------")

    # Create and clear the result file
    if args.dataset == "cifar10":
        recovery_file_path = os.path.join(
            savePath,
            f"output_index_{args.output_index}",
            f"recovery_results_layer_{args.layerID}.txt",
        )
    else:
        recovery_file_path = os.path.join(
            savePath, f"recovery_results_layer_{args.layerID}.txt"
        )
    with open(recovery_file_path, "w") as recovery_file:
        recovery_file.write("")

    # A and B contain the real weights and biases of layers [0, LAYER-1]
    # We just assume we have recovered the previous layers
    # Note: if we want to implement the end-to-end recovery, we need to use the extracted weights and biases
    if args.real_attack == 0:
        A, B = weights[:-1], biases[:-1]
    else:
        A, B = [], []
        for i in range(1, args.layerID):
            # read the extracted weights and biases
            print("Get recovered weights and biases for layer", i)
            if i == 1:
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
                os.path.join(recoveryPath, f"recovered_{i}.npz")
            )
            A.append(recovered_parameters["extracted_normal"])
            B.append(recovered_parameters["extracted_bias"])

    # Get the index of the target layer
    LAYER = layerId - 1
    print("\nThe target layer is", layerId, "with the index", LAYER)

    COUNT = args.Count
    print("\nThe number of required critical points is", COUNT)
    print(
        "Now we have",
        len(original_critical_points),
        "critical points from the original list",
    )
    COUNT = min(COUNT, len(original_critical_points))

    # The final results are stored in clustered_components and recovered_weights
    recovered_weights = []
    clustered_components = []

    # Check the quality of the original critical points
    print("\n[Checkpoint]Checking quality of all original critical points")
    good_count, bad_count, neurons = check_filtered_critical_points_quality(
        args,
        model,
        original_critical_points,
        dimOfLayer,
    )
    with open(recovery_file_path, "a") as recovery_file:
        recovery_file.write("---------------------[Step 0]---------------------\n")
        recovery_file.write(
            f"Now we have {len(original_critical_points)} critical points in total from the original list.\n"
        )
        recovery_file.write(
            f"[Cheating]The number of critical points on the target layer is {good_count}.\n"
        )
        recovery_file.write(
            f"[Cheating]The number of bad critical points on deeper layers is {bad_count}.\n"
        )
        recovery_file.write(
            f"[Cheating]The distribution of the critical points on the target layer: {neurons}.\n"
        )

    # type of each element in this_layer_critical_points: [(ratio, critical_point, h_matrix, index)]
    this_layer_critical_points = []
    this_layer_critical_point_count = 0
    this_layer_partial_weight_rows = None
    this_layer_partial_biases = None

    def check_fn(point):
        # check whether a critical point is on the target layer
        # and whether it is caused by a recovered neuron
        if this_layer_partial_weight_rows is None:
            return True
        hidden = matmul(
            forward(point, A, B, with_relu=True),
            this_layer_partial_weight_rows.T,
            this_layer_partial_biases,
        )
        print("[check_fn] hidden", hidden)
        if np.any(np.abs(hidden) < 1e-4):
            return False
        return True

    """
    Start to recover the signature of the target layer
    """
    # Cheat part to speed up
    # filtered_points = []
    # for point in original_critical_points:
    #     layer_id, neuron_id = on_which_hidden_layer(point, model)
    #     if layer_id == LAYER:
    #         filtered_points.append(point)
    # original_critical_points = filtered_points

    # Step 1. Find critical points on the target layer from the list of critical points
    # Solve the noise issue
    print(
        "\n[Step 1]Find critical points on the taget layer from remaining critical points"
    )
    this_layer_new_critical_points = []

    t_filter_start = time.time()
    # If a critical point is identified as good, return
    # CriticalPoint(this_layer_critical_point_count, ratio, point, h_matrix)
    # If a critical point is identified as noise, return
    # CriticalPoint(original index in the list, ratio, point, h_matrix)
    (
        this_layer_new_critical_points,
        noise_critical_points,
        bad_quality_critical_point_indices,
    ) = gather_ratios(
        args,
        this_layer_critical_point_count,
        original_critical_points,
        A,
        B,
        check_fn,
        LAYER,
        COUNT,
        model,
        dimInput,
        dimOfPrevLayer,
        dimOfLayer,
        special,
    )
    t_filter_stop = time.time()
    t_filter = t_filter_stop - t_filter_start
    print("Time for filtering critical points and computing ratios: ", t_filter)

    this_layer_critical_points.extend(this_layer_new_critical_points)
    this_layer_critical_point_count += len(this_layer_new_critical_points)

    # noise_indices: the indices of noise critical points in the original list
    noise_indices = []
    for cp in noise_critical_points:
        noise_indices.append(cp.index)
    # remain_indices: the indices of good critical points in the original list
    remain_indices = []
    for i in range(COUNT):
        if i not in noise_indices:
            remain_indices.append(i)

    print("[Step 1]We filter out", len(noise_critical_points), "noise critical points.")
    print(
        "[Step 1]And we keep",
        len(this_layer_critical_points),
        "critical points for the following steps.",
    )

    # Check the quality of the remained critical points after filtering
    print("\n[Checkpoint]Checking quality of remained critical points after filtering")
    good_count, bad_count, neurons = check_filtered_critical_points_quality(
        args,
        model,
        [cp.point for cp in this_layer_critical_points],
        dimOfLayer,
    )
    with open(recovery_file_path, "a") as recovery_file:
        recovery_file.write("---------------------[Step 1]---------------------\n")
        recovery_file.write(
            f"We filter out {len(noise_critical_points)} noise critical points.\n"
        )
        recovery_file.write(
            f"And we keep {len(this_layer_critical_points)} critical points for the following steps.\n"
        )
        recovery_file.write(
            f"[Cheating]The number of critical points on the target layer is {good_count}.\n"
        )
        recovery_file.write(
            f"[Cheating]The number of bad critical points on deeper layers is {bad_count}.\n"
        )
        recovery_file.write(
            f"[Cheating]The distribution of the critical points on the target layer: {neurons}.\n"
        )

    # Step 2. Use filtered critical points (most are on the target layer) to recover signatures
    # good: critical points having full rank h matrix
    # bad: critical points having small rank h matrix
    good_critical_point_indices = []
    bad_critical_point_indices = []
    special_solutions = []
    kernel_spaces = []

    # filter out duplicate critical points
    # Let's not add points that are identical to ones we've already done.
    # Seems like it will take lots of time --> maybe instead could do this later for partial signature computation, whether or not to do it
    filtered_points = []
    for i, cp1 in enumerate(this_layer_critical_points):
        for cp2 in this_layer_critical_points[i + 1 :]:
            if np.sum((cp1.point - cp2.point) ** 2) ** 0.5 < 1e-10:
                break
        else:
            filtered_points.append(cp1)

    this_layer_critical_points = filtered_points
    print(
        "After filtering duplicates, we have",
        len(this_layer_critical_points),
        "points on the target layer",
    )

    # Identify good critical points with full rank h matrix
    print("\n[Checkpoint]Identify good critical points with full rank h matrix")
    t_identify_start = time.time()
    (
        good_indices,
        bad_indices,
        special_sols,
        ker_spaces,
    ) = identify_good_critical_points(args, model, this_layer_new_critical_points)
    t_identify_stop = time.time()
    t_identify = t_identify_stop - t_identify_start
    print(
        "Time for identifying good CPs with full h rank and computing kernel space of bad CPs: ",
        t_identify,
    )

    good_critical_point_indices.extend(good_indices)
    bad_critical_point_indices.extend(bad_indices)
    special_solutions.extend(special_sols)
    kernel_spaces.extend(ker_spaces)

    print("\n-------------------------")
    print(
        "After obtaining",
        len(this_layer_critical_points),
        "critical points after the filtering, we have:",
    )
    print(
        "Good critical points with full rank h matrix:",
        len(good_critical_point_indices),
    )
    print(
        "Bad critical points with small rank h matrix:", len(bad_critical_point_indices)
    )
    print("Special solutions (length):", len(special_solutions))
    print("Kernel spaces (length):", len(kernel_spaces))

    with open(recovery_file_path, "a") as recovery_file:
        recovery_file.write("---------------------[Step 2]---------------------\n")
        recovery_file.write(
            f"After obtaining {len(this_layer_critical_points)} after the filtering, we have:\n"
        )
        recovery_file.write(
            f"Good critical points with full rank h matrix: {len(good_critical_point_indices)}\n"
        )
        recovery_file.write(
            f"Bad critical points with small rank h matrix: {len(bad_critical_point_indices)}\n"
        )
        recovery_file.write(f"Special solutions (length): {len(special_solutions)}\n")
        recovery_file.write(f"Kernel spaces (length): {len(kernel_spaces)}\n")

    # Step 3. Start to merge partial weights from the filtered critical points
    print("\n-------------------------")
    print("[Step 3]Start to merge critical points to recover weights")
    try:
        (
            components,
            extracted_normals,
            noise_components,
            noise_partial_rows,
            merged_noise_critical_points,
            T,
        ) = cluster_critical_points(
            args,
            model,
            this_layer_critical_points,
            noise_critical_points,
            special_solutions,
            kernel_spaces,
            good_critical_point_indices,
            bad_critical_point_indices,
            dimOfLayer,
            LAYER=LAYER,
            debug=True,
            savePath=savePath,
        )
    except GatherMoreData as e:
        print(
            "[Step 4]Graph solving failed because we didn't explore all sides of at least one neuron"
        )
        print(
            "[Step 4]Fall back to the hyperplane following algorithm in order to get more data"
        )
        # breakpoint()

        clustered_components = e.data[0]
        recovered_weights = e.data[1]
        target_neurons = e.data[2]
        target_indices = e.data[3]
        identified_criticals = e.data[4]
        noise_components = e.data[5]
        noise_partial_rows = e.data[6]
        merged_noise_critical_points = e.data[7]
        T = e.data[8]

        T.append(t_filter)
        T.append(t_identify)

        # The neurons that we need to apply the hyperplane following algorithm
        partial_components = [
            component
            for index, component in enumerate(clustered_components)
            if index in target_neurons
        ]
        partial_rows = [
            weight
            for index, weight in enumerate(recovered_weights)
            if index in target_neurons
        ]

        def mine(r):
            while len(r) > 0:
                yield r[0]
                r = r[1:]

        prev_A, prev_B = A[:-1], B[:-1]

        dead_neurons = []
        # Dead Neurons in the previous layer
        # This assumption makes sense because we assume the previous layers are already recovered
        # and we can also get them from the recovered weights
        if args.real_attack == 0:
            if dimOfLayer == 8 and args.dataset == "mnist":
                # our specfic MNIST-8*8-1 NN
                if args.layerID == 3:
                    dead_neurons = [7]
                if args.layerID == 5:
                    dead_neurons = [4]
                if args.layerID == 6:
                    dead_neurons = [1, 7]
                if args.layerID == 7:
                    dead_neurons = [3]
            elif dimOfLayer == 8 and args.dataset == "cifar10":
                if args.layerID == 6:
                    dead_neurons = [1, 2]
            elif dimOfLayer == 16 and args.dataset == "mnist":
                # our specfic MNIST-16*8-1 NN
                if args.layerID == 7:
                    dead_neurons = [6]
                else:
                    dead_neurons = []
        else:
            if args.layerID == 2:
                recoveryPath = os.path.join(
                    args.output_path,
                    "recovered results",
                    f"Seed{args.seed}-Count500-RealAttack{args.real_attack}",
                    # f"Seed{args.seed}-Count1000-RealAttack{args.real_attack}",
                )
            if args.layerID > 2:
                recoveryPath = os.path.join(
                    args.output_path,
                    "recovered results",
                    f"Seed{args.seed}-Count{args.Count}-RealAttack{args.real_attack}",
                )
            recovered_parameters = np.load(
                os.path.join(recoveryPath, f"recovered_{args.layerID-1}.npz")
            )

            for index, b in enumerate(recovered_parameters["extracted_bias"]):
                if abs(b) == 0:
                    dead_neurons.append(index)
            print("The dead neurons from the previous layer are", dead_neurons)

        print("dead neurons are", dead_neurons)
        T_follow_hyperplane = []
        for i in tqdm(
            range(len(target_neurons)),
            desc="Follow hyperplane",
        ):
            print("\n********************************")
            print("[Step 4]Target Neuron ", target_neurons[i])
            print("[Step 4]Target Indices ", target_indices[i])
            print("[Step 4]Found Critical Points", len(identified_criticals[i]))
            print(
                "[Step 4]Recovered partial weight ratio ",
                partial_rows[i],
            )

            with open(recovery_file_path, "a") as recovery_file:
                recovery_file.write(
                    "---------------------[Step 4]---------------------"
                )
                recovery_file.write(f"Target Neuron {target_neurons[i]}\n")
                recovery_file.write(f"Target Indices {target_indices[i]}\n")
                recovery_file.write(
                    f"Found Critical Points {len(identified_criticals[i])}\n"
                )
                recovery_file.write(
                    f"Recovered partial weight ratio {partial_rows[i]}\n"
                )

            good_indices_not_merged = []
            bad_indices_not_merged = []

            # Find more critical points
            t_follow_hyperplane_start = time.time()
            for target_index in range(len(target_indices[i])):
                target_weight_index = target_indices[i][target_index]
                print(
                    "\nTarget Index ",
                    target_weight_index,
                    "of Target Neuron ",
                    target_neurons[i],
                )

                if not np.isnan(partial_rows[i][target_weight_index]):
                    print("This weight is already recovered in the previous step")
                    continue
                if target_weight_index in dead_neurons:
                    print("This weight is caused by a dead neuron")
                    continue

                more_critical_points = get_more_crit_pts(
                    args,
                    prev_A,
                    prev_B,
                    A[-1],
                    B[-1],
                    mine(identified_criticals[i]),
                    LAYER - 1,
                    model,
                    dead_neurons,
                    dimInput,
                    dimOfPrevLayer,
                    dimOfLayer,
                    special,
                    already_checked_critical_points=True,
                    only_need_positive=True,
                    target_neuron=target_weight_index,
                )
                print(
                    "After using the hyperplane following algorithm, we collect ",
                    len(more_critical_points),
                    "more critical points",
                )

                # Compute their weight ratios
                this_layer_new_critical_points = gather_ratios_without_filter(
                    args,
                    this_layer_critical_point_count,
                    more_critical_points,
                    A,
                    B,
                    check_fn,
                    LAYER,
                    1e6,
                    model,
                    special,
                )

                print("\n[Step 4]Checking quality of new found critical points")
                _, _, _ = check_filtered_critical_points_quality(
                    args,
                    model,
                    [cp.point for cp in this_layer_new_critical_points],
                    dimOfLayer,
                )

                this_layer_critical_points.extend(this_layer_new_critical_points)
                this_layer_critical_point_count += len(this_layer_new_critical_points)

                print(
                    "And now up to ",
                    len(this_layer_critical_points),
                    "critical points are in use",
                )
                print(
                    "And now up to ",
                    len(noise_critical_points),
                    "filtered noise critical points",
                )
                print(
                    "Next index of critical points on the target layer: ",
                    this_layer_critical_point_count,
                )
                print("Done adding")

                # Check the quality of the critical points
                # Identify good critical points
                print(
                    "\n[Checkpoint]Identify good critical points with full rank h matrix"
                )
                (
                    good_indices,
                    bad_indices,
                    special_sols,
                    ker_spaces,
                ) = identify_good_critical_points(
                    args, model, this_layer_new_critical_points
                )
                good_critical_point_indices.extend(good_indices)
                bad_critical_point_indices.extend(bad_indices)
                special_solutions.extend(special_sols)
                kernel_spaces.extend(ker_spaces)

                good_indices_not_merged.extend(good_indices)
                bad_indices_not_merged.extend(bad_indices)

                partial_components[i], partial_rows[i], identified_criticals[i] = (
                    merge_extra_cps_into_a_specific_component(
                        args,
                        model,
                        partial_components[i],
                        partial_rows[i],
                        identified_criticals[i],
                        this_layer_critical_points,
                        good_indices_not_merged,
                        bad_indices_not_merged,
                        special_solutions,
                        kernel_spaces,
                        target_neurons[i],
                    )
                )

            t_follow_hyperplane_stop = time.time()
            t_follow_hyperplane = t_follow_hyperplane_stop - t_follow_hyperplane_start
            print(
                "Time for following hyperplane for target neuron ",
                target_neurons[i],
                ": ",
                t_follow_hyperplane,
            )
            T_follow_hyperplane.append(t_follow_hyperplane)

        print("\n*****************************")
        print("[Step 4]After following the critical hyperplane, we have:")
        for c_count, component in enumerate(clustered_components):
            print("----------------------")
            print("On component", c_count)
            if c_count not in target_neurons:
                print("Recovered weight ratio is", recovered_weights[c_count])
                print("Witness length is", len(component))
            else:
                print(
                    "Before this part, Recovered weight ratio is",
                    recovered_weights[c_count],
                )
                print("Witness length is", len(component))
                print(
                    "After this part, Recovered weight ratio is",
                    partial_rows[target_neurons.index(c_count)],
                )
                print(
                    "Witness length is",
                    len(partial_components[target_neurons.index(c_count)]),
                )

                recovered_weights[c_count] = partial_rows[target_neurons.index(c_count)]
                clustered_components[c_count] = partial_components[
                    target_neurons.index(c_count)
                ]

        print("-------------------------------")
        print("The final clustering results are:")
        print("We have", len(clustered_components), "clustered_components")
        all_criticals = [cp.point for cp in this_layer_critical_points]
        all_criticals = np.array(all_criticals, dtype=np.float64)  # float64

        for c_count, component in enumerate(clustered_components):
            print("----------------------")
            print("On component", c_count, "with indexs", component)
            print("This component has", len(component), "witnesses")
            print("The guessed row is", recovered_weights[c_count])

        with open(recovery_file_path, "a") as recovery_file:
            recovery_file.write("---------------------[Step 4]---------------------\n")
            recovery_file.write(f"After following the critical hyperplane, we have:\n")

        # (Optional) Check the quality of clustered_components
        # This is a part of cheating test
        print("[Step 4]After following the critical hyperplane, we have:")
        check_components_quality(
            args,
            model,
            clustered_components,
            recovered_weights,
            all_criticals,
            savePath,
        )

        # Step 5. Recover the weights and Check the noise components
        # Set nan to zero, since they are dead weights
        for weight in recovered_weights:
            weight[np.isnan(weight)] = 0

        print(
            "\n[Step 5]Number of critical points per component",
            [len(x) for x in clustered_components],
        )
        point_per_component = [
            this_layer_critical_points[x[0]].point for x in clustered_components
        ]

        if len(clustered_components) < dimOfLayer:
            print(
                "We only recovered",
                len(clustered_components),
                "neurons in this layer. We need to set remaining weights as zeros.",
            )
            for _ in range(dimOfLayer - len(clustered_components)):
                recovered_weights.extend([np.zeros(dimOfLayer)])
                point_per_component.append(np.zeros(dimInput))

        extracted_normals = np.array(recovered_weights).T
        print("[Step 5]Extracted normal: ", extracted_normals)
        print("[Step 5]Extracted normals shape: ", extracted_normals.shape)

        # Compute the bias because we know wx+b=0
        extracted_bias = []
        for i in range(dimOfLayer):
            if np.mean(point_per_component[i]) == 0:
                print("we cannot recover the bias for neuron", i)
                extracted_bias.append([0])
                continue

            possible_bias = []
            for cp in clustered_components[i]:
                point = this_layer_critical_points[cp].point
                bias = matmul(
                    forward(point, A, B, with_relu=True),
                    extracted_normals[:, i],
                    c=None,
                )

                added = False
                for b in range(len(possible_bias)):
                    if np.abs(bias - possible_bias[b][0]) < 1e-3:
                        possible_bias[b][1] += 1
                        added = True
                        break
                if not added:
                    possible_bias.append([bias, 1])
            possible_bias = sorted(possible_bias, key=lambda x: x[1], reverse=True)

            extracted_bias.append(possible_bias[0][0])

        # Don't forget to negate it.
        # That's important.
        # No, I definitely didn't forget this line the first time around.
        extracted_bias = -np.array(extracted_bias)

        # For the failed-to-identify neurons, set the bias to zero
        extracted_bias *= np.any(extracted_normals != 0, axis=0)[:, np.newaxis]
        print("[Step 5]Extracted bias: ", extracted_bias)

        with open(recovery_file_path, "a") as recovery_file:
            recovery_file.write("---------------------[Step 4]---------------------\n")
            recovery_file.write("The final clustering results are:\n")
            recovery_file.write(
                f"We have {len(clustered_components)} clustered_components\n"
            )
            recovery_file.write(
                f"Number of critical points per component {[len(x) for x in clustered_components]}\n"
            )
            recovery_file.write(f"Extracted normal: {extracted_normals}\n")
            recovery_file.write(f"Extracted normals shape: {extracted_normals.shape}\n")
            recovery_file.write(f"Extracted bias: {extracted_bias}\n")

        # Check the noise components
        # Some noise critical points may be merged with the good critical points
        print("\n[Step 5]Checking noise components")
        extracted_sign = cheat_on_sign(
            args, model, recovered_weights, clustered_components, all_criticals
        )

        new_recovered_weights, new_clustered_components = add_good_components_back(
            args,
            model,
            A,
            B,
            dimOfLayer,
            extracted_normals,
            extracted_bias,
            extracted_sign,
            noise_components,
            noise_partial_rows,
            merged_noise_critical_points,
            all_criticals,
        )

        with open(recovery_file_path, "a") as recovery_file:
            recovery_file.write("---------------------[Step 5]---------------------\n")
            recovery_file.write("Checking noise components\n")
            recovery_file.write(
                f"For the new recovered weights from noise components\n"
            )
            recovery_file.write(
                f"We find {len(new_clustered_components)} new clustered_components\n"
            )

        # (Optional) Check the quality of components
        # This is a part of cheating test
        print("\nFor the new recovered weights from noise components")
        print("We find", len(new_clustered_components), "new clustered_components")
        check_components_quality(
            args,
            model,
            new_clustered_components,
            new_recovered_weights,
            all_criticals,
            savePath,
        )

        # The new neurons that we need to apply the hyperplane following algorithm
        new_target_neurons = []
        new_target_indices = []
        new_identified_criticals = []

        for i in range(len(new_recovered_weights)):
            if (
                np.any(np.isnan(new_recovered_weights[i]))
                and len(new_clustered_components[i]) >= 3
            ):
                nan_indices = np.argwhere(np.isnan(new_recovered_weights[i])).flatten()
                new_target_neurons.append(i)
                new_target_indices.append(nan_indices)
                new_identified_criticals.append(
                    [all_criticals[x] for x in new_clustered_components[i]]
                )

        t_follow_hyperplane_start = time.time()
        for i in tqdm(
            range(len(new_target_neurons)),
            desc="Follow hyperplane again for necessary noise components",
        ):
            print("\n********************************")
            print("[Step 4]Target Neuron ", new_target_neurons[i])
            print("[Step 4]Target Indices ", new_target_indices[i])
            print("[Step 4]Found Critical Points", len(new_identified_criticals[i]))
            print(
                "[Step 4]Recovered partial weight ratio ",
                new_recovered_weights[i],
            )

            good_indices_not_merged = []
            bad_indices_not_merged = []

            # Find more critical points
            for target_index in range(len(new_target_indices[i])):
                target_weight_index = new_target_indices[i][target_index]
                print(
                    "\nTarget Index ",
                    target_weight_index,
                    "of Target Neuron ",
                    new_target_neurons[i],
                )

                if not np.isnan(new_recovered_weights[i][target_weight_index]):
                    print("This weight is already recovered in the previous step")
                    continue
                if target_weight_index in dead_neurons:
                    print("This weight is caused by a dead neuron")
                    continue

                more_critical_points = get_more_crit_pts(
                    args,
                    prev_A,
                    prev_B,
                    A[-1],
                    B[-1],
                    mine(new_identified_criticals[i]),
                    LAYER - 1,
                    model,
                    dead_neurons,
                    dimInput,
                    dimOfPrevLayer,
                    dimOfLayer,
                    special,
                    already_checked_critical_points=True,
                    only_need_positive=True,
                    target_neuron=target_weight_index,
                )
                print(
                    "After using the hyperplane following algorithm, we collect ",
                    len(more_critical_points),
                    "more critical points",
                )

                # Compute their weight ratios
                this_layer_new_critical_points = gather_ratios_without_filter(
                    args,
                    this_layer_critical_point_count,
                    more_critical_points,
                    A,
                    B,
                    check_fn,
                    LAYER,
                    1e6,
                    model,
                    special,
                )

                print("\n[Step 4]Checking quality of new found critical points")
                _, _, _ = check_filtered_critical_points_quality(
                    args,
                    model,
                    [cp.point for cp in this_layer_new_critical_points],
                    dimOfLayer,
                )

                this_layer_critical_points.extend(this_layer_new_critical_points)
                this_layer_critical_point_count += len(this_layer_new_critical_points)

                print(
                    "And now up to ",
                    len(this_layer_critical_points),
                    "critical points",
                )
                print(
                    "And now up to ",
                    len(noise_critical_points),
                    "remaining points",
                )
                print(
                    "Next index of critical points on the target layer: ",
                    this_layer_critical_point_count,
                )
                print("Done adding")

                # Check the quality of the critical points
                # Identify good critical points
                print(
                    "\n[Checkpoint]Identify good critical points with full rank h matrix"
                )
                (
                    good_indices,
                    bad_indices,
                    special_sols,
                    ker_spaces,
                ) = identify_good_critical_points(
                    args, model, this_layer_new_critical_points
                )
                good_critical_point_indices.extend(good_indices)
                bad_critical_point_indices.extend(bad_indices)
                special_solutions.extend(special_sols)
                kernel_spaces.extend(ker_spaces)

                good_indices_not_merged.extend(good_indices)
                bad_indices_not_merged.extend(bad_indices)

                (
                    new_clustered_components[i],
                    new_recovered_weights[i],
                    new_identified_criticals[i],
                ) = merge_extra_cps_into_a_specific_component(
                    args,
                    model,
                    new_clustered_components[i],
                    new_recovered_weights[i],
                    new_identified_criticals[i],
                    this_layer_critical_points,
                    good_indices_not_merged,
                    bad_indices_not_merged,
                    special_solutions,
                    kernel_spaces,
                    target_neurons[i],
                )

        for i in range(len(new_recovered_weights)):
            clustered_components.append(new_clustered_components[i])
            recovered_weights[len(clustered_components) - 1] = new_recovered_weights[i]

        t_follow_hyperplane_stop = time.time()
        t_follow_hyperplane = t_follow_hyperplane_stop - t_follow_hyperplane_start
        print(
            "Time for following hyperplane for necessary noise components: ",
            t_follow_hyperplane,
        )
        T_follow_hyperplane.append(t_follow_hyperplane)

        print("\n*****************************")
        print(
            "[Step 5]After following the critical hyperplane for necessary noise components, we have:"
        )
        print("The final clustering results are:")
        print("We have", len(clustered_components), "clustered_components")
        all_criticals = [cp.point for cp in this_layer_critical_points]
        all_criticals = np.array(all_criticals, dtype=np.float64)  # float64

        for c_count, component in enumerate(clustered_components):
            print("----------------------")
            print("On component", c_count, "with indexs", component)
            print("This component has", len(component), "witnesses")
            print("The guessed row is", recovered_weights[c_count])

        with open(recovery_file_path, "a") as recovery_file:
            recovery_file.write("---------------------[Step 5]---------------------\n")
            recovery_file.write(
                "After following the critical hyperplane for necessary noise components, we have:\n"
            )

        # (Optional) Check the quality of clustered_components
        # This is a part of cheating test
        print(
            "[Step 5]After following the critical hyperplane for necessary noise components, we have:"
        )
        extracted_sequence, _ = check_components_quality(
            args,
            model,
            clustered_components,
            recovered_weights,
            all_criticals,
            savePath,
        )
        print("Extracted sequence: ", extracted_sequence)

        # Set nan to zero, since they are dead weights
        for weight in recovered_weights:
            weight[np.isnan(weight)] = 0

        print(
            "[Step 5]Number of critical points per component",
            [len(x) for x in clustered_components],
        )
        point_per_component = [
            this_layer_critical_points[x[0]].point for x in clustered_components
        ]

        if len(clustered_components) < dimOfLayer:
            print(
                "We only recovered",
                len(clustered_components),
                "neurons in this layer. We need to set remaining weights as zeros.",
            )
            for _ in range(dimOfLayer - len(clustered_components)):
                point_per_component.append(np.zeros(dimInput))

        extracted_normals = np.array(recovered_weights).T
        print("[Step 5]Extracted normal: ", extracted_normals)
        print("[Step 5]Extracted normals shape: ", extracted_normals.shape)

        # Compute the bias because we know wx+b=0
        extracted_bias = []
        for i in range(dimOfLayer):
            if np.mean(point_per_component[i]) == 0:
                print("we cannot recover the bias for neuron", i)
                extracted_bias.append([0])
                continue

            possible_bias = []
            for cp in clustered_components[i]:
                point = this_layer_critical_points[cp].point
                bias = matmul(
                    forward(point, A, B, with_relu=True),
                    extracted_normals[:, i],
                    c=None,
                )

                added = False
                for b in range(len(possible_bias)):
                    if np.abs(bias - possible_bias[b][0]) < 1e-3:
                        possible_bias[b][1] += 1
                        added = True
                        break
                if not added:
                    possible_bias.append([bias, 1])
            possible_bias = sorted(possible_bias, key=lambda x: x[1], reverse=True)
            extracted_bias.append(possible_bias[0][0])

        # Don't forget to negate it.
        # That's important.
        # No, I definitely didn't forget this line the first time around.
        extracted_bias = -np.array(extracted_bias)

        # For the failed-to-identify neurons, set the bias to zero
        extracted_bias *= np.any(extracted_normals != 0, axis=0)[:, np.newaxis]
        extracted_bias = extracted_bias.flatten()
        print("[Step 5]Extracted bias: ", extracted_bias)

        with open(recovery_file_path, "a") as recovery_file:
            recovery_file.write("---------------------[Step 5]---------------------\n")
            recovery_file.write("The final clustering results are:\n")
            recovery_file.write(
                f"We have {len(clustered_components)} clustered_components\n"
            )
            recovery_file.write(
                f"Number of critical points per component {[len(x) for x in clustered_components]}\n"
            )
            recovery_file.write(f"Extracted normal: {extracted_normals}\n")
            recovery_file.write(f"Extracted normals shape: {extracted_normals.shape}\n")
            recovery_file.write(f"Extracted bias: {extracted_bias}\n")
            recovery_file.write(f"Extracted sequence: {extracted_sequence}\n")

        # Cheat on sign
        extracted_sign = cheat_on_sign(
            args,
            model,
            extracted_normals.T,
            clustered_components,
            all_criticals,
        )

        # assign the sign to the extracted normal and bias by column
        extracted_sign = np.array(extracted_sign)
        print("[Cheating] cheat sign is ", extracted_sign)

        if args.real_attack == 0:
            extracted_sign = cheat_on_sign(
                args,
                model,
                extracted_normals.T,
                clustered_components,
                all_criticals,
            )

            # assign the sign to the extracted normal and bias by column
            extracted_sign = np.array(extracted_sign)
            print("cheat_sign", extracted_sign)

            with open(recovery_file_path, "a") as recovery_file:
                recovery_file.write(f"Real signs are {extracted_sign}\n")
        if args.real_attack == 1 and args.cheat_sign == 1:
            extracted_sign = cheat_on_sign(
                args,
                model,
                extracted_normals.T,
                clustered_components,
                all_criticals,
            )

            # assign the sign to the extracted normal and bias by column
            extracted_sign = np.array(extracted_sign)
            print("cheat_sign", extracted_sign)

            extracted_normals = extracted_normals * extracted_sign
            extracted_bias = extracted_bias * extracted_sign

            with open(recovery_file_path, "a") as recovery_file:
                recovery_file.write(
                    "---------------------[Step 5-2]---------------------\n"
                )
                recovery_file.write(
                    "After assigning the correct signs, the final clustering results are:\n"
                )
                recovery_file.write(f"Real signs are {extracted_sign}\n")
                recovery_file.write(
                    f"We have {len(clustered_components)} clustered_components\n"
                )
                recovery_file.write(
                    f"Number of critical points per component {[len(x) for x in clustered_components]}\n"
                )
                recovery_file.write(f"Extracted normal: {extracted_normals}\n")
                recovery_file.write(
                    f"Extracted normals shape: {extracted_normals.shape}\n"
                )
                recovery_file.write(f"Extracted bias: {extracted_bias}\n")
                recovery_file.write(f"Extracted sequence: {extracted_sequence}\n")

        _, zero_indicies = check_components_quality(
            args,
            model,
            clustered_components,
            recovered_weights,
            all_criticals,
            savePath,
        )
        print("The zero indicies are", zero_indicies)

        critical_groups = []
        for i in range(len(clustered_components)):
            critical_group_i = []
            for j in range(len(clustered_components[i])):
                if j in zero_indicies[i]:
                    continue
                critical_group_i.append(all_criticals[clustered_components[i][j]])

            critical_groups.append(critical_group_i)

        merged_indices = []
        for i in range(len(clustered_components)):
            for j in range(len(clustered_components[i])):
                if clustered_components[i][j] < len(remain_indices):
                    merged_indices.append(remain_indices[clustered_components[i][j]])

        original_critical_points = [
            original_critical_points[i] for i in range(COUNT) if i not in merged_indices
        ]

        return (
            extracted_normals,
            extracted_bias,
            extracted_sign,
            extracted_sequence,
            original_critical_points,
            T,
            T_follow_hyperplane,
            critical_groups,
        )

    print("[test] shape of extracted_normals", len(extracted_normals))
    print("[test] shape of extracted_normals", len(extracted_normals[0]))
    clustered_components = components
    recovered_weights = extracted_normals
    T.append(t_filter)
    T.append(t_identify)

    print("-------------------------------")
    print("[Step 5]Without Hyperplane Following, the final clustering results are:")
    print("We have", len(clustered_components), "clustered_components")
    all_criticals = [cp.point for cp in this_layer_critical_points]
    all_criticals = np.array(all_criticals, dtype=np.float64)  # float64

    for c_count, component in enumerate(clustered_components):
        print("----------------------")
        print("On component", c_count, "with indexs", component)
        print("This component has", len(component), "witnesses")
        # print("The guessed row is", recovered_weights[c_count])

    with open(recovery_file_path, "a") as recovery_file:
        recovery_file.write("---------------------[Step 5]---------------------\n")
        recovery_file.write(
            "Without Hyperplane Following, the final clustering results are:\n"
        )
        recovery_file.write(
            f"We have {len(clustered_components)} clustered_components\n"
        )
        recovery_file.write(
            f"Number of critical points per component {[len(x) for x in clustered_components]}\n"
        )

    # (Optional) Check the quality of clustered_components
    # This is a part of cheating test
    extracted_sequence, _ = check_components_quality(
        args, model, clustered_components, recovered_weights, all_criticals, savePath
    )
    print("Extracted sequence is", extracted_sequence)

    # Set nan to zero, since they are dead weights
    for weight in recovered_weights:
        weight[np.isnan(weight)] = 0

    print(
        "Number of critical points per component",
        [len(x) for x in clustered_components],
    )
    point_per_component = [
        this_layer_critical_points[x[0]].point for x in clustered_components
    ]

    if len(recovered_weights) < dimOfLayer:
        print(
            "We only recovered",
            len(recovered_weights),
            "neurons in this layer. We need to recover more.",
        )
        for _ in range(dimOfLayer - len(recovered_weights)):
            recovered_weights.extend([np.zeros(dimOfPrevLayer)])
            point_per_component.append(np.zeros(dimInput))

    extracted_normals = np.array(recovered_weights).T
    print("Extracted normal: ", extracted_normals)
    print("Extracted normals shape: ", extracted_normals.shape)

    # Compute the bias because we know wx+b=0
    extracted_bias = []
    for i in range(dimOfLayer):
        if np.mean(point_per_component[i]) == 0:
            print("we cannot recover the bias for neuron", i)
            extracted_bias.append([0])
            continue

        possible_bias = []
        for cp in clustered_components[i]:
            point = this_layer_critical_points[cp].point
            bias = matmul(
                forward(point, A, B, with_relu=True),
                extracted_normals[:, i],
                c=None,
            )

            added = False
            for b in range(len(possible_bias)):
                if np.abs(bias - possible_bias[b][0]) < 1e-3:
                    possible_bias[b][1] += 1
                    added = True
                    break
            if not added:
                possible_bias.append([bias, 1])
        possible_bias = sorted(possible_bias, key=lambda x: x[1], reverse=True)
        extracted_bias.append(possible_bias[0][0])

    # Don't forget to negate it.
    # That's important.
    # No, I definitely didn't forget this line the first time around.
    extracted_bias = -np.array(extracted_bias)

    # For the failed-to-identify neurons, set the bias to zero
    extracted_bias *= np.any(extracted_normals != 0, axis=0)[:, np.newaxis]
    extracted_bias = extracted_bias.flatten()
    print("Extracted bias: ", extracted_bias)

    with open(recovery_file_path, "a") as recovery_file:
        recovery_file.write("---------------------[Step 5]---------------------\n")
        recovery_file.write(f"Extracted normal: {extracted_normals}\n")
        recovery_file.write(f"Extracted normals shape: {extracted_normals.shape}\n")
        recovery_file.write(f"Extracted bias: {extracted_bias}\n")
        recovery_file.write(f"Extracted sequence: {extracted_sequence}\n")

    # Cheat on sign
    extracted_sign = cheat_on_sign(
        args,
        model,
        extracted_normals.T,
        clustered_components,
        all_criticals,
    )

    # assign the sign to the extracted normal and bias by column
    extracted_sign = np.array(extracted_sign)
    print("[Cheating] cheat sign is ", extracted_sign)

    if args.real_attack == 0:
        extracted_sign = cheat_on_sign(
            args,
            model,
            extracted_normals.T,
            clustered_components,
            all_criticals,
        )

        # assign the sign to the extracted normal and bias by column
        extracted_sign = np.array(extracted_sign)
        print("cheat_sign", extracted_sign)

        with open(recovery_file_path, "a") as recovery_file:
            recovery_file.write(f"Real signs are {extracted_sign}\n")
    if args.real_attack == 1 and args.cheat_sign == 1:
        extracted_sign = cheat_on_sign(
            args,
            model,
            extracted_normals.T,
            clustered_components,
            all_criticals,
        )

        # assign the sign to the extracted normal and bias by column
        extracted_sign = np.array(extracted_sign)
        print("cheat_sign", extracted_sign)
        print("Extracted normals shape", extracted_normals.shape)

        extracted_normals = extracted_normals * extracted_sign
        extracted_bias = extracted_bias * extracted_sign

        with open(recovery_file_path, "a") as recovery_file:
            recovery_file.write(
                "---------------------[Step 5-2]---------------------\n"
            )
            recovery_file.write(
                "After assigning the correct signs, the final clustering results are:\n"
            )
            recovery_file.write(f"Real signs are {extracted_sign}\n")
            recovery_file.write(
                f"We have {len(clustered_components)} clustered_components\n"
            )
            recovery_file.write(
                f"Number of critical points per component {[len(x) for x in clustered_components]}\n"
            )
            recovery_file.write(f"Extracted normal: {extracted_normals}\n")
            recovery_file.write(f"Extracted normals shape: {extracted_normals.shape}\n")
            recovery_file.write(f"Extracted bias: {extracted_bias}\n")
            recovery_file.write(f"Extracted sequence: {extracted_sequence}\n")

    _, zero_indicies = check_components_quality(
        args,
        model,
        clustered_components,
        recovered_weights,
        all_criticals,
        savePath,
    )
    print("The zero indicies are", zero_indicies)

    critical_groups = []
    for i in range(len(clustered_components)):
        critical_group_i = []
        for j in range(len(clustered_components[i])):
            if j in zero_indicies[i]:
                continue
            critical_group_i.append(all_criticals[clustered_components[i][j]])

        critical_groups.append(critical_group_i)

    merged_indices = []
    for i in range(len(clustered_components)):
        for j in range(len(clustered_components[i])):
            if clustered_components[i][j] < len(remain_indices):
                merged_indices.append(remain_indices[clustered_components[i][j]])

    original_critical_points = [
        original_critical_points[i] for i in range(COUNT) if i not in merged_indices
    ]

    return (
        extracted_normals,
        extracted_bias,
        extracted_sign,
        extracted_sequence,
        original_critical_points,
        T,
        None,
        critical_groups,
    )


def cheat_on_sign(args, model, recovered_weights, clustered_components, all_criticals):
    print("Run the sign cheating algorithm")
    signs = []
    weights, biases = getAllWeightsAndBiases(model)
    A = weights[args.layerID - 1]
    B = biases[args.layerID - 1]

    # For the real attack
    if args.real_attack == 1 and args.layerID == 2:
        recoveryPath = os.path.join(
            args.output_path,
            "recovered results",
            f"Seed{args.seed}-Count500-RealAttack{args.real_attack}",
            # f"Seed{args.seed}-Count1000-RealAttack{args.real_attack}",
        )
        recovered_parameters = np.load(
            os.path.join(recoveryPath, f"recovered_{args.layerID-1}.npz")
        )
        extracted_sequence = recovered_parameters["extracted_sequence"]
        print("The extracted sequence is", extracted_sequence)
    if args.real_attack == 1 and args.layerID > 2:
        recoveryPath = os.path.join(
            args.output_path,
            "recovered results",
            f"Seed{args.seed}-Count{args.Count}-RealAttack{args.real_attack}",
        )
        recovered_parameters = np.load(
            os.path.join(recoveryPath, f"recovered_{args.layerID-1}.npz")
        )
        extracted_sequence = recovered_parameters["extracted_sequence"]
        print("The extracted sequence is", extracted_sequence)

    for i in range(len(clustered_components)):
        if args.debug:
            print("length of all criticals is", len(all_criticals))
            print("The first critical point is No.", clustered_components[i][0])

        for j in range(len(clustered_components[i])):
            point = all_criticals[clustered_components[i][j]]
            layer_id, neuron_id = on_which_hidden_layer(point, model)

            if layer_id == args.layerID - 1:
                if args.real_attack == 1 and args.layerID >= 2:
                    position = extracted_sequence[0]
                    signs.append(
                        np.sign(A[position][neuron_id])
                        / np.sign(recovered_weights[i][0])
                    )
                    break
                else:
                    position = 0
                    while recovered_weights[i][position] == 0:
                        position += 1
                    signs.append(
                        np.sign(A[position][neuron_id])
                        / np.sign(recovered_weights[i][position])
                    )
                    break

    while len(signs) < len(recovered_weights):
        signs.append(0)

    return signs
