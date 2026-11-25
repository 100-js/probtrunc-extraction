import copy
import numpy as np
import math
from tqdm import tqdm
from ..utils import (
    forward,
    getAllWeightsAndBiases,
    on_which_hidden_layer,
    compute_difference,
)
from .classify_critical_points import (
    identify_good_critical_points,
)


# Some components may be a subset of others
def merge_components(args, model, components, partial_rows, threshold=1e-4):
    """
    Merge components that are subsets of others.
    """

    merged_indices = []
    merged_components = copy.deepcopy(components)
    merged_partial_rows = copy.deepcopy(partial_rows)

    for i in tqdm(range(len(merged_components)), desc="Merging components"):
        print("\nParital row", i)
        if i in merged_indices:
            continue

        weight_ratio_i = partial_rows[i]
        weight_ratio_i_non_zero = ~np.isnan(weight_ratio_i)

        for j in range(i + 1, len(components)):
            if j in merged_indices:
                continue

            weight_ratio_j = partial_rows[j]
            weight_ratio_j_non_zero = ~np.isnan(weight_ratio_j)

            difference, _, factor, sequence = compute_difference(
                weight_ratio_j, weight_ratio_i
            )
            print("With Parital row", j, "The diffenece is", difference)

            if difference == 0.0:
                print("The common length is less than 2")
                continue
            if difference < threshold:
                print("Can merge With Parital row", j)
                merged_components[i].extend(merged_components[j])
                merged_indices.append(j)

                new_weight_indices = (
                    weight_ratio_j_non_zero.astype(int)
                    - weight_ratio_i_non_zero.astype(int)
                ) > 0

                if args.debug:
                    print("The new weight indices are:", new_weight_indices)
                for l in range(len(new_weight_indices)):
                    if new_weight_indices[l]:
                        if args.debug:
                            print(merged_partial_rows[i][l])
                            print(merged_partial_rows[i][l])
                        if np.isnan(weight_ratio_i[l]):
                            if sequence == 1:
                                merged_partial_rows[i][l] = (
                                    merged_partial_rows[j][l] * factor
                                )
                            else:
                                merged_partial_rows[i][l] = (
                                    merged_partial_rows[j][l] / factor
                                )

    return merged_components, merged_partial_rows, merged_indices


# Filter noise components
def filter_noise_components(
    args,
    model,
    components,
    partial_rows,
    noise_critical_points,
    expected_neurons,
    threshold=0.1,
):
    filtered_components = copy.deepcopy(components)
    filtered_partial_rows = copy.deepcopy(partial_rows)

    # Remove small components
    largest_component_size = len(components[0])
    largest_second_component_size = len(components[1])
    size_lower_bound = (largest_component_size + largest_second_component_size) / (
        2 * 10
    )
    size_lower_bound = math.ceil(size_lower_bound)
    if size_lower_bound > 10:
        size_lower_bound -= 2

    if args.real_attack == 1:
        dimOfLayer = model.layers[args.layerID].output_shape[1]
        if dimOfLayer == 16:
            size_lower_bound += 2
        if size_lower_bound > 50:
            size_lower_bound -= 2

    small_indices = []
    for i in range(len(components)):
        if len(components[i]) < size_lower_bound:
            small_indices.append(i)
    print("The size lower bound is", size_lower_bound)
    print("The small indices are", small_indices)

    filtered_components = [
        filtered_components[i]
        for i in range(len(filtered_components))
        if i not in small_indices
    ]
    filtered_partial_rows = [
        filtered_partial_rows[i]
        for i in range(len(filtered_partial_rows))
        if i not in small_indices
    ]

    # After filtering, the number of components is still larger than 1.5 * expected_neurons
    if len(filtered_components) > 1.5 * expected_neurons:
        print("The number of components is still larger than 1.5 * expected_neurons")
        filtered_components = filtered_components[: int(1.5 * expected_neurons)]
        filtered_partial_rows = filtered_partial_rows[: int(1.5 * expected_neurons)]

    # The number of noise critical points can be merged into the components
    _, _, noise_special_sols, noise_ker_spaces = identify_good_critical_points(
        args, model, noise_critical_points
    )

    noise_components_count = [0] * len(filtered_components)
    merged_noise_critical_points = [[] for _ in range(len(filtered_components))]

    for i in tqdm(range(len(noise_critical_points)), desc="Filtering noise components"):
        layer_id, neuron_id = on_which_hidden_layer(
            noise_critical_points[i].point, model
        )
        print(
            "[Cheating]This noise critical point is of layer",
            layer_id + 1,
            "Neuron",
            neuron_id + 1,
        )

        if args.debug:
            weights, biases = getAllWeightsAndBiases(model)
            x = noise_critical_points[i].point
            for j in range(0, args.layerID - 1):
                x = np.matmul(x, weights[j]) + biases[j]
                x = x * (x > 0)
            x = np.matmul(x, weights[args.layerID - 1]) + biases[args.layerID - 1]
            print("The output of target layer is:", x)

        noise_h_matrix = noise_critical_points[i].h_matrix
        noise_special_sol = noise_special_sols[i]
        noise_ker_space = noise_ker_spaces[i]

        column_is_zero = np.mean(np.abs(noise_h_matrix) < 1e-8, axis=0) > 0.5
        column_non_zero = ~column_is_zero
        ker_size = np.sum(column_non_zero) - np.linalg.matrix_rank(noise_h_matrix)
        if args.debug:
            print("The non zero part is", column_non_zero)
            print("Thr kernel size is", ker_size)

        for j in range(len(filtered_components)):
            recovered_weight_ratio = filtered_partial_rows[j].copy()
            recovered_weight_non_zero = ~np.isnan(recovered_weight_ratio)
            if args.debug:
                print("The recovered partial weight ratio is:", recovered_weight_ratio)
                print("The non-zero part is:", recovered_weight_non_zero)

            # Set nan to 0
            recovered_weight_ratio_nonan = recovered_weight_ratio.copy()
            recovered_weight_ratio_nonan[np.isnan(recovered_weight_ratio_nonan)] = 0
            good_weight = np.array(recovered_weight_ratio_nonan).reshape(-1, 1)
            column_non_zero_union = recovered_weight_non_zero & column_non_zero

            if args.debug:
                print("The union of two non-zero parts is", column_non_zero_union)

            if ker_size == 0:
                difference, _, _, _ = compute_difference(
                    noise_special_sol, recovered_weight_ratio
                )
                print("For component", j, "The diffenece is", difference)

                if difference == 0.0:
                    print("Not enough intersection")
                    continue

                if args.real_attack == 0:
                    diff_threshold = 1e-5
                else:
                    diff_threshold = 1e-6

                    # for test
                    # diff_threshold = 1e-5
                if difference < diff_threshold:
                    # if difference < 1e-6:
                    print("Yes!!!Yes!!! They are caused by the same neuron.")
                    print("Add the noise critical point to the component", j)
                    merged_noise_critical_points[j].append(
                        noise_critical_points[i].point
                    )
                    noise_components_count[j] += 1
                    break
                else:
                    print("No!!!No!!! They are not caused by the same neuron.")
            else:
                if 0 + ker_size + 2 <= np.sum(column_non_zero_union):
                    print("For component", j, "satisfy the bound!!!")
                    merge_matrix = np.hstack(
                        (
                            -np.array(noise_ker_space),
                            -np.array(noise_special_sol).reshape(-1, 1),
                        )
                    )

                    merge_matrix = merge_matrix[column_non_zero_union]
                    merge_matrix = merge_matrix[: 0 + ker_size + 1, :]
                    if np.linalg.matrix_rank(merge_matrix) < ker_size + 1:
                        print("Something very bad happened")
                        print("The rank of the merge matrix is not enough!!!!")
                        print("We cannot merge them")
                        continue

                    merge_y = -good_weight
                    merge_y = merge_y[column_non_zero_union]
                    merge_y = merge_y[: 0 + ker_size + 1]

                    merge_sol = np.linalg.lstsq(merge_matrix, merge_y)
                    merge_coefficient = merge_sol[0]

                    weight_ratio = np.array(noise_special_sol).reshape(-1, 1)
                    for k in range(ker_size):
                        weight_ratio += (
                            merge_coefficient[k] / merge_coefficient[-1]
                        ) * noise_ker_space[:, k].reshape(-1, 1)

                    # Final Check
                    # last_intersection_index = np.where(column_non_zero_union)[0][-1]
                    weight_ratio_multiply_k = weight_ratio * merge_coefficient[-1]

                    # if (
                    #     abs(
                    #         good_weight[last_intersection_index]
                    #         - weight_ratio_multiply_k[last_intersection_index]
                    #     )
                    #     < 1e-6
                    # ):
                    difference, _, _, _ = compute_difference(
                        good_weight, weight_ratio_multiply_k
                    )
                    print("The difference is", difference)
                    if difference < 1e-6:
                        print("Yes!!!Yes!!! They are caused by the same neuron.")
                        print("Add the noise critical point to the component", j)
                        merged_noise_critical_points[j].append(
                            noise_critical_points[i].point
                        )
                        noise_components_count[j] += 1
                        break
                    else:
                        print("No!!!No!!! They are not caused by the same neuron.")

                else:
                    print("For component", j, "does't satisfy the bound!!!")

    print(
        "\nThe number of noise critical points can be merged into each component:",
        noise_components_count,
    )
    # compute the noise ratio of components
    # K = #merged noise critical points/ |component|
    # the smaller, the better
    noise_components_ratio = []
    for i in range(len(noise_components_count)):
        noise_components_ratio.append(
            noise_components_count[i] / len(filtered_components[i])
        )
    print("\nThe noise_components_ratio is", noise_components_ratio)

    # Check kappa ratio and number of nans
    good_components_index = []
    noise_components_index = []

    if args.real_attack == 1 and expected_neurons == 16:
        nan_threshold = expected_neurons // 3
    else:
        nan_threshold = expected_neurons // 2
    print("The nan threshold is", nan_threshold)

    for i in range(len(filtered_components)):
        if noise_components_ratio[i] == 0:
            print("The noise ratio of component", i, "is 0")
            good_components_index.append(i)
        else:
            nan_in_partial_row = np.isnan(filtered_partial_rows[i])
            nan_count = np.sum(nan_in_partial_row)
            print("The number of nans in component", i, "is", nan_count)

            if noise_components_ratio[i] <= threshold and nan_count < nan_threshold:
                print("The noise ratio of component", i, "is less than the threshold")
                print(
                    "The number of nans in component",
                    i,
                    "is less than the nan threshold",
                )
                good_components_index.append(i)
            else:
                noise_components_index.append(i)
                if noise_components_ratio[i] > threshold:
                    print(
                        "The noise ratio of component",
                        i,
                        "is larger than the threshold",
                    )
                if nan_count >= nan_threshold:
                    print(
                        "The number of nans in component",
                        i,
                        "is larger than the nan threshold",
                    )

    return (
        size_lower_bound,
        small_indices,
        noise_components_count,
        noise_components_ratio,
        merged_noise_critical_points,
        good_components_index,
        noise_components_index,
    )


# Add good components among noise components back
# They are caused by the second kind of noise issue
def add_good_components_back(
    args,
    model,
    A,
    B,
    dimOfLayer,
    extracted_normals,
    extracted_bias,
    cheat_sign,
    noise_components,
    noise_partial_rows,
    merged_noise_critical_points,
    all_criticals,
):
    new_recovered_weights = []
    new_clustered_components = []
    for index, noise_component in enumerate(noise_components):
        print("\nNoise component", index, " has", len(noise_component), "witnesses")

        # Filter out the noise critical points inside the noise component
        zero_indicies = []
        non_zero_indicies = []
        minus_num = 0
        for cp in noise_component:
            point = all_criticals[cp]

            hidden = forward(point, A, B, with_relu=True)
            if args.debug:
                print(
                    "The previous hidden values are:",
                    forward(point, A, B, with_relu=True),
                )
                print(np.matmul(hidden, extracted_normals))
            hidden = np.matmul(hidden, extracted_normals) + extracted_bias.flatten()
            hidden = np.array(hidden)
            hidden = hidden * cheat_sign
            if args.debug:
                print("The hidden values are:", hidden)
                weights, biases = getAllWeightsAndBiases(model)
                x = point
                for i in range(0, args.layerID):
                    x = np.matmul(x, weights[i]) + biases[i]
                    x = x * (x > 0)
                print("The output of target layer is:", x)

            if np.all(hidden <= 0):
                zero_indicies.append(cp)
                minus_num += 1
            else:
                non_zero_indicies.append(cp)

        # Filter out the noise critical points inside the merged noise critical points
        special_noise_cp_num = 0
        for noise_cp in merged_noise_critical_points[index]:
            hidden = forward(noise_cp, A, B, with_relu=True)
            hidden = np.matmul(hidden, extracted_normals) + extracted_bias.flatten()
            hidden = np.array(hidden)
            hidden = hidden * cheat_sign

            if args.debug:
                weights, biases = getAllWeightsAndBiases(model)
                x = noise_cp
                for i in range(0, args.layerID - 1):
                    x = np.matmul(x, weights[i]) + biases[i]
                    x = x * (x > 0)

                    print("The output of previous layer is:", x)
                x = np.matmul(x, weights[args.layerID - 1]) + biases[args.layerID - 1]
                print("The output of target layer is:", x)

            if np.all(hidden <= 0):
                special_noise_cp_num += 1

        print("The number of zero hidden values is", len(zero_indicies))
        print("We have ", minus_num, "all negative hidden values")
        print("non_zero_indicies are", non_zero_indicies)
        print("The number of special noise critical points is", special_noise_cp_num)

        correct_component_size = len(non_zero_indicies)
        correct_merged_noise_size = (
            len(merged_noise_critical_points[index]) - special_noise_cp_num
        )

        if dimOfLayer == 16:
            # 16*8
            threshold = 0.2
            if args.Count == 6000 or args.real_attack == 1:
                threshold = 0.25
        else:
            # 8*8
            threshold = 0.1

            if args.real_attack == 1:
                threshold = 0.2

        if args.real_attack == 1 and dimOfLayer == 16:
            nan_threshold = dimOfLayer // 3
        else:
            nan_threshold = dimOfLayer // 2
        print("The nan threshold is", nan_threshold)

        nan_in_partial_row = np.isnan(noise_partial_rows[index])
        nan_count = np.sum(nan_in_partial_row)
        print("The number of nans in component", index, "is", nan_count)

        print(
            "The new kappa ratio is", correct_merged_noise_size / correct_component_size
        )
        print("The number of nans in component", index, "is", nan_count)
        if (
            correct_merged_noise_size / correct_component_size <= threshold
            and nan_count < nan_threshold
        ):
            print("This noise component is actually a good one!!!!")
            new_recovered_weights.append(noise_partial_rows[index])
            new_clustered_components.append(non_zero_indicies)

    return new_recovered_weights, new_clustered_components
