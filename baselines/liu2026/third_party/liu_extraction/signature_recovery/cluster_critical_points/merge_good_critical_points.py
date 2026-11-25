import numpy as np
import random
import os
import gc
import jax
import jax.numpy as jnp
import networkx as nx
import copy
from tqdm import tqdm
from collections import Counter

from global_vars import (
    GlobalConfig,
    MIN_SAME_SIZE,
    BLOCK_MULTIPLY_FACTOR,
    DEAD_NEURON_THRESHOLD,
)

from ..utils import (
    on_which_hidden_layer,
    getAllWeightsAndBiases,
    compute_difference,
    compute_weight_difference,
    AcceptableFailure,
    GatherMoreData,
)


# ----------
# Helper functions for graph_solve
# ----------
@jax.jit
def process_block(ratios, other_ratios):
    # Let jax efficiently compute pairwise similarity by blocking things.

    differences = jnp.abs(ratios[:, jnp.newaxis, :] - other_ratios[jnp.newaxis, :, :])
    differences = differences / jnp.abs(
        ratios[:, jnp.newaxis, :]
    ) + differences / jnp.abs(other_ratios[jnp.newaxis, :, :])

    close = differences < GlobalConfig.BLOCK_ERROR_TOL * jnp.log(ratios.shape[1])

    if ratios.shape[1] > 100:
        print("ratios.shape[1]: ", ratios.shape[1])
        pairings = jnp.sum(close, axis=2) >= ratios.shape[1] // 2
        # pairings = jnp.sum(close, axis=2) >= 6
        return pairings

    pairings = jnp.sum(close, axis=2) >= max(
        MIN_SAME_SIZE, BLOCK_MULTIPLY_FACTOR * (np.log(ratios.shape[1]) - 2)
    )

    return pairings


@jax.jit
def process_block_first_layer(ratios, other_ratios, threshold):
    print("The graph solving threshold is", threshold)
    # Let jax efficiently compute pairwise similarity by blocking things.
    print("process_block_first_layer", ratios.shape, other_ratios.shape)

    differences = jnp.abs(ratios[:, jnp.newaxis, :] - other_ratios[jnp.newaxis, :, :])
    differences = differences / jnp.abs(
        ratios[:, jnp.newaxis, :]
    ) + differences / jnp.abs(other_ratios[jnp.newaxis, :, :])

    close = differences < GlobalConfig.BLOCK_ERROR_TOL * jnp.log(ratios.shape[1])

    print("ratios.shape[1]: ", ratios.shape[1])
    # pairings = jnp.sum(close, axis=2) >= ratios.shape[1] // 2
    pairings = jnp.sum(close, axis=2) >= threshold // 2
    return pairings


def ratio_normalize(possible_matrix_rows):
    # We get a set of a bunch of numbers
    # a1 b1 c1 d1 e1 f1 g1
    # a2 b2 c2 d2 e2 f2 g2
    # such that some of them are nan
    # We want to compute the pairwise ratios ignoring the nans
    # print("possible_matrix_rows", possible_matrix_rows)

    ratio_evidence = [
        [[] for _ in range(possible_matrix_rows.shape[1])]
        for _ in range(possible_matrix_rows.shape[1])
    ]

    for row in possible_matrix_rows:
        for i in range(len(row)):
            for j in range(len(row)):
                ratio_evidence[i][j].append(row[i] / row[j])

    ratio_evidence = np.array(ratio_evidence, dtype=np.float64)
    # print("ratio_evidence", ratio_evidence)

    # for i in range(ratio_evidence.shape[0]):
    #     for j in range(ratio_evidence.shape[1]):
    #         print("i", i, "j", j, ratio_evidence[i][j] * ratio_evidence[j][i])

    medians = np.nanmedian(ratio_evidence, axis=2)
    errors = (
        np.nanstd(ratio_evidence, axis=2)
        / np.sum(~np.isnan(ratio_evidence), axis=2) ** 0.5
    )
    errors += 1e-2 * (np.sum(~np.isnan(ratio_evidence), axis=2) == 1)
    errors /= np.abs(medians)
    errors[np.isnan(errors)] = 1e6

    ratio_evidence = medians
    # print("ratio_evidence", ratio_evidence)

    # Choose the column with the fewest nans to return
    nancount = np.sum(np.isnan(ratio_evidence), axis=0)

    # print("Column nan count", nancount)

    column_ok = np.min(nancount) == nancount
    # print("Column ok", column_ok)

    best = (None, np.inf)

    cost_i_over_j = ratio_evidence[:, :, np.newaxis]
    cost_j_over_k = ratio_evidence
    cost_i_over_k = cost_i_over_j * cost_j_over_k

    cost_i_j_k = cost_i_over_k
    # print("cost_i_j_k", cost_i_j_k)
    # cost from i through j to k

    for column in range(len(column_ok)):
        if not column_ok[column]:
            continue

        quality = np.nansum(np.abs(cost_i_j_k[:, column, :] - ratio_evidence))
        # print('q',quality)
        if quality < best[1]:
            best = (column, quality)

    column, best_error = best
    print("Best column", column, "with error", best_error)
    # breakpoint()

    return ratio_evidence[:, column], column, best_error


def graph_solve(
    args,
    model,
    critical_points,
    good_indices,
):
    print("\n--------------------")
    print("Start to solve the graph and cluster the critical points")
    print("Length of good critical points with a full rank: ", len(good_indices))

    # 1. Load the critical points and ratios we precomputed
    all_ratios = [cp.ratio for cp in critical_points]
    all_criticals = [cp.point for cp in critical_points]

    all_ratios = np.array(all_ratios, dtype=np.float64)  # float64
    all_criticals = np.array(all_criticals, dtype=np.float64)  # float64

    good_ratios = all_ratios[good_indices]
    good_criticals = all_criticals[good_indices]

    if args.real_attack == 1 and args.layerID == 1:
        # Remove the zero columns from the ratios
        non_zero_index = []
        dimOfInput = all_ratios.shape[1]
        for i in range(len(all_criticals[0])):
            if all_criticals[0][i] != 0:
                non_zero_index.append(i)
        print("non_zero_index:", non_zero_index)
        good_ratios = good_ratios[:, non_zero_index]
        all_ratios = all_ratios[:, non_zero_index]

    # Batch them to be sensibly sized
    ratios_group = [good_ratios[i : i + 1000] for i in range(0, len(good_ratios), 1000)]
    criticals_group = [
        good_criticals[i : i + 1000] for i in range(0, len(good_criticals), 1000)
    ]

    # 2. Compute the similarity pairwise between the ratios we've computed
    if args.debug:
        print("Go up to", len(criticals_group))
    all_pairings = [[] for _ in range(sum(map(len, ratios_group)))]
    for batch_index, (criticals, ratios) in enumerate(
        zip(criticals_group, ratios_group)
    ):
        print("Batch index is:", batch_index)

        # Compute the all-pairs similarity
        axis = list(range(all_ratios.shape[1]))
        random.shuffle(axis)
        axis = axis[:20]
        print("Axis to compute:", axis)
        for dim in axis:
            # We may have an error on one of the directions, so let's try all of them
            scaled_all_ratios = good_ratios / good_ratios[:, dim : dim + 1]
            scaled_ratios = ratios / ratios[:, dim : dim + 1]
            # batch_pairings = process_block(scaled_ratios, scaled_all_ratios)
            if args.layerID == 1 and args.real_attack == 1:
                batch_pairings = process_block_first_layer(
                    scaled_ratios, scaled_all_ratios, len(non_zero_index)
                )
            else:
                batch_pairings = process_block(scaled_ratios, scaled_all_ratios)

            # To get the offset, Compute the cumsum of the length up to batch_index
            batch_offset = sum(map(len, ratios_group[:batch_index]))

            # And now create the graph matching ratios that are similar
            for this_batch_i, global_j in zip(*np.nonzero(np.array(batch_pairings))):
                all_pairings[this_batch_i + batch_offset].append(global_j)

    graph = nx.Graph()
    # Add the edges to the graph, removing self-loops
    graph.add_edges_from(
        [(i, j) for i, js in enumerate(all_pairings) for j in js if abs(i - j) > 1]
    )
    components = list(nx.connected_components(graph))

    # Tranfer the index in good_cp array to the index in all_cp array
    if args.debug:
        print("Transfering the index in good_cp array to the index in all_cp array")
        print(
            "Before transfering, components are [index in good_cp array]: ", components
        )
    for i in range(len(components)):
        component = list(components[i])
        for j in range(len(components[i])):
            component[j] = good_indices[component[j]]
        components[i] = component
    if args.debug:
        print("After transfering, components are [index in all_cp array]: ", components)

    sorted_components = sorted(components, key=lambda x: -len(x))
    print(
        "Graph search found",
        len(components),
        "different components with the following counts",
        list(map(len, sorted_components)),
    )
    print("components are:", sorted_components)

    if args.layerID != 1 or args.dataset != "cifar10":
        previous_num_components = np.inf
        while (
            previous_num_components > len(sorted_components)
            and len(sorted_components) != 0
        ):
            previous_num_components = len(sorted_components)
            candidate_rows = []
            candidate_components = []

            datas = [all_ratios[list(component)] for component in sorted_components]

            # This part is taking lots of time because of each pool init, would have to make it global but switch it to sequential for now
            # results =mp.Pool(MPROC_THREADS//4).map(ratio_normalize, datas)
            results = list(map(ratio_normalize, datas))

            candidate_rows = [x[0] for x in results]
            candidate_components = sorted_components
            candidate_rows = np.array(candidate_rows)

            new_pairings = [[] for _ in range(len(candidate_rows))]

            # Re-do the pairings
            for dim in range(all_ratios.shape[1]):
                scaled_ratios = candidate_rows / candidate_rows[:, dim : dim + 1]
                # batch_pairings = process_block(scaled_ratios, scaled_ratios)
                if args.layerID == 1 and args.real_attack == 1:
                    batch_pairings = process_block_first_layer(
                        scaled_ratios, scaled_ratios, len(non_zero_index)
                    )
                else:
                    batch_pairings = process_block(scaled_ratios, scaled_ratios)

                # And now create the graph matching ratios that are similar
                for this_batch_i, global_j in zip(
                    *np.nonzero(np.array(batch_pairings))
                ):
                    new_pairings[this_batch_i].append(global_j)
            graph = nx.Graph()
            # Add the edges to the graph, ALLOWING self-loops this time
            graph.add_edges_from(
                [(i, j) for i, js in enumerate(new_pairings) for j in js]
            )
            components = list(nx.connected_components(graph))
            print("After re-doing the graph, the components are:", components)

            components = [
                sum([list(candidate_components[y]) for y in comp], [])
                for comp in components
            ]

            sorted_components = sorted(components, key=lambda x: -len(x))

            print(
                "After re-doing the graph, the component counts is",
                len(components),
                "with items",
                list(map(len, sorted_components)),
            )

    # 3. Recover the weights from the components we've found
    guessed_rows = []

    # compute the normalized ratios for each component
    if args.layerID == 1 and args.dataset == "cifar10":
        for c_count, component in enumerate(sorted_components):
            potential_matrix_rows = all_ratios[list(component)]

            # guessed_row = np.nanmean(potential_matrix_rows, axis=0)
            guessed_row = potential_matrix_rows[0]
            max_element = np.max(np.abs(guessed_row))
            guessed_row = guessed_row / max_element
            guessed_rows.append([guessed_row, 0, 0])

            # guessed_row, normalize_axis, normalize_error = ratio_normalize(
            #     potential_matrix_rows
            # )
            # guessed_rows.append([guessed_row, normalize_axis, normalize_error])
    else:
        for c_count, component in enumerate(sorted_components):
            potential_matrix_rows = all_ratios[list(component)]
            guessed_row, normalize_axis, normalize_error = ratio_normalize(
                potential_matrix_rows
            )
            guessed_rows.append([guessed_row, normalize_axis, normalize_error])

    if args.real_attack == 1 and args.layerID == 1:
        dimOfLayer = model.layers[args.layerID].output_shape[1]
        sorted_components = sorted_components[:dimOfLayer]
        guessed_rows = guessed_rows[:dimOfLayer]
        if not guessed_rows:
            raise RuntimeError("No first-layer components were recovered from the available critical points")
        print("Guessed rows:", guessed_rows[0])

        print("dimofinput", dimOfInput)
        for i in range(len(guessed_rows)):
            full_guessed_row = np.zeros(dimOfInput)
            for j in range(len(non_zero_index)):
                full_guessed_row[non_zero_index[j]] = guessed_rows[i][0][j]
            guessed_rows[i][0] = full_guessed_row
        print("Guessed rows:", guessed_rows[0])
        print("non_zero_index:", non_zero_index)

    return sorted_components, guessed_rows


# Merge remaining good critical points with better recovered partial weights again
def merge_good_critical_points_into_components(
    args,
    model,
    critical_points,
    special_sols,
    good_indices_not_merged,
    components,
    partial_rows,
):
    print("\n--------------------")
    print("Start to merge good critical points with better recovered good components")

    # Load the critical points and ratios we precomputed
    all_criticals = [cp.point for cp in critical_points]
    all_criticals = np.array(all_criticals, dtype=np.float64)  # float64

    merged_indices = []
    merged_components = copy.deepcopy(components)
    merged_partial_rows = copy.deepcopy(partial_rows)

    for i in tqdm(
        range(len(good_indices_not_merged)),
        desc="merge good critical points into components",
    ):
        print("------------------------------")
        print("For good critical point", i)
        print(
            "The original index of the good critical point is:",
            good_indices_not_merged[i],
        )
        if i in merged_indices:
            continue

        good_point_weight_ratio = special_sols[good_indices_not_merged[i]]
        good_point_weight_ratio_non_zero = ~np.isnan(good_point_weight_ratio)
        if args.debug:
            print("The good point weight ratio is:", good_point_weight_ratio)
            print("The non-zero part is:", good_point_weight_ratio_non_zero)

        for j in range(len(merged_partial_rows)):
            recovered_weight_ratio = merged_partial_rows[j].copy()
            recovered_weight_non_zero = ~np.isnan(recovered_weight_ratio)

            if args.debug:
                print("The recovered partial weight ratio is:", recovered_weight_ratio)
                print("The non-zero part is:", recovered_weight_non_zero)

            difference, _, factor, sequence = compute_difference(
                good_point_weight_ratio, recovered_weight_ratio
            )
            print("For component", j, "The diffenece is", difference)

            if difference == 0.0:
                print("The common length is less than 2")
                continue

            print("The difference is", difference)
            if args.real_attack == 1:
                # threshold = 1e-6
                threshold = 1e-7
            else:
                threshold = 1e-7

                # for test
                # threshold = 1e-6
            if difference < threshold:
                print("This good critical point can be merged into component", j)
                merged_indices.append(i)
                merged_components[j].append(good_indices_not_merged[i])

                new_weight_indices = (
                    good_point_weight_ratio_non_zero.astype(int)
                    - recovered_weight_non_zero.astype(int)
                ) > 0
                if args.debug:
                    print("The new weight indices are:", new_weight_indices)
                for l in range(len(new_weight_indices)):
                    if new_weight_indices[l]:
                        if args.debug:
                            print(merged_partial_rows[j][l])
                            print(good_point_weight_ratio[l])
                        if np.isnan(merged_partial_rows[j][l]):
                            if sequence == 1:
                                merged_partial_rows[j][l] = (
                                    good_point_weight_ratio[l] * factor
                                )
                            else:
                                merged_partial_rows[j][l] = (
                                    good_point_weight_ratio[l] / factor
                                )
                break

    return merged_components, merged_partial_rows, merged_indices


# Merge good critical points with a specific recovered partial weight
def merge_good_critical_points_into_a_specific_component(
    args,
    model,
    critical_points,
    special_sols,
    good_indices,
    component,
    partial_row,
):
    print("\n--------------------")
    print(
        "Start to merge good critical points with a specific recovered partial weight"
    )

    # Load the critical points and ratios we precomputed
    all_criticals = [cp.point for cp in critical_points]
    all_criticals = np.array(all_criticals, dtype=np.float64)  # float64

    merged_indices = []
    merged_component = copy.deepcopy(component)
    merged_partial_row = copy.deepcopy(partial_row)

    for i in range(len(good_indices)):
        print("------------------------------")
        print("For good critical point", i)
        if i in merged_indices:
            continue

        good_point_weight_ratio = special_sols[good_indices[i]]
        good_point_weight_ratio_non_zero = ~np.isnan(good_point_weight_ratio)
        if args.debug:
            print("The good point weight ratio is:", good_point_weight_ratio)
            print("The non-zero part is:", good_point_weight_ratio_non_zero)
        # print("The good point weight ratio is:", good_point_weight_ratio)
        # print("The non-zero part is:", good_point_weight_ratio_non_zero)

        recovered_weight_ratio = merged_partial_row.copy()
        recovered_weight_non_zero = ~np.isnan(recovered_weight_ratio)
        if args.debug:
            print("The recovered partial weight ratio is:", recovered_weight_ratio)
            print("The non-zero part is:", recovered_weight_non_zero)
        # print("The recovered partial weight ratio is:", recovered_weight_ratio)
        # print("The non-zero part is:", recovered_weight_non_zero)

        common_non_zero = recovered_weight_non_zero & good_point_weight_ratio_non_zero
        if np.sum(common_non_zero) <= 1:
            print("common non zero index is less than 1")
            continue

        difference, _, factor, sequence = compute_difference(
            good_point_weight_ratio, recovered_weight_ratio
        )
        print("The diffenece is", difference)

        if difference == 0.0:
            print("The common length is less than 2")
            continue

        if difference < 1e-7:
            print("This good critical point can be merged into the specfic component")
            merged_indices.append(i)
            merged_component.append(good_indices[i])

            new_weight_indices = (
                good_point_weight_ratio_non_zero.astype(int)
                - recovered_weight_non_zero.astype(int)
            ) > 0
            if args.debug:
                print("The new weight indices are:", new_weight_indices)
            for l in range(len(new_weight_indices)):
                if new_weight_indices[l]:
                    if args.debug:
                        print(merged_partial_row[l])
                        print(good_point_weight_ratio[l])
                    if np.isnan(merged_partial_row[l]):
                        if sequence == 1:
                            merged_partial_row[l] = good_point_weight_ratio[l] * factor
                        else:
                            merged_partial_row[l] = good_point_weight_ratio[l] / factor
        else:
            print("The difference is too large to merge")

    return merged_component, merged_partial_row, merged_indices


# helper function for check_components_quality
def max_element_ratio(lst):
    if not lst:
        return 0, 0
    count = Counter(lst)
    most_common_element, most_common_count = count.most_common(1)[0]
    return most_common_element, most_common_count / len(lst)


def check_components_quality(
    args,
    model,
    components,
    partial_rows,
    all_criticals,
    savePath,
):
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

    weights, biases = getAllWeightsAndBiases(model)

    # Check the quality of components
    # This is a part of cheating test
    print("[Checkpoint]--------------------------")
    print("[Cheating Test] Checking each connected component in turn.")

    sequence = []
    all_zero_indices = []
    for c_count, component in enumerate(components):
        print("\nThe index of the component is", c_count)
        print("This component has:", component)
        print("The size of the component is", len(component))

        with open(recovery_file_path, "a") as recovery_file:
            recovery_file.write(f"\nThe index of the component is {c_count}\n")
            recovery_file.write(f"This component has: {component}\n")
            recovery_file.write(
                f"The recovered weight ratio is {partial_rows[c_count]}\n"
            )
            recovery_file.write(f"The size of the component is {len(component)}\n")

        layers = []
        neurons = []
        for index in component:
            layer_id, neuron_id = on_which_hidden_layer(all_criticals[index], model)
            layers.append(layer_id)
            neurons.append(neuron_id)

        most_common_layer, most_common_layer_ratio = max_element_ratio(layers)
        most_common_neuron, most_common_neuron_ratio = max_element_ratio(neurons)
        # if most_common_layer_ratio > 0.9:
        #     if most_common_neuron_ratio > 0.9:

        zero_indicies = []
        if args.Count == 3000:
            bound = 1
            if args.real_attack == 1 and args.layerID == 1:
            # if args.real_attack == 1:
                bound = 0.9
        else:
            bound = 0.9
        if most_common_layer_ratio >= bound:
            if most_common_neuron_ratio >= bound:
                print("most_common_layer_ratio", most_common_layer_ratio)
                print("most_common_neuron_ratio", most_common_neuron_ratio)
                print("[y] The component seems good. most neurons on the same layer.")

                with open(recovery_file_path, "a") as recovery_file:
                    recovery_file.write(
                        f"most_common_layer_ratio {most_common_layer_ratio}\n"
                    )
                    recovery_file.write(
                        f"most_common_neuron_ratio {most_common_neuron_ratio}\n"
                    )
                    recovery_file.write(
                        "[y] The component seems good. most neurons on the same layer.\n"
                    )
            else:
                print("The most common layer is", most_common_layer + 1)
                print("Layers are", layers)
                print("most_common_layer_ratio", most_common_layer_ratio)
                print("The most common neuron is", most_common_neuron + 1)
                print("Neurons are", neurons)
                print("most_common_neuron_ratio", most_common_neuron_ratio)
                print(
                    "[x] The component contains critical points of different neurons on the same layer."
                )

                with open(recovery_file_path, "a") as recovery_file:
                    recovery_file.write(
                        f"The most common layer is {most_common_layer + 1}\n"
                    )
                    recovery_file.write(f"Layers are {layers}\n")
                    recovery_file.write(
                        f"most_common_layer_ratio {most_common_layer_ratio}\n"
                    )
                    recovery_file.write(
                        f"The most common neuron is {most_common_neuron + 1}\n"
                    )
                    recovery_file.write(f"Neurons are {neurons}\n")
                    recovery_file.write(
                        f"most_common_neuron_ratio {most_common_neuron_ratio}\n"
                    )
                    recovery_file.write(
                        "[x] The component contains critical points of different neurons on the same layer.\n"
                    )

                # Check if the component has the second case of noise issue
                # zero_indicies: the index of the critical points that have only one non-zero element
                zero_indicies = []
                for index in range(len(component)):
                    x = all_criticals[component[index]]
                    for i in range(0, args.layerID):
                        # print("[filter]component index", index)
                        x = np.matmul(x, weights[i]) + biases[i]
                        x = x * (x > 0)
                        # print("[filter]component x", x)

                    if sum(1 for y in x if y != 0) == 1:
                        zero_indicies.append(index)

                    # A special case: the point is not a critical point
                    if layers[index] == -1:
                        zero_indicies.append(index)

                filtered_layers = [
                    layers[index]
                    for index in range(len(layers))
                    if index not in zero_indicies
                ]
                filtered_neurons = [
                    neurons[index]
                    for index in range(len(neurons))
                    if index not in zero_indicies
                ]

                filtered_most_common_layer, filtered_most_common_layer_ratio = (
                    max_element_ratio(filtered_layers)
                )
                filtered_most_common_neuron, filtered_most_common_neuron_ratio = (
                    max_element_ratio(filtered_neurons)
                )

                print("filtered layers", filtered_layers)
                print("filtered neurons", filtered_neurons)

                # if filtered_most_common_layer_ratio > 0.9:
                #     if filtered_most_common_neuron_ratio > 0.9:
                if filtered_most_common_layer_ratio >= bound:
                    if filtered_most_common_neuron_ratio >= bound:
                        print(
                            "[y]After filtering second noise issue, the component seems good."
                        )
                        print(
                            "The most common layer is", filtered_most_common_layer + 1
                        )
                        print(
                            "most_common_layer_ratio", filtered_most_common_layer_ratio
                        )
                        print(
                            "The most common neuron is", filtered_most_common_neuron + 1
                        )
                        print(
                            "most_common_neuron_ratio",
                            filtered_most_common_neuron_ratio,
                        )

                        most_common_layer = filtered_most_common_layer
                        most_common_neuron = filtered_most_common_neuron
                        most_common_layer_ratio = filtered_most_common_layer_ratio
                        most_common_neuron_ratio = filtered_most_common_neuron_ratio

                        with open(recovery_file_path, "a") as recovery_file:
                            recovery_file.write(
                                "[y]After filtering second noise issue, the component seems good.\n"
                            )
                            recovery_file.write(
                                f"The most common layer is {filtered_most_common_layer + 1}\n"
                            )
                            recovery_file.write(
                                f"most_common_layer_ratio {most_common_layer_ratio}\n"
                            )
                            recovery_file.write(
                                f"The most common neuron is {filtered_most_common_neuron + 1}\n"
                            )
                            recovery_file.write(
                                f"most_common_neuron_ratio {most_common_neuron_ratio}\n"
                            )
                    else:
                        print(
                            "[x]After filtering second noise issue, the component is still bad."
                        )
                        print(
                            "The most common layer is", filtered_most_common_layer + 1
                        )
                        print(
                            "most_common_layer_ratio", filtered_most_common_layer_ratio
                        )
                        print(
                            "The most common neuron is", filtered_most_common_neuron + 1
                        )
                        print(
                            "most_common_neuron_ratio",
                            filtered_most_common_neuron_ratio,
                        )
                        with open(recovery_file_path, "a") as recovery_file:
                            recovery_file.write(
                                "[x]After filtering second noise issue, the component is still bad.\n"
                            )
                            recovery_file.write(
                                f"The most common neuron is {most_common_neuron + 1}\n"
                            )
                            recovery_file.write(f"Layers are {layers}\n")
                            recovery_file.write(
                                f"most_common_layer_ratio {most_common_layer_ratio}\n"
                            )
                            recovery_file.write(f"Neurons are {neurons}\n")
                            recovery_file.write(
                                f"most_common_neuron_ratio {most_common_neuron_ratio}\n"
                            )
                        continue
                else:
                    print(
                        "[x]After filtering second noise issue, the component is still bad."
                    )
                    print("The most common layer is", filtered_most_common_layer + 1)
                    print("most_common_layer_ratio", filtered_most_common_layer_ratio)
                    print("The most common neuron is", filtered_most_common_neuron + 1)
                    print(
                        "most_common_neuron_ratio",
                        filtered_most_common_neuron_ratio,
                    )

                    with open(recovery_file_path, "a") as recovery_file:
                        recovery_file.write(
                            "[x]After filtering second noise issue, the component is still bad.\n"
                        )
                        recovery_file.write(
                            f"The most common neuron is {most_common_neuron + 1}\n"
                        )
                        recovery_file.write(f"Layers are {layers}\n")
                        recovery_file.write(
                            f"most_common_layer_ratio {most_common_layer_ratio}\n"
                        )
                        recovery_file.write(f"Neurons are {neurons}\n")
                        recovery_file.write(
                            f"most_common_neuron_ratio {most_common_neuron_ratio}\n"
                        )
                    continue

        else:
            print("The most common layer is", most_common_layer + 1)
            print("Layers are", layers)
            print("most_common_layer_ratio", most_common_layer_ratio)
            print("Neurons are", neurons)
            print("most_common_neuron_ratio", most_common_neuron_ratio)
            print("[x] The component contains critical points on different layers.")

            with open(recovery_file_path, "a") as recovery_file:
                recovery_file.write(
                    f"The most common layer is {most_common_layer + 1}\n"
                )
                recovery_file.write(f"Layers are {layers}\n")
                recovery_file.write(
                    f"most_common_layer_ratio {most_common_layer_ratio}\n"
                )
                recovery_file.write(f"Neurons are {neurons}\n")
                recovery_file.write(
                    f"most_common_neuron_ratio {most_common_neuron_ratio}\n"
                )
                recovery_file.write(
                    "[x] The component contains critical points on different layers.\n"
                )

            # Check if the component has the second case of noise issue
            # zero_indicies: the index of the critical points that have only one non-zero element
            zero_indicies = []
            for index in range(len(component)):
                x = all_criticals[component[index]]
                for i in range(0, args.layerID):
                    # print("[filter]component index", index)
                    x = np.matmul(x, weights[i]) + biases[i]
                    x = x * (x > 0)
                    # print("[filter]component x", x)

                if sum(1 for y in x if y != 0) == 1:
                    zero_indicies.append(index)

                # A special case: the point is not a critical point
                if layers[index] == -1:
                    zero_indicies.append(index)

            filtered_layers = [
                layers[index]
                for index in range(len(layers))
                if index not in zero_indicies
            ]
            filtered_neurons = [
                neurons[index]
                for index in range(len(neurons))
                if index not in zero_indicies
            ]

            filtered_most_common_layer, filtered_most_common_layer_ratio = (
                max_element_ratio(filtered_layers)
            )
            filtered_most_common_neuron, filtered_most_common_neuron_ratio = (
                max_element_ratio(filtered_neurons)
            )

            print("filtered layers", filtered_layers)
            print("filtered neurons", filtered_neurons)

            # if filtered_most_common_layer_ratio > 0.9:
            #     if filtered_most_common_neuron_ratio > 0.9:
            if filtered_most_common_layer_ratio >= bound:
                if filtered_most_common_neuron_ratio >= bound:
                    print(
                        "[y]After filtering second noise issue, the component seems good."
                    )
                    print("The most common neuron is", filtered_most_common_neuron + 1)
                    print("most_common_layer_ratio", filtered_most_common_layer_ratio)
                    print("The most common neuron is", filtered_most_common_neuron + 1)
                    print("most_common_neuron_ratio", filtered_most_common_neuron_ratio)

                    most_common_layer = filtered_most_common_layer
                    most_common_neuron = filtered_most_common_neuron
                    most_common_layer_ratio = filtered_most_common_layer_ratio
                    most_common_neuron_ratio = filtered_most_common_neuron_ratio

                    with open(recovery_file_path, "a") as recovery_file:
                        recovery_file.write(
                            "[y]After filtering second noise issue, the component seems good.\n"
                        )
                        print(
                            "The most common layer is", filtered_most_common_layer + 1
                        )
                        print(
                            "most_common_layer_ratio", filtered_most_common_layer_ratio
                        )
                        print(
                            "The most common neuron is", filtered_most_common_neuron + 1
                        )
                        print(
                            "most_common_neuron_ratio",
                            filtered_most_common_neuron_ratio,
                        )
                else:
                    print(
                        "[x]After filtering second noise issue, the component is still bad."
                    )
                    print("The most common layer is", filtered_most_common_layer + 1)
                    print("most_common_layer_ratio", filtered_most_common_layer_ratio)
                    print("The most common neuron is", filtered_most_common_neuron + 1)
                    print(
                        "most_common_neuron_ratio",
                        filtered_most_common_neuron_ratio,
                    )

                    with open(recovery_file_path, "a") as recovery_file:
                        recovery_file.write(
                            "[x]After filtering second noise issue, the component is still bad.\n"
                        )
                        recovery_file.write(
                            f"The most common neuron is {most_common_neuron + 1}\n"
                        )
                        recovery_file.write(f"Layers are {layers}\n")
                        recovery_file.write(
                            f"most_common_layer_ratio {most_common_layer_ratio}\n"
                        )
                        recovery_file.write(f"Neurons are {neurons}\n")
                        recovery_file.write(
                            f"most_common_neuron_ratio {most_common_neuron_ratio}\n"
                        )
                    continue
            else:
                print(
                    "[x]After filtering second noise issue, the component is still bad."
                )
                print("The most common layer is", filtered_most_common_layer + 1)
                print("most_common_layer_ratio", filtered_most_common_layer_ratio)
                print("The most common neuron is", filtered_most_common_neuron + 1)
                print(
                    "most_common_neuron_ratio",
                    filtered_most_common_neuron_ratio,
                )

                with open(recovery_file_path, "a") as recovery_file:
                    recovery_file.write(
                        "[x]After filtering second noise issue, the component is still bad.\n"
                    )
                    recovery_file.write(
                        f"The most common neuron is {most_common_neuron + 1}\n"
                    )
                    recovery_file.write(f"Layers are {layers}\n")
                    recovery_file.write(
                        f"most_common_layer_ratio {most_common_layer_ratio}\n"
                    )
                    recovery_file.write(f"Neurons are {neurons}\n")
                    recovery_file.write(
                        f"most_common_neuron_ratio {most_common_neuron_ratio}\n"
                    )
                continue

        print(
            "This component is of layer",
            most_common_layer + 1,
            "Neuron",
            most_common_neuron + 1,
        )

        with open(recovery_file_path, "a") as recovery_file:
            recovery_file.write(
                f"This component is of layer {most_common_layer + 1} Neuron {most_common_neuron + 1}\n"
            )

        if most_common_layer + 1 != args.layerID:
            print("[x] A wrong component on deeper layers!!!.")

            with open(recovery_file_path, "a") as recovery_file:
                recovery_file.write("[x] A wrong component on deeper layers!!!.\n")
            continue

        # Check if the component has the second case of noise issue
        # zero_indicies: the index of the critical points that have only one non-zero element
        for index in range(len(component)):
            x = all_criticals[component[index]]
            for i in range(0, args.layerID):
                # print("[filter]component index", index)
                x = np.matmul(x, weights[i]) + biases[i]
                x = x * (x > 0)
                # print("[filter]component x", x)

                if sum(1 for y in x if y != 0) == 1 and index not in zero_indicies:
                    zero_indicies.append(index)

        sequence.append(most_common_neuron)
        all_zero_indices.append(zero_indicies)

        # Create a mask of non-NaN values
        guessed_row = partial_rows[c_count]
        weights_cheat = weights[most_common_layer].T
        weights_cheat = np.array(weights_cheat)
        if args.real_attack == 1 and args.layerID >= 2:
            previous_bias_cheat = biases[most_common_layer - 1]
            previous_bias_cheat = np.array(previous_bias_cheat)

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
            extracted_bias = recovered_parameters["extracted_bias"]

            print("The extracted sequence is", extracted_sequence)
            weights_cheat = weights_cheat[:, extracted_sequence]
            previous_bias_cheat = previous_bias_cheat[extracted_sequence]
            previous_alpha = extracted_bias / previous_bias_cheat

            # if args.layerID == 2:
            #     previous_alpha = np.ones(previous_bias_cheat.shape[0])

            weights_cheat = weights_cheat / previous_alpha
            print("The previous alpha is", previous_alpha)

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
            extracted_bias = recovered_parameters["extracted_bias"]

            print("The extracted sequence is", extracted_sequence)
            weights_cheat = weights_cheat[:, extracted_sequence]
            previous_bias_cheat = previous_bias_cheat[extracted_sequence]
            previous_alpha = extracted_bias / previous_bias_cheat

            # if args.layerID == 2:
            #     previous_alpha = np.ones(previous_bias_cheat.shape[0])

            weights_cheat = weights_cheat / previous_alpha
            print("The previous alpha is", previous_alpha)

        print("Compare the recovered component with the ground truth.")
        difference, _, _, _ = compute_difference(
            guessed_row, weights_cheat[most_common_neuron]
        )
        print("The diffenece is", difference)

        with open(recovery_file_path, "a") as recovery_file:
            recovery_file.write(
                "Compare the recovered component with the ground truth.\n"
            )
            recovery_file.write(f"The diffenece is {difference}\n")

        if difference == 0.0:
            print("This component only recovers one weight!!!!!!!!!")
            with open(recovery_file_path, "a") as recovery_file:
                recovery_file.write(
                    "This component only recovers one weight!!!!!!!!!\n"
                )
        else:
            if difference > 1e-2:
                print("[x] The recovered component is bad.")
                with open(recovery_file_path, "a") as recovery_file:
                    recovery_file.write("[x] The recovered component is bad.\n")
            else:
                print("[y] The recovered component is good")
                with open(recovery_file_path, "a") as recovery_file:
                    recovery_file.write("[y] The recovered component is good.\n")

    if len(sequence) < weights[args.layerID - 1].shape[1]:
        print("The sequence is not complete")
        for i in range(weights[args.layerID - 1].shape[1]):
            if i not in sequence:
                sequence.append(i)
    print("The good sequence is", sequence)
    print("The all zero indices are", all_zero_indices)
    return sequence, all_zero_indices


def compare_models(args, model, point):
    weights, biases = getAllWeightsAndBiases(model)

    A, B = [], []
    dead_neurons = []
    extracted_sequence = []
    for i in range(1, args.layerID):
        # load the extracted weights and biases
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
        recovered_parameters = np.load(os.path.join(recoveryPath, f"recovered_{i}.npz"))

        if i == 1:
            # cheat on the 1st hidden layer
            A.append(weights[0])
            B.append(biases[0])
            # change the sequence of columns in A
            A[0] = A[0][:, recovered_parameters["extracted_sequence"]]
            B[0] = B[0][recovered_parameters["extracted_sequence"]]
            extracted_sequence.append(recovered_parameters["extracted_sequence"])
        else:
            A.append(recovered_parameters["extracted_normal"])
            B.append(recovered_parameters["extracted_bias"])
            extracted_sequence.append(recovered_parameters["extracted_sequence"])

        dead_neuron = []
        for index, b in enumerate(recovered_parameters["extracted_bias"]):
            if abs(b) == 0:
                dead_neuron.append(index)
        dead_neurons.append(dead_neuron)

    x = point
    output_recovered = np.matmul(x, A[0]) + B[0]
    output_recovered = output_recovered * (output_recovered > 0)

    output_real = np.matmul(x, weights[0]) + biases[0]
    output_real = output_real * (output_real > 0)

    sign_difference = np.sign(output_real[extracted_sequence[0]]) - np.sign(
        output_recovered
    )
    print("The index of the layer is 1")
    print("The sign difference on the first layer is", sign_difference)

    for i in range(2, args.layerID):
        print("The index of the layer is", i)
        output_recovered = np.matmul(output_recovered, A[i - 1]) + B[i - 1]
        output_recovered = output_recovered * (output_recovered > 0)

        output_real = np.matmul(output_real, weights[i - 1]) + biases[i - 1]
        output_real = output_real * (output_real > 0)

        sign_difference = np.sign(output_real[extracted_sequence[i - 1]]) - np.sign(
            output_recovered
        )
        print("The sign difference on the layer is", sign_difference)

    output_real = (
        np.matmul(output_real, weights[args.layerID - 1]) + biases[args.layerID - 1]
    )
