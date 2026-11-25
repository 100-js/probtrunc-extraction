# ******************
# This is the main function for clustering critical points
# ******************
import numpy as np
import time
import os
import copy
import gc
import math
from tqdm import tqdm
from ..utils import (
    on_which_hidden_layer,
    getAllWeightsAndBiases,
    get_second_grad_unsigned,
    AcceptableFailure,
    GatherMoreData,
)
from .merge_good_critical_points import (
    graph_solve,
    check_components_quality,
    merge_good_critical_points_into_components,
)
from .merge_bad_critical_points import (
    merge_bad_critical_points_into_components,
    merge_bad_critical_points_with_each_other,
)
from .classify_critical_points import (
    identify_good_critical_points,
)
from .merge_good_critical_points import ratio_normalize
from .operations_on_components import merge_components, filter_noise_components
from global_vars import (
    GlobalConfig,
    DEAD_NEURON_THRESHOLD,
)

fixed_neurons = []


def cluster_critical_points(
    args,
    model,
    critical_points,
    noise_critical_points,
    special_sols,
    ker_spaces,
    good_indices,
    bad_indices,
    expected_neurons,
    LAYER,
    debug=False,
    savePath=None,
):
    global fixed_neurons
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

    resulting_rows = []
    resulting_examples = []

    # good_indices_not_merged: the indices of good critical points that have not been merged
    # bad_indices_not_merged: the indices of bad critical points that have not been merged
    good_indices_not_merged = good_indices.copy()
    bad_indices_not_merged = bad_indices.copy()

    # 1. Load the critical points and ratios we precomputed
    all_ratios = [cp.ratio for cp in critical_points]
    all_criticals = [cp.point for cp in critical_points]

    all_ratios = np.array(all_ratios, dtype=np.float64)  # float64
    all_criticals = np.array(all_criticals, dtype=np.float64)  # float64

    # only for visualization (the benchmark)
    # all_indices = np.arange(len(all_ratios))
    # 2. Cluster the good critical points with full rank
    # TODO: Graph_solve with merged components if we want adaptive COUNT
    t_merge_good_CPs_start = time.time()
    components, guessed_rows = graph_solve(
        args,
        model,
        critical_points,
        # only for visualization (the benchmark)
        # all_indices,
        # correct critical points
        good_indices_not_merged,
    )

    partial_rows = [guessed_row for guessed_row, _, _ in guessed_rows]
    normalize_errors = [normalize_error for _, _, normalize_error in guessed_rows]

    # Filter out components with small size and high error
    good_components_indices = []
    for i in range(len(components)):
        if LAYER == 0:
            if normalize_errors[i] > 1 and len(components[i]) < 3:
                print(normalize_errors[i])
                print(
                    "Component",
                    i,
                    "size less than 3 with high error; this isn't enough to be sure",
                )
                continue
            else:
                good_components_indices.append(i)
                continue

        if args.real_attack == 1 and args.layerID > 5:
            print("For real attack on deep layers, be more sensitive")
            if normalize_errors[i] > 5e-3:
                print(
                    "Component",
                    i,
                    "has high error; this isn't enough to be sure",
                )
                print("The error is", normalize_errors[i])
                continue

        if normalize_errors[i] > 1e-2 and len(components[i]) <= 5:
            print(
                "Component",
                i,
                "size less than 5 with high error; this isn't enough to be sure",
            )
            continue
        else:
            good_components_indices.append(i)

    t_merge_good_CPs_stop = time.time()
    t_merge_good_CPs = t_merge_good_CPs_stop - t_merge_good_CPs_start
    print("Time to merge good critical points:", t_merge_good_CPs)

    # Update the merged components and partial rows
    components = [components[i] for i in good_components_indices]
    partial_rows = [partial_rows[i] for i in good_components_indices]
    normalize_errors = [normalize_errors[i] for i in good_components_indices]

    # Remove the merged good critical points from the good_indices_not_merged list
    for component in components:
        for index in component:
            if index in good_indices_not_merged:
                good_indices_not_merged.remove(index)

    print("\n**************************************************")
    print("[Step 3-1]After only clustering good critical points, we have:")
    for c_count, component in enumerate(components):
        print("----------------------")
        print("On component", c_count, "with indexs", component)
        print("Shape of guessed row", partial_rows[c_count].shape)
        if LAYER != 0:
            print("The guessed row is", partial_rows[c_count])
        print("The guessed row is", partial_rows[c_count])
        print("This component has", len(component), "witnesses")
    print(
        "We still have", len(good_indices_not_merged), "good critical points not merged"
    )
    print("We have", len(bad_indices_not_merged), "bad critical points left")

    with open(recovery_file_path, "a") as recovery_file:
        recovery_file.write("---------------------[Step 3-1]---------------------\n")
        recovery_file.write(f"After only clustering good critical points, we have:\n")
        recovery_file.write(
            f"We still have {len(good_indices_not_merged)} good critical points not merged\n"
        )
        recovery_file.write(
            f"We have {len(bad_indices_not_merged)} bad critical points left\n"
        )

    # (Optional) Check the quality of components
    # This is a part of cheating test
    print("[Step 3-1]After only clustering good critical points, we have:")
    check_components_quality(
        args, model, components, partial_rows, all_criticals, savePath
    )

    if args.layerID == 1:
        t_filter_critical_points_start = time.time()
        if args.real_attack == 1:
            weights_all, biases_all = getAllWeightsAndBiases(model)

            # Filter out bad critical points in components
            if args.dataset == "mnist":
                dimOfLayer = np.array(weights_all[0]).shape[1]
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

            for com_index in range(len(components)):
                print("For the", com_index, "th component")
                component = components[com_index]
                removed_indices = []
                bad_ratios = []

                for point_index in range(len(component)):
                    point = all_criticals[component[point_index]]
                    bad_quality_points_count = 0
                    ratio_sum = 0
                    # Check the quality of the new critical point
                    for i in range(len(point)):
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
                            bad_ratios.append(abs(ratio_val))
                            bad_quality_points_count += 1
                            if bad_quality_points_count >= flat_num:
                                print(
                                    "Too many bad quality directions, stopping the search for this critical point"
                                )
                                filtered = True
                                removed_indices.append(point_index)
                                break

                component = np.delete(component, removed_indices).tolist()
                components[com_index] = component
                partial_rows[com_index] = all_ratios[components[com_index][0]]

            # compute the normalized ratios for each component
            guessed_rows = []
            if args.layerID == 1 and args.dataset == "cifar10":
                for c_count, component in enumerate(components):
                    potential_matrix_rows = all_ratios[list(component)]
                    guessed_row = potential_matrix_rows[0]
                    max_element = np.max(np.abs(guessed_row))
                    guessed_row = guessed_row / max_element
                    guessed_rows.append([guessed_row, 0, 0])

            else:
                for c_count, component in enumerate(components):
                    potential_matrix_rows = all_ratios[list(component)]
                    guessed_row, normalize_axis, normalize_error = ratio_normalize(
                        potential_matrix_rows
                    )
                    guessed_rows.append([guessed_row, normalize_axis, normalize_error])
            partial_rows = [guessed_row for guessed_row, _, _ in guessed_rows]
        t_filter_critical_points_end = time.time()
        t_filter_critical_points = (
            t_filter_critical_points_end - t_filter_critical_points_start
        )
        print(
            "For layer 1, time for filtering critical points is",
            t_filter_critical_points,
        )

        return (
            components,
            partial_rows,
            None,
            None,
            None,
            [t_merge_good_CPs, t_filter_critical_points],
        )
    # breakpoint()

    # 3. Merge the bad critical points into the components
    t_merge_bad_CPs_into_components_start = time.time()
    merged_components, merged_partial_rows, merged_indices = (
        merge_bad_critical_points_into_components(
            args,
            model,
            critical_points,
            special_sols,
            ker_spaces,
            bad_indices_not_merged,
            components,
            partial_rows,
        )
    )
    t_merge_bad_CPs_into_components_stop = time.time()
    t_merge_bad_CPs_into_components = (
        t_merge_bad_CPs_into_components_stop - t_merge_bad_CPs_into_components_start
    )
    print(
        "Time to merge bad critical points into components:",
        t_merge_bad_CPs_into_components,
    )

    # Update the merged components and partial rows
    components = merged_components
    partial_rows = merged_partial_rows

    # Remove the merged bad critical points from the bad_indices_not_merged list
    bad_indices_not_merged = [
        item for i, item in enumerate(bad_indices_not_merged) if i not in merged_indices
    ]

    print("\n**************************************************")
    print(
        "[Step 3-2]After merging bad critical points with recovered partial weights, we have:"
    )
    for c_count, component in enumerate(components):
        print("----------------------")
        print("On component", c_count, "with indexs", component)
        if LAYER != 0:
            print("The guessed row is", partial_rows[c_count])
        print("This component has", len(component), "witnesses")
    print("We have", len(good_indices_not_merged), "good critical points not merged")
    print("We still have", len(bad_indices_not_merged), "bad critical points left")

    with open(recovery_file_path, "a") as recovery_file:
        recovery_file.write("---------------------[Step 3-2]---------------------\n")
        recovery_file.write(
            f"After merging bad critical points with recovered partial weights, we have:\n"
        )
        recovery_file.write(
            f"We have {len(good_indices_not_merged)} good critical points not merged\n"
        )
        recovery_file.write(
            f"We still have {len(bad_indices_not_merged)} bad critical points left\n"
        )

    # (Optional) Check the quality of components
    # This is a part of cheating test
    print(
        "[Step 3-2]After merging bad critical points with recovered partial weights, we have:"
    )
    check_components_quality(
        args, model, components, partial_rows, all_criticals, savePath
    )

    # 4. Merge the bad critical points with each other
    t_merge_bad_CPs_with_each_other_start = time.time()
    merged_components, merged_partial_rows, merged_indices = (
        merge_bad_critical_points_with_each_other(
            args,
            model,
            critical_points,
            special_sols,
            ker_spaces,
            bad_indices_not_merged,
            good_indices_not_merged,
            components,
            partial_rows,
        )
    )
    t_merge_bad_CPs_with_each_other_stop = time.time()
    t_merge_bad_CPs_with_each_other = (
        t_merge_bad_CPs_with_each_other_stop - t_merge_bad_CPs_with_each_other_start
    )
    print(
        "Time to merge bad critical points with each other:",
        t_merge_bad_CPs_with_each_other,
    )

    components = merged_components
    partial_rows = merged_partial_rows

    # Remove the merged good and bad critical points from the list
    good_indices_not_merged = [
        item
        for i, item in enumerate(good_indices_not_merged)
        if i + len(bad_indices_not_merged) not in merged_indices
    ]
    bad_indices_not_merged = [
        item for i, item in enumerate(bad_indices_not_merged) if i not in merged_indices
    ]

    print("\n**************************************************")
    print("[Step 3-3]After merging bad critical points with each other, we have:")
    for c_count, component in enumerate(components):
        print("----------------------")
        print("On component", c_count, "with indexs", component)
        if LAYER != 0:
            print("The guessed row is", partial_rows[c_count])
        print("This component has", len(component), "witnesses")
    print(
        "We still have", len(good_indices_not_merged), "good critical points not merged"
    )
    print("We have", len(bad_indices_not_merged), "bad critical points left")

    with open(recovery_file_path, "a") as recovery_file:
        recovery_file.write("---------------------[Step 3-3]---------------------\n")
        recovery_file.write(
            f"After merging bad critical points with each other, we have:\n"
        )
        recovery_file.write(
            f"We still have {len(good_indices_not_merged)} good critical points not merged\n"
        )
        recovery_file.write(
            f"We have {len(bad_indices_not_merged)} bad critical points left\n"
        )

    # (Optional) Check the quality of components
    # This is a part of cheating test
    print("[Step 3-3]After merging bad critical points with each other, we have:")
    check_components_quality(
        args, model, components, partial_rows, all_criticals, savePath
    )

    # 5. Merge remaining good and bad critical points with better recovered partial weights again
    # until no more critical points can be merged
    print("---------------------------")
    print(
        "[Step 3-4]Start to merge good and bad critical points with better recovered partial weights again"
    )
    t_merge_good_and_bad_CPs_with_components_start = time.time()
    while True:
        merged_num = 0

        merged_components, merged_partial_rows, merged_indices = (
            merge_good_critical_points_into_components(
                args,
                model,
                critical_points,
                special_sols,
                good_indices_not_merged,
                components,
                partial_rows,
            )
        )

        # Update the merged components and partial rows
        merged_num += len(merged_indices)
        components = merged_components
        partial_rows = merged_partial_rows

        # Remove the merged good critical points from the list
        good_indices_not_merged = [
            item
            for i, item in enumerate(good_indices_not_merged)
            if i not in merged_indices
        ]

        print("\n**************************************************")
        print(
            "[Step 3-4]After merging good critical points with better recovered partial weights, we have:"
        )
        for c_count, component in enumerate(components):
            print("----------------------")
            print("On component", c_count, "with indexs", component)
            if LAYER != 0:
                print("The guessed row is", partial_rows[c_count])
            print("This component has", len(component), "witnesses")
        print(
            "We have", len(good_indices_not_merged), "good critical points not merged"
        )
        print("We have", len(bad_indices_not_merged), "bad critical points left")

        merged_components, merged_partial_rows, merged_indices = (
            merge_bad_critical_points_into_components(
                args,
                model,
                critical_points,
                special_sols,
                ker_spaces,
                bad_indices_not_merged,
                components,
                partial_rows,
            )
        )

        # Update the merged components and partial rows
        merged_num += len(merged_indices)
        components = merged_components
        partial_rows = merged_partial_rows

        # Remove the merged bad critical points from the list
        bad_indices_not_merged = [
            item
            for i, item in enumerate(bad_indices_not_merged)
            if i not in merged_indices
        ]

        print("\n**************************************************")
        print(
            "[Step 3-4]After merging bad critical points with better recovered partial weights, we have:"
        )
        for c_count, component in enumerate(components):
            print("----------------------")
            print("On component", c_count, "with indexs", component)
            if LAYER != 0:
                print("The guessed row is", partial_rows[c_count])
            print("This component has", len(component), "witnesses")
        print(
            "We have", len(good_indices_not_merged), "good critical points not merged"
        )
        print("We have", len(bad_indices_not_merged), "bad critical points left")

        print("In this iteration, the number of merged critical points is ", merged_num)
        if merged_num == 0:
            print("No more critical points can be merged.")
            print(
                "We have",
                len(good_indices_not_merged),
                "good critical points not merged",
            )
            print("We have", len(bad_indices_not_merged), "bad critical points left")
            break

    t_merge_good_and_bad_CPs_with_components_stop = time.time()
    t_merge_good_and_bad_CPs_with_components = (
        t_merge_good_and_bad_CPs_with_components_stop
        - t_merge_good_and_bad_CPs_with_components_start
    )
    print(
        "Time to merge good and bad critical points with better components:",
        t_merge_good_and_bad_CPs_with_components,
    )

    print("\n**************************************************")
    print(
        "[Step 3-4]After merging good and bad critical points with better recovered partial weights again, we have:"
    )
    for c_count, component in enumerate(components):
        print("----------------------")
        print("On component", c_count, "with indexs", component)
        if LAYER != 0:
            print("The guessed row is", partial_rows[c_count])
        print("This component has", len(component), "witnesses")
    print(
        "We still have", len(good_indices_not_merged), "good critical points not merged"
    )
    print("We have", len(bad_indices_not_merged), "bad critical points left")

    with open(recovery_file_path, "a") as recovery_file:
        recovery_file.write("---------------------[Step 3-4]---------------------\n")
        recovery_file.write(
            f"After merging good and bad critical points with better recovered partial weights again, we have:\n"
        )
        recovery_file.write(
            f"We still have {len(good_indices_not_merged)} good critical points not merged\n"
        )
        recovery_file.write(
            f"We have {len(bad_indices_not_merged)} bad critical points left\n"
        )

    # (Optional) Check the quality of components
    # This is a part of cheating test
    print(
        "[Step 3-4]After merging good and bad critical points with better recovered partial weights again, we have:"
    )
    check_components_quality(
        args, model, components, partial_rows, all_criticals, savePath
    )

    # 6. Merge components
    # Some components may be a subset of others
    print("\n**************************************************")
    print("[Step 3-5]Start to merge components")

    t_merge_components_start = time.time()
    # merged_components, merged_partial_rows, merged_indices = merge_components(
    #     args, model, components, partial_rows, 1e-4
    # )
    merged_components, merged_partial_rows, merged_indices = merge_components(
        args, model, components, partial_rows, 1e-5
    )
    t_merge_components_stop = time.time()
    t_merge_components = t_merge_components_stop - t_merge_components_start
    print("Time to merge components:", t_merge_components)

    components = [
        item for i, item in enumerate(merged_components) if i not in merged_indices
    ]
    partial_rows = [
        item for i, item in enumerate(merged_partial_rows) if i not in merged_indices
    ]

    # Filter out components with small size
    good_components_indices = []
    for i in range(len(components)):
        if LAYER == 0:
            good_components_indices.append(i)
        if len(components[i]) >= 3 and LAYER != 0:
            good_components_indices.append(i)

    components = [components[i] for i in good_components_indices]
    partial_rows = [partial_rows[i] for i in good_components_indices]

    print("-------------------------------")
    print("[Step 3-5]After megring components, the clustering results are:")
    print("We have", len(components), "components")
    for c_count, component in enumerate(components):
        print("----------------------")
        print("On component", c_count, "with indexs", component)
        print("This component has", len(component), "witnesses")
        if LAYER != 0:
            print("The guessed row is", partial_rows[c_count])

    with open(recovery_file_path, "a") as recovery_file:
        recovery_file.write("---------------------[Step 3-5]---------------------\n")
        recovery_file.write(f"After megring components, the clustering results are:\n")
        recovery_file.write(f"And also filter out small components:\n")

    # (Optional) Check the quality of components
    # This is a part of cheating test
    print("[Step 3-5]After megring components, the clustering results are:")
    print("[Step 3-5]And also filter out small components:")
    check_components_quality(
        args, model, components, partial_rows, all_criticals, savePath
    )

    # Sort the components based on the length of components
    def custom_sort(
        x,
    ):
        component_length = len(x[0])
        return -component_length

    paired_components = list(zip(components, partial_rows))
    sorted_paired_components = sorted(paired_components, key=custom_sort)
    components = [comp for comp, _ in sorted_paired_components]
    partial_rows = [guessed for _, guessed in sorted_paired_components]

    with open(recovery_file_path, "a") as recovery_file:
        recovery_file.write("---------------------[Step 3-5]---------------------\n")
        recovery_file.write(
            f"After sorting components based on the size of components:\n"
        )

    # (Optional) Check the quality of components
    # This is a part of cheating test
    print(
        "[Step 3-5]After sorting components based on the size, the clustering results are:"
    )
    check_components_quality(
        args, model, components, partial_rows, all_criticals, savePath
    )

    # 7. Filter noise components
    print("\n**************************************************")
    print("[Step 3-6]Start to filter noise components")

    # if expected_neurons == 16 or len(model.layers) == 16:
    if expected_neurons == 16:
        # 16*8
        threshold = 0.2
        if args.Count == 6000 or args.real_attack == 1:
            threshold = 0.25
    else:
        # 8*8
        threshold = 0.1

        if args.real_attack == 1:
            threshold = 0.2

    t_filter_noise_components_start = time.time()
    print("layer number", len(model.layers))
    if args.layerID == 2:
        print("This is the second hidden layer")
        print("The noise issue is not serious")
        print("The rank issue does not exist")

        filtered_components = []
        filtered_partial_rows = []
        noise_components = []
        noise_partial_rows = []
        merged_noise_critical_points = []
        with open(recovery_file_path, "a") as recovery_file:
            recovery_file.write(
                "---------------------[Step 3-6]---------------------\n"
            )
            recovery_file.write(
                "This is the second hidden layer, the noise issue is not serious\n"
            )
            recovery_file.write("The rank issue does not exist\n")

    elif args.layerID == len(model.layers) - 2:
        print("This is the last hidden layer")
        print("The noise issue does not exist")

        if args.real_attack == 1:
            print("In the end-to-end attack, some components are not good")
            print("We need to remove them")

            # Remove small components
            filtered_components = copy.deepcopy(components)
            filtered_partial_rows = copy.deepcopy(partial_rows)

            largest_component_size = len(components[0])
            largest_second_component_size = len(components[1])
            size_lower_bound = (
                largest_component_size + largest_second_component_size
            ) / (2 * 10)
            size_lower_bound = math.ceil(size_lower_bound)
            if size_lower_bound > 10:
                size_lower_bound -= 2
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

            components = filtered_components
            partial_rows = filtered_partial_rows

        nan_indices = []
        for i in range(len(components)):
            nan_in_partial_row = np.isnan(partial_rows[i])
            nan_count = np.sum(nan_in_partial_row)
            print("The number of nans in component", i, "is", nan_count)

            if nan_count > expected_neurons // 2:
                nan_indices.append(i)
        print("The index of components with more than half nans is", nan_indices)
        print("They are almost always-off neurons")

        filtered_components = [
            components[i] for i in range(len(components)) if i not in nan_indices
        ]
        filtered_partial_rows = [
            partial_rows[i] for i in range(len(partial_rows)) if i not in nan_indices
        ]
        noise_components = []
        noise_partial_rows = []
        merged_noise_critical_points = []

        # Update the components and partial rows
        components = filtered_components
        partial_rows = filtered_partial_rows

        with open(recovery_file_path, "a") as recovery_file:
            recovery_file.write(
                "---------------------[Step 3-6]---------------------\n"
            )
            recovery_file.write(
                "This is the last hidden layer, the noise issue does not exist\n"
            )
            recovery_file.write(
                "The index of components with more than half nans is: {nan_indices}\n"
            )
            recovery_file.write("They are almost always-off neurons\n")

        print("-------------------------------")
        print(
            "[Step 3-6]After filtering noise components, the final clustering results are:"
        )
        for c_count, component in enumerate(components):
            print("----------------------")
            print("On component", c_count, "with indexs", component)
            print("This component has", len(component), "witnesses")
            if LAYER != 0:
                print("The guessed row is", partial_rows[c_count])

        # (Optional) Check the quality of components
        # This is a part of cheating test
        print(
            "[Step 3-6]After filtering noise components, the final clustering results are:"
        )
        check_components_quality(
            args, model, components, partial_rows, all_criticals, savePath
        )

    else:
        (
            size_lower_bound,
            small_indices,
            noise_components_count,
            noise_components_ratio,
            merged_noise_critical_points,
            good_components_index,
            noise_components_index,
        ) = filter_noise_components(
            args,
            model,
            components,
            partial_rows,
            noise_critical_points,
            expected_neurons,
            threshold,
        )

        print("\n[Step 3-6] The size lower bound is", size_lower_bound)
        print(
            "[Step 3-6]The index of components smaller than the size_lower_bound is:",
            small_indices,
        )
        print(
            "[Step 3-6]The number of noise critical points can be merged into each component:",
            noise_components_count,
        )
        print("[Step 3-6]noise_components_ratio", noise_components_ratio)
        print("[Step 3-6]good_components_index", good_components_index)
        print("[Step 3-6]noise_components_index", noise_components_index)

        filtered_components = [
            components[i] for i in range(len(components)) if i not in small_indices
        ]
        filtered_partial_rows = [
            partial_rows[i] for i in range(len(partial_rows)) if i not in small_indices
        ]
        good_components = [filtered_components[i] for i in good_components_index]
        good_partial_rows = [filtered_partial_rows[i] for i in good_components_index]

        noise_components = [filtered_components[i] for i in noise_components_index]
        noise_partial_rows = [filtered_partial_rows[i] for i in noise_components_index]
        merged_noise_critical_points = [
            merged_noise_critical_points[i] for i in noise_components_index
        ]

        # Update the components and partial rows
        components = good_components
        partial_rows = good_partial_rows

        with open(recovery_file_path, "a") as recovery_file:
            recovery_file.write(
                "---------------------[Step 3-6]---------------------\n"
            )
            recovery_file.write(f"The size lower bound is {size_lower_bound}\n")
            recovery_file.write(
                f"The index of components smaller than the size_lower_bound is: {small_indices}\n"
            )
            recovery_file.write(
                f"The number of noise critical points can be merged into each component: {noise_components_count}\n"
            )
            recovery_file.write(f"noise_components_ratio: {noise_components_ratio}\n")
            recovery_file.write(f"good_components_index: {good_components_index}\n")
            recovery_file.write(f"noise_components_index: {noise_components_index}\n")
            recovery_file.write(
                f"After filtering noise components, the final clustering results are:\n"
            )

    t_filter_noise_components_stop = time.time()
    t_filter_noise_components = (
        t_filter_noise_components_stop - t_filter_noise_components_start
    )
    print("Time to filter noise components:", t_filter_noise_components)

    print("-------------------------------")
    print(
        "[Step 3-6]After filtering noise components, the final clustering results are:"
    )
    for c_count, component in enumerate(components):
        print("----------------------")
        print("On component", c_count, "with indexs", component)
        print("This component has", len(component), "witnesses")
        if LAYER != 0:
            print("The guessed row is", partial_rows[c_count])

    # (Optional) Check the quality of components
    # This is a part of cheating test
    print(
        "[Step 3-6]After filtering noise components, the final clustering results are:"
    )
    check_components_quality(
        args, model, components, partial_rows, all_criticals, savePath
    )

    # 7.1 Filter small components again
    if args.layerID != 2 and args.layerID != len(model.layers) - 2:
        print("\n**************************************************")
        print("[Step 3-6-2]Start to filter small components")

        i = len(components) - 1
        while i > 0:
            if args.Count == 6000 and len(components[i]) >= 3.5 * len(
                components[i - 1]
            ):
                i -= 1
            if args.Count == 3000 and len(components[i - 1]) >= 3 * len(components[i]):
                i -= 1
            else:
                break

        # Update the components and partial rows
        components = components[: i + 1]
        partial_rows = partial_rows[: i + 1]

        if args.real_attack == 1 and args.dataset == "cifar10":
            remove_component_indices = []
            for i in range(len(components) - 1, -1, -1):
                if noise_components_ratio[good_components_index[i]] > 0.1 and len(
                    components[i - 1]
                ) >= 2 * len(components[i]):
                    remove_component_indices.append(i)
                    print(
                        "Remove the",
                        i,
                        "th component with size",
                        len(components[i]),
                        "because it has a large noise ratio",
                    )
                    continue

            # Update the components and partial rows
            components = [
                components[i]
                for i in range(len(components))
                if i not in remove_component_indices
            ]
            partial_rows = [
                partial_rows[i]
                for i in range(len(partial_rows))
                if i not in remove_component_indices
            ]

        print("-------------------------------")
        print(
            "[Step 3-6-2]After filtering noise components, the final clustering results are:"
        )
        for c_count, component in enumerate(components):
            print("----------------------")
            print("On component", c_count, "with indexs", component)
            print("This component has", len(component), "witnesses")
            if LAYER != 0:
                print("The guessed row is", partial_rows[c_count])

        with open(recovery_file_path, "a") as recovery_file:
            recovery_file.write(
                "---------------------[Step 3-6-2]---------------------\n"
            )
            recovery_file.write(
                f"After filtering small components, the final clustering results are:\n"
            )

        # (Optional) Check the quality of components
        # This is a part of cheating test
        print(
            "[Step 3-6-2]After filtering noise components, the final clustering results are:"
        )
        check_components_quality(
            args, model, components, partial_rows, all_criticals, savePath
        )

    # 8 Merge components again using a smaller threshold
    # Some components may be a subset of others
    print("\n**************************************************")
    print("[Step 3-7]Start to merge components")

    t_merge_components_start = time.time()
    if args.Count == 6000:
        merged_components, merged_partial_rows, merged_indices = merge_components(
            args, model, components, partial_rows, 1e-3
        )
    else:
        if args.real_attack == 0:
            merged_components, merged_partial_rows, merged_indices = merge_components(
                args, model, components, partial_rows, 1e-4
            )
        else:
            merged_components, merged_partial_rows, merged_indices = merge_components(
                args, model, components, partial_rows, 1e-3
            )
    t_merge_components_stop = time.time()
    t_merge_components_again = t_merge_components_stop - t_merge_components_start
    print("Time to merge components again:", t_merge_components_again)

    components = [
        item for i, item in enumerate(merged_components) if i not in merged_indices
    ]
    partial_rows = [
        item for i, item in enumerate(merged_partial_rows) if i not in merged_indices
    ]

    print("-------------------------------")
    print("[Step 3-7]After megring components again, the clustering results are:")
    print("We have", len(components), "components")
    for c_count, component in enumerate(components):
        print("----------------------")
        print("On component", c_count, "with indexs", component)
        print("This component has", len(component), "witnesses")
        if LAYER != 0:
            print("The guessed row is", partial_rows[c_count])

    with open(recovery_file_path, "a") as recovery_file:
        recovery_file.write("---------------------[Step 3-7]---------------------\n")
        recovery_file.write(
            f"After megring components again, the clustering results are:\n"
        )

    # (Optional) Check the quality of components
    # This is a part of cheating test
    print("[Step 3-7]After megring components again, the clustering results are:")
    check_components_quality(
        args, model, components, partial_rows, all_criticals, savePath
    )

    # 9. Sort the components based on the number of nans
    def custom_sort(
        x,
    ):
        # We sort by number of components in a cluster except for clusters where full signature is recovered.
        # They are prioritized.
        # This prevents and infinite loop that we previously had in Carlini's implementation.
        num_nans = np.isnan(x[1]).sum()
        component_length = len(x[0])
        if num_nans == 0:
            component_length = np.inf
        return (-component_length, num_nans)

    paired_components = list(zip(components, partial_rows))
    sorted_paired_components = sorted(paired_components, key=custom_sort)
    components = [comp for comp, _ in sorted_paired_components]
    partial_rows = [guessed for _, guessed in sorted_paired_components]

    print("-------------------------------")
    print(
        "[Step 3-8]After the sorting based on the number of nans, the final clustering results are:"
    )
    print("We have", len(components), "components")
    for c_count, component in enumerate(components):
        print("----------------------")
        print("On component", c_count, "with indexs", component)
        print("This component has", len(component), "witnesses")
        if LAYER != 0:
            print("The guessed row is", partial_rows[c_count])

    with open(recovery_file_path, "a") as recovery_file:
        recovery_file.write("---------------------[Step 3-8]---------------------\n")
        recovery_file.write(
            f"After the sorting based on the number of nans, the final clustering results are:\n"
        )

    # (Optional) Check the quality of components
    # This is a part of cheating test
    print(
        "[Step 3-8]After the sorting based on the number of nans, the final clustering results are:"
    )
    check_components_quality(
        args, model, components, partial_rows, all_criticals, savePath
    )

    if len(components) > expected_neurons:
        components = components[:expected_neurons]
        partial_rows = partial_rows[:expected_neurons]

        print(
            "[Step 3-8]We only want",
            expected_neurons,
            "neurons, the final clustering results are:",
        )
        check_components_quality(
            args, model, components, partial_rows, all_criticals, savePath
        )

    # 9. Raise different exceptions based on the clustering results
    # Either colloect more critical points
    # Or follow the hyperplane
    skips_because_of_nan = 0
    failure = None

    print("\n***********************************************")
    print("Start to handle the final clustering results")
    for c_count, component in enumerate(components):
        print("Component", c_count)
        guessed_row = partial_rows[c_count]

        # If there is a nan value in a row, i.e., no full signature recovery for a neuron
        if np.any(np.isnan(guessed_row)) and c_count < expected_neurons:
            print("Got NaN in Component", c_count, ", need more data")

            if len(component) >= 3:
                # If less components than layer size (expected neurons) and more than 2 critical points for the concerned neuron
                print("Component: ", c_count, " triggers GatherMoreData.")
                nan_indices = np.argwhere(np.isnan(guessed_row)).flatten()
                failure = (
                    c_count,
                    nan_indices[0],
                    [critical_points[x] for x in component],
                )
                skips_because_of_nan += 1
            continue

        elif not np.any(np.isnan(guessed_row)):
            # If there is no nan value in a row, i.e., full signature recovery for a neuron
            fixed_neurons.append([component, guessed_row])

            # we add them to resulting_rows => all neurons with full signature recovered
            resulting_rows.append(guessed_row)
            resulting_examples.append([all_criticals[x] for x in component])

    # We set failure when something went wrong but we want to defer crashing
    # (so that we can use the partial solution)
    gc.collect()
    print("\nskips_because_of_nan: ", skips_because_of_nan)

    # Here we know it's a GatherMoreData failure, but we want to only do this
    # if all neurons have been found up to DeadNeuronThreshold and we have at least 3 critical points for the target neuron
    # to initiate the faster targeted critical point search
    if failure is not None:
        # Once enter this branch, we will raise GatherMoreData exception
        # If the number of recovered partial weights is larger than expected neurons, we will recover all neurons
        # If the number of all ratios is greater than DEAD_NEURON_THRESHOLD, we will only recover partial neurons
        # and here are always-on neurons or always-off neurons

        print("Need to raise a previously generated failure.")
        print("We have", len(all_ratios), "neurons in total.")
        print("We have", len(resulting_rows), "neurons with full signature recovered.")
        print(
            "We have", skips_because_of_nan, "neurons with partial signature recovered."
        )

        # Based on the merged results to decide whether to raise GatherMoreData
        target_neurons = []
        target_indices = []
        identified_criticals = []

        for i in range(len(partial_rows)):
            if args.debug:
                print("\nNeuron ", i)
                print("The partial weight ratio is:", partial_rows[i])
                print("The length of critical points is:", len(components[i]))

            if np.any(np.isnan(partial_rows[i])) and len(components[i]) >= 3:
                nan_indices = np.argwhere(np.isnan(partial_rows[i])).flatten()
                if args.debug:
                    print(
                        "We need to raise GatherMoreData for neuron",
                        i,
                        "with indices",
                        nan_indices,
                    )
                target_neurons.append(i)
                target_indices.append(nan_indices)
                identified_criticals.append([all_criticals[x] for x in components[i]])

        print(
            "[Cheackpoint]The information of components that need more critical points to recover weights:"
        )
        print("The target neurons are:", target_neurons)
        print("The target indices are:", target_indices)

        print("\nGatherMoreData")
        raise GatherMoreData(
            (
                components,
                partial_rows,
                target_neurons,
                target_indices,
                identified_criticals,
                noise_components,
                noise_partial_rows,
                merged_noise_critical_points,
                [
                    t_merge_good_CPs,
                    t_merge_bad_CPs_into_components,
                    t_merge_bad_CPs_with_each_other,
                    t_merge_good_and_bad_CPs_with_components,
                    t_merge_components,
                    t_filter_noise_components,
                    t_merge_components_again,
                ],
            )
        )

    print("Successfully returning a solution attempt.\n")
    return (
        components,
        partial_rows,
        noise_components,
        noise_partial_rows,
        merged_noise_critical_points,
        [
            t_merge_good_CPs,
            t_merge_bad_CPs_into_components,
            t_merge_bad_CPs_with_each_other,
            t_merge_good_and_bad_CPs_with_components,
            t_merge_components,
            t_filter_noise_components,
            t_merge_components_again,
        ],
    )


def check_clustered_component_quality(args, model, point, partial_row):
    # This is a part of cheating test
    print("[Cheating Test] Checking each clustered component in turn.")

    layer_id, neuron_id = on_which_hidden_layer(point, model)
    print(
        "It is of layer",
        layer_id + 1,
        "Neuron",
        neuron_id + 1,
    )
    if layer_id + 1 != args.layerID:
        print("[x] A wrong component!!!.")
        return
    else:
        weights, biases = getAllWeightsAndBiases(model)
        weights_cheat = weights[layer_id].T
        weights_cheat = weights_cheat[neuron_id]

        delta = 0
        factor = 0
        for i, weight in enumerate(partial_row):
            if weight is not None:
                factor = weights_cheat[i] / weight
                break
        for i, weight in enumerate(partial_row):
            if weight is not None:
                delta += abs(weight * factor - weights_cheat[i])

        if abs(delta) > 0.01:
            print("[x] The recovered component is bad.")
        else:
            print("[y] The recovered component is good")
    return
