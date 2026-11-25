import numpy as np
import copy
from tqdm import tqdm
from ..utils import on_which_hidden_layer, compute_difference
from .merge_good_critical_points import ratio_normalize


# ***************************************
# Two functions to address the rank issue
# ***************************************


# Merge bad critical points with recovered good components
def merge_bad_critical_points_into_components(
    args,
    model,
    critical_points,
    special_sols,
    ker_spaces,
    bad_indices,
    components,
    partial_rows,
):
    print("\n--------------------")
    print("Start to merge bad critical points with recovered good components")

    # Load the critical points and ratios we precomputed
    all_criticals = [cp.point for cp in critical_points]
    all_h_matrices = [cp.h_matrix for cp in critical_points]
    all_criticals = np.array(all_criticals, dtype=np.float64)  # float64

    merged_indices = []
    merged_components = copy.deepcopy(components)
    merged_partial_rows = copy.deepcopy(partial_rows)

    for i in tqdm(
        range(len(bad_indices)), desc="merge bad critical points with components"
    ):
        if i in merged_indices:
            continue

        bad_critical_point = all_criticals[bad_indices[i]]
        bad_h_matrix = all_h_matrices[bad_indices[i]]
        bad_special_sol = special_sols[bad_indices[i]]
        bad_ker_space = ker_spaces[bad_indices[i]]

        print("----------------------------------------------")
        print("With bad CP ", i)
        print("This is original CP", bad_indices[i])
        layer_id, neuron_id = on_which_hidden_layer(bad_critical_point, model)
        print(
            "[Cheating] This is a critical point of layer",
            layer_id + 1,
            "Neuron",
            neuron_id + 1,
        )

        column_is_zero = np.mean(np.abs(bad_h_matrix) < 1e-8, axis=0) > 0.5
        column_non_zero = ~column_is_zero
        ker_size = np.sum(column_non_zero) - np.linalg.matrix_rank(bad_h_matrix)
        if args.debug:
            print("The non zero part is", column_non_zero)
            print("The kernel size is", ker_size)

        for j in range(len(merged_components)):
            # print("\nFor component", j)
            recovered_weight_ratio = merged_partial_rows[j].copy()
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

            if 0 + ker_size + 2 <= np.sum(column_non_zero_union):
                print("For component", j, "Satisfy the bound!!!")
                merge_matrix = np.hstack(
                    (
                        -np.array(bad_ker_space),
                        -np.array(bad_special_sol).reshape(-1, 1),
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

                weight_ratio = np.array(bad_special_sol).reshape(-1, 1)
                for k in range(ker_size):
                    weight_ratio += (
                        merge_coefficient[k] / merge_coefficient[-1]
                    ) * bad_ker_space[:, k].reshape(-1, 1)

                # Final Check
                # last_intersection_index = np.where(column_non_zero_union)[0][-1]
                weight_ratio_multiply_k = weight_ratio * merge_coefficient[-1]

                difference, _, _, _ = compute_difference(
                    good_weight, weight_ratio_multiply_k
                )
                # print("merge_matrix is", merge_matrix)
                # print("merge_y is", merge_y)
                # print("residual is", merge_y - merge_matrix @ merge_coefficient)
                # print("The good weight is", good_weight)
                # print("The weight ratio multiply k is", weight_ratio_multiply_k)
                # print("The difference is", difference)

                # if (
                #     abs(
                #         good_weight[last_intersection_index]
                #         - weight_ratio_multiply_k[last_intersection_index]
                #     )
                # )

                if args.dataset == "mnist":
                    if args.real_attack == 0:
                        threshold = 1e-5
                    else:
                        threshold = 1e-6

                        # for test
                        # threshold = 1e-5
                elif args.dataset == "cifar10":
                    threshold = 1e-6
                if difference < threshold:
                    # print("For component", j)
                    print("Yes!!!Yes!!! They are caused by the same neuron.")
                    print("We can add CP", bad_indices[i], "into component", j)
                    # print("good_weight", good_weight)
                    # print("weight_ratio_multiply_k", weight_ratio_multiply_k)

                    merged_components[j].append(bad_indices[i])
                    merged_indices.append(i)

                    new_weight_indices = (
                        column_non_zero.astype(int)
                        - recovered_weight_non_zero.astype(int)
                    ) > 0
                    if args.debug:
                        print("The new weight indices are:", new_weight_indices)
                    for l in range(len(new_weight_indices)):
                        if new_weight_indices[l]:
                            if args.debug:
                                print(merged_partial_rows[j][l])
                                print(weight_ratio_multiply_k[l])
                            if np.isnan(merged_partial_rows[j][l]):
                                merged_partial_rows[j][l] = weight_ratio_multiply_k[l]
                    break

                else:
                    print("No!!!No!!! They are not caused by the same neuron.")
            else:
                print("For component", j, "We cannot merge them!!!")

    print("\n*********************************************")
    print("Show the merged components")
    for i, partial_weight in enumerate(merged_partial_rows):
        print("For component", i)
        if args.layerID != 1:
            print(
                "Before merging, the partial weight ratio is:\n",
                partial_rows[i],
            )
            print(
                "After merging, the partial weight ratio is:\n",
                partial_weight,
            )

    return merged_components, merged_partial_rows, merged_indices


# Merge remaining bad critical points with each other
def merge_bad_critical_points_with_each_other(
    args,
    model,
    critical_points,
    special_sols,
    ker_spaces,
    bad_indices_not_merged,
    good_indices_not_merged,
    components,
    partial_rows,
):
    print("\n--------------------")
    print("Start to merge bad critical points with each other")

    # Load the critical points and ratios we precomputed
    all_criticals = [cp.point for cp in critical_points]
    all_h_matrices = [cp.h_matrix for cp in critical_points]
    all_criticals = np.array(all_criticals, dtype=np.float64)  # float64

    merged_indices = []
    merged_components = copy.deepcopy(components)
    merged_partial_rows = copy.deepcopy(partial_rows)

    not_merged_indices = copy.deepcopy(bad_indices_not_merged)
    not_merged_indices.extend(good_indices_not_merged)

    for i in tqdm(
        range(0, len(bad_indices_not_merged)),
        desc="merge bad critical points with each other",
    ):
        if i in merged_indices:
            continue

        new_partial_weight = []
        new_merged_critical_points = []

        bad_critical_point_i = all_criticals[bad_indices_not_merged[i]]
        bad_h_matrix_i = all_h_matrices[bad_indices_not_merged[i]]
        bad_special_sol_i = special_sols[bad_indices_not_merged[i]]
        bad_ker_space_i = ker_spaces[bad_indices_not_merged[i]]

        column_is_zero_i = np.mean(np.abs(bad_h_matrix_i) < 1e-8, axis=0) > 0.5
        column_non_zero_i = ~column_is_zero_i
        ker_size_i = np.sum(column_non_zero_i) - np.linalg.matrix_rank(bad_h_matrix_i)

        first_merged_j = True

        for j in range(i + 1, len(not_merged_indices)):
            if j in merged_indices:
                continue

            bad_critical_point_j = all_criticals[not_merged_indices[j]]
            bad_h_matrix_j = all_h_matrices[not_merged_indices[j]]
            bad_special_sol_j = special_sols[not_merged_indices[j]]
            bad_ker_space_j = ker_spaces[not_merged_indices[j]]

            print("----------------------------------------------")
            print("With bad CPs: ", i, "and", j)
            print(
                "They are original CP",
                not_merged_indices[i],
                "and",
                not_merged_indices[j],
            )
            layer_id_i, neuron_id_i = on_which_hidden_layer(bad_critical_point_i, model)
            print(
                "[Cheating] CP1 is a critical point of layer",
                layer_id_i + 1,
                "Neuron",
                neuron_id_i + 1,
            )

            layer_id_j, neuron_id_j = on_which_hidden_layer(bad_critical_point_j, model)
            print(
                "[Cheating] CP2 is a critical point of layer",
                layer_id_j + 1,
                "Neuron",
                neuron_id_j + 1,
            )

            column_is_zero_j = np.mean(np.abs(bad_h_matrix_j) < 1e-8, axis=0) > 0.5
            column_non_zero_j = ~column_is_zero_j
            ker_size_j = np.sum(column_non_zero_j) - np.linalg.matrix_rank(
                bad_h_matrix_j
            )
            column_non_zero_union = column_non_zero_i & column_non_zero_j

            if args.debug:
                print("[CP1] The non zero part is", column_non_zero_i)
                print("[CP1] The kernel size is", ker_size_i)
                print("[CP2] The non zero part is", column_non_zero_j)
                print("[CP2] The kernel size is", ker_size_j)
                print("The union of two non-zero parts is", column_non_zero_union)

            if ker_size_i + ker_size_j + 2 <= np.sum(column_non_zero_union):
                print("Satisfy the bound!!!")

                if ker_size_i != 0 and ker_size_j == 0:
                    merge_matrix = np.hstack(
                        (
                            np.array(bad_ker_space_i),
                            -np.array(bad_special_sol_j).reshape(-1, 1),
                        )
                    )
                elif ker_size_i != 0 and ker_size_j != 0:
                    merge_matrix = np.hstack(
                        (
                            np.array(bad_ker_space_i),
                            -np.array(bad_ker_space_j),
                            -np.array(bad_special_sol_j).reshape(-1, 1),
                        )
                    )
                merge_matrix = merge_matrix[column_non_zero_union]
                merge_matrix = merge_matrix[: ker_size_i + ker_size_j + 1, :]
                if args.debug:
                    print(
                        "The rank of the merge matrix is:",
                        np.linalg.matrix_rank(merge_matrix),
                    )
                if np.linalg.matrix_rank(merge_matrix) < ker_size_i + ker_size_j + 1:
                    print("Something very bad happened")
                    print("The rank of the merge matrix is not enough!!!!")
                    print(
                        "bad_ker_space_i and bad_ker_space_j are not linear independent"
                    )
                    print("We cannot merge them***")
                    continue

                det_merge_matrix = np.linalg.det(merge_matrix)
                if args.debug:
                    print("Determinant of merge_matrix:", det_merge_matrix)
                if abs(det_merge_matrix) < 1e-6:
                    print("The merge matrix is singular")
                    print("We cannot merge them")
                    continue

                merge_y = -np.array(bad_special_sol_i).reshape(-1, 1)
                merge_y = merge_y[column_non_zero_union]
                merge_y = merge_y[: ker_size_i + ker_size_j + 1]

                merge_sol = np.linalg.lstsq(merge_matrix, merge_y)
                merge_coefficient = merge_sol[0]

                weight_ratio_i = np.array(bad_special_sol_i).reshape(-1, 1)
                for k in range(ker_size_i):
                    weight_ratio_i += merge_coefficient[k] * bad_ker_space_i[
                        :, k
                    ].reshape(-1, 1)

                weight_ratio_j = np.array(bad_special_sol_j).reshape(-1, 1)
                for k in range(ker_size_j):
                    weight_ratio_j += (
                        merge_coefficient[ker_size_i + k] / merge_coefficient[-1]
                    ) * bad_ker_space_j[:, k].reshape(-1, 1)

                # Final Check
                # last_intersection_index = np.where(column_non_zero_union)[0][-1]
                weight_ratio_j_multiply_k = weight_ratio_j * merge_coefficient[-1]
                if args.debug:
                    print("[CP 1] The new weight ratio is:\n", weight_ratio_i)
                    print(
                        "[CP 2] The new weight ratio is:\n",
                        weight_ratio_j_multiply_k,
                    )

                # if (
                #     abs(
                #         weight_ratio_i[last_intersection_index]
                #         - weight_ratio_j_multiply_k[last_intersection_index]
                #     )
                #     < 1e-6
                # ):
                difference, _, _, _ = compute_difference(
                    weight_ratio_i, weight_ratio_j_multiply_k
                )
                print("The difference is", difference)

                if args.real_attack == 0:
                    threshold = 1e-6
                else:
                    threshold = 1e-7
                if difference < threshold:
                    print("Yes!!!Yes!!! They are caused by the same neuron.")

                    # The first time to merge
                    if first_merged_j:
                        merged_indices.append(i)
                        new_merged_critical_points.append(not_merged_indices[i])
                        new_partial_weight = weight_ratio_i.copy()
                        first_merged_j = False

                    merged_indices.append(j)
                    new_merged_critical_points.append(not_merged_indices[j])

                    new_partial_weight = new_partial_weight.flatten()
                    new_partial_weight_non_zero = ~np.isnan(new_partial_weight)

                    new_weight_indices = (
                        column_non_zero_j.astype(int)
                        - new_partial_weight_non_zero.astype(int)
                    ) > 0
                    if args.debug:
                        print("The new indices are:", new_weight_indices)

                    same_indices = np.where(
                        new_partial_weight_non_zero & column_non_zero_j
                    )[0]
                    common_index = same_indices[0]
                    factor = (
                        new_partial_weight[common_index]
                        / weight_ratio_j_multiply_k[common_index]
                    )

                    for l in range(len(new_weight_indices)):
                        if new_weight_indices[l]:
                            if args.debug:
                                print(new_partial_weight[l])
                                print(weight_ratio_j_multiply_k[l])
                                print(factor)
                            if np.isnan(new_partial_weight[l]):
                                new_partial_weight[l] = (
                                    weight_ratio_j_multiply_k[l] * factor
                                )

                else:
                    print("No!!!No!!! They are not caused by the same neuron.")

            else:
                print("We cannot merge them")

        print("***************************************************")
        print("From CP ", i, "we get the new partial weight ratio:")
        print(new_partial_weight)

        if len(new_partial_weight) > 0:
            merged = False
            for j, partial_weight in enumerate(merged_partial_rows):
                new_partial_weight = new_partial_weight.flatten()
                difference, _, _, _ = compute_difference(
                    partial_weight, new_partial_weight
                )
                print("For component", j, "The diffenece is", difference)

                if difference == 0.0:
                    print("The common length is less than 2")
                    continue
                else:
                    # if difference < 1e-4:
                    if difference < 1e-5:
                        # Merge it with a recovered row
                        print(
                            "[y] The new partial weight ratio and the component are casued by the same neuron"
                        )

                        partial_weight_column_non_zero = ~np.isnan(partial_weight)
                        new_partial_weight_non_zero = ~np.isnan(new_partial_weight)

                        new_weight_indices = (
                            partial_weight_column_non_zero.astype(int)
                            - new_partial_weight_non_zero.astype(int)
                        ) > 0
                        if args.debug:
                            print("The new indices are:", new_weight_indices)

                        same_indices = np.where(
                            partial_weight_column_non_zero & new_partial_weight_non_zero
                        )[0]
                        common_index = same_indices[0]
                        factor = (
                            new_partial_weight[common_index]
                            / partial_weight[common_index]
                        )

                        for l in range(len(new_weight_indices)):
                            if new_weight_indices[l]:
                                if args.debug:
                                    print(partial_weight[l])
                                    print(new_partial_weight[l])
                                    print(factor)
                                new_partial_weight[l] = partial_weight[l] * factor
                        print("Finally, the new partial weight ratio is:")
                        print(new_partial_weight)

                        merged_partial_rows[j] = new_partial_weight
                        merged_components[j].extend(new_merged_critical_points)
                        merged = True
                        break
                    else:
                        print(
                            "[x] The new partial weight ratio and the component are not casued by the same neuron"
                        )
            if not merged:
                print("Find a new neuron's partial weight ratio")
                merged_partial_rows = np.vstack(
                    (merged_partial_rows, new_partial_weight)
                )
                merged_components.extend([new_merged_critical_points])

    print("\n*********************************************")
    print("Show the merged components")
    for i, partial_weight in enumerate(merged_partial_rows):
        if i < len(partial_rows):
            if args.layerID != 1:
                print(
                    "Before merging, the partial weight ratio is:\n",
                    partial_rows[i],
                )
                print(
                    "After merging, the partial weight ratio is:\n",
                    partial_weight,
                )
        else:
            print(
                "We find a new neuron's partial weight ratio:\n",
                partial_weight,
            )

    return merged_components, merged_partial_rows, merged_indices


# Merge bad critical points with a specific recovered partial weight
def merge_bad_critical_points_into_a_specific_component(
    args,
    model,
    critical_points,
    special_sols,
    ker_spaces,
    bad_indices,
    component,
    partial_row,
):
    print("\n--------------------")
    print("Start to merge bad critical points with recovered good components")

    # Load the critical points and ratios we precomputed
    all_criticals = [cp.point for cp in critical_points]
    all_h_matrices = [cp.h_matrix for cp in critical_points]
    all_criticals = np.array(all_criticals, dtype=np.float64)  # float64

    merged_indices = []
    merged_component = copy.deepcopy(component)
    merged_partial_row = copy.deepcopy(partial_row)
    weak_indices = []
    while True:
        weak_len_before = len(weak_indices)

        for i in range(len(bad_indices)):
            if i in merged_indices:
                continue

            bad_critical_point = all_criticals[bad_indices[i]]
            bad_h_matrix = all_h_matrices[bad_indices[i]]
            bad_special_sol = special_sols[bad_indices[i]]
            bad_ker_space = ker_spaces[bad_indices[i]]

            print("----------------------------------------------")
            print("With bad CP ", i)
            print("This is original CP", bad_indices[i])
            layer_id, neuron_id = on_which_hidden_layer(bad_critical_point, model)
            print(
                "[Cheating] This is a critical point of layer",
                layer_id + 1,
                "Neuron",
                neuron_id + 1,
            )

            column_is_zero = np.mean(np.abs(bad_h_matrix) < 1e-8, axis=0) > 0.5
            column_non_zero = ~column_is_zero
            ker_size = np.sum(column_non_zero) - np.linalg.matrix_rank(bad_h_matrix)
            if args.debug:
                print("The non zero part is", column_non_zero)
                print("Thr kernel size is", ker_size)

            recovered_weight_ratio = merged_partial_row.copy()
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

            if 0 + ker_size + 2 <= np.sum(column_non_zero_union):
                print("Satisfy the bound!!!")
                merge_matrix = np.hstack(
                    (
                        -np.array(bad_ker_space),
                        -np.array(bad_special_sol).reshape(-1, 1),
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

                weight_ratio = np.array(bad_special_sol).reshape(-1, 1)
                for k in range(ker_size):
                    weight_ratio += (
                        merge_coefficient[k] / merge_coefficient[-1]
                    ) * bad_ker_space[:, k].reshape(-1, 1)

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
                    merged_component.append(bad_indices[i])
                    print("We can add CP", bad_indices[i])
                    merged_indices.append(i)

                    new_weight_indices = (
                        column_non_zero.astype(int)
                        - recovered_weight_non_zero.astype(int)
                    ) > 0
                    if args.debug:
                        print("The new weight indices are:", new_weight_indices)
                    for l in range(len(new_weight_indices)):
                        if new_weight_indices[l]:
                            if args.debug:
                                print(merged_partial_row[l])
                                print(weight_ratio_multiply_k[l])
                            if np.isnan(merged_partial_row[l]):
                                merged_partial_row[l] = weight_ratio_multiply_k[l]

                # else:
                #     print("No!!!No!!! They are not caused by the same neuron.")

            # We believe that the bad critical point is caused by the same neuron
            elif 0 + ker_size + 1 <= np.sum(column_non_zero_union):
                print("A weak critical point")
                if i not in weak_indices:
                    print("Add it into weak indices")
                    weak_indices.append(i)

            else:
                print("We cannot merge them***")

        print("\n*********************************************")
        print("[Strong condition]Show the merged components")
        if args.layerID != 1:
            print(
                "Before merging, the partial weight ratio is:\n",
                partial_row,
            )
            print(
                "After merging, the partial weight ratio is:\n",
                merged_partial_row,
            )

        weak_len_after = len(weak_indices)
        print(
            "[Weak condition]The length of new weak indices is",
            weak_len_after - weak_len_before,
        )
        if weak_len_after - weak_len_before == 0:
            break

        possible_weight_flag = [False] * len(merged_partial_row)
        possible_weight = [[] for _ in range(len(merged_partial_row))]
        for i in range(len(merged_partial_row)):
            if not np.isnan(merged_partial_row[i]):
                possible_weight_flag[i] = True
                possible_weight[i] = [merged_partial_row[i], 1]

        for i in weak_indices:
            bad_critical_point = all_criticals[bad_indices[i]]
            bad_h_matrix = all_h_matrices[bad_indices[i]]
            bad_special_sol = special_sols[bad_indices[i]]
            bad_ker_space = ker_spaces[bad_indices[i]]

            print("----------------------------------------------")
            print("With bad CP ", i)
            print("This is original CP", bad_indices[i])
            layer_id, neuron_id = on_which_hidden_layer(bad_critical_point, model)
            print(
                "[Cheating] This is a critical point of layer",
                layer_id + 1,
                "Neuron",
                neuron_id + 1,
            )

            column_is_zero = np.mean(np.abs(bad_h_matrix) < 1e-8, axis=0) > 0.5
            column_non_zero = ~column_is_zero
            ker_size = np.sum(column_non_zero) - np.linalg.matrix_rank(bad_h_matrix)
            if args.debug:
                print("The non zero part is", column_non_zero)
                print("Thr kernel size is", ker_size)

            recovered_weight_ratio = merged_partial_row.copy()
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
            print("We try to merge them!!!")

            merge_matrix = np.hstack(
                (
                    -np.array(bad_ker_space),
                    -np.array(bad_special_sol).reshape(-1, 1),
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

            weight_ratio = np.array(bad_special_sol).reshape(-1, 1)
            for k in range(ker_size):
                weight_ratio += (
                    merge_coefficient[k] / merge_coefficient[-1]
                ) * bad_ker_space[:, k].reshape(-1, 1)
            weight_ratio_multiply_k = weight_ratio * merge_coefficient[-1]

            for l in range(len(possible_weight_flag)):
                if not possible_weight_flag[l]:
                    if not np.isnan(weight_ratio_multiply_k[l]):
                        if len(possible_weight[l]) == 0:
                            possible_weight[l].append([weight_ratio_multiply_k[l], 1])
                        else:
                            for m in range(len(possible_weight[l])):
                                if (
                                    abs(
                                        possible_weight[l][m][0]
                                        - weight_ratio_multiply_k[l]
                                    )
                                    < 1e-4
                                ):
                                    possible_weight[l][m][1] += 1
                                    break
                                else:
                                    if m == len(possible_weight[l]) - 1:
                                        possible_weight[l].append(
                                            [weight_ratio_multiply_k[l], 1]
                                        )

        for i in range(len(possible_weight_flag)):
            if not possible_weight_flag[i]:
                if len(possible_weight[i]) > 0:
                    print(possible_weight[i])
                    max_count = 0
                    max_index = 0
                    for j in range(len(possible_weight[i])):
                        if possible_weight[i][j][1] > max_count:
                            max_count = possible_weight[i][j][1]
                            max_index = j
                    if max_count > 1:
                        merged_partial_row[i] = possible_weight[i][max_index][0]

        print(
            "\nAfter mering weak critical points, the new partial weight ratio is:",
            merged_partial_row,
        )

    return merged_component, merged_partial_row, merged_indices
