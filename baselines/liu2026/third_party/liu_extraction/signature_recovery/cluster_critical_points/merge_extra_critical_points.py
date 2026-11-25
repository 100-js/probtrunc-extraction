import copy
from .merge_good_critical_points import (
    merge_good_critical_points_into_a_specific_component,
)
from .merge_bad_critical_points import (
    merge_bad_critical_points_into_a_specific_component,
)


# Merge the critical points identified by following the hyperplane with the component
def merge_extra_cps_into_a_specific_component(
    args,
    model,
    partial_component,
    partial_row,
    identified_critical,
    this_layer_critical_points,
    good_indices_not_merged,
    bad_indices_not_merged,
    special_solutions,
    kernel_spaces,
    target_neuron,
):
    result_component = copy.deepcopy(partial_component)
    result_partial_row = copy.deepcopy(partial_row)
    result_critical = copy.deepcopy(identified_critical)

    while True:
        merged_num = 0

        # Merge with these critical points
        # 1. merge good critical points with the component
        merged_component, merged_partial_row, merged_indices = (
            merge_good_critical_points_into_a_specific_component(
                args,
                model,
                this_layer_critical_points,
                special_solutions,
                good_indices_not_merged,
                result_component,
                result_partial_row,
            )
        )

        print("\n**************************************************")
        print("For target neuron", target_neuron)
        print(
            "Before merging good critical points into the recovered component, we have:"
        )
        print(result_partial_row)
        print("length of witness is ", len(result_component))
        print(
            "After merging good critical points into the recovered component, we have:"
        )
        print(merged_partial_row)
        print("length of witness is ", len(merged_component))

        result_component = merged_component
        result_partial_row = merged_partial_row
        for idx in merged_indices:
            result_critical.append(
                this_layer_critical_points[good_indices_not_merged[idx]].point
            )
            merged_num += 1
        good_indices_not_merged = [
            item
            for i, item in enumerate(good_indices_not_merged)
            if i not in merged_indices
        ]

        print(
            "We still have",
            len(good_indices_not_merged),
            "good critical points not merged",
        )
        print(
            "We have",
            len(bad_indices_not_merged),
            "bad critical points left",
        )

        # 2. merge bad critical points with the component
        merged_component, merged_partial_row, merged_indices = (
            merge_bad_critical_points_into_a_specific_component(
                args,
                model,
                this_layer_critical_points,
                special_solutions,
                kernel_spaces,
                bad_indices_not_merged,
                result_component,
                result_partial_row,
            )
        )

        print("\n**************************************************")
        print("For target neuron", target_neuron)
        print(
            "Before merging bad critical points into the recovered component, we have:"
        )
        print(result_partial_row)
        print("length of witness is ", len(result_component))
        print(
            "After merging bad critical points into the recovered component, we have:"
        )
        print(merged_partial_row)
        print("length of witness is ", len(merged_component))

        result_component = merged_component
        result_partial_row = merged_partial_row

        for idx in merged_indices:
            result_critical.append(
                this_layer_critical_points[bad_indices_not_merged[idx]].point
            )
            merged_num += 1
        bad_indices_not_merged = [
            item
            for i, item in enumerate(bad_indices_not_merged)
            if i not in merged_indices
        ]

        print(
            "We still have",
            len(good_indices_not_merged),
            "good critical points not merged",
        )
        print(
            "We have",
            len(bad_indices_not_merged),
            "bad critical points left",
        )

        if merged_num == 0:
            break

    return result_component, result_partial_row, result_critical
