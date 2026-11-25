import numpy as np
import gc
from tqdm import tqdm
import os

from .utils import (
    get_hidden_layers,
    forward,
    forward_at,
    get_second_grad_unsigned,
    basis,
    getAllWeightsAndBiases,
    on_which_hidden_layer,
    compute_difference,
    compute_weight_difference,
    get_prev_ratio,
    AcceptableFailure,
)
from whitebox import getWhiteboxRealSigns, getWhiteboxSignatures


# ----------
# Helper functions for actual Signature Recovery
# ----------
def get_ratios(args, model, critical_points, dimInput, with_sign=True, eps=1e-5):
    """
    Compute the input weights to one neuron on the first layer.
    One of the core algorithms described in the paper.

    Given a set of critical point, compute the gradient for the first N directions.
    In practice N = range(DIM)

    Compute the second partial derivitive along each of the axes. This gives
    us the unsigned ratios corresponding to the ratio of the weights.

                      /
                  ^  /
                  | /
                  |/
             <----X---->  direction_1
                 /|
                / |
               /  V
              /  direction_2

    If we want to recover signs then we should also query on direction_1+direction_2
    And check to see if we get the correct solution.
    """
    weights, biases = getAllWeightsAndBiases(model)
    N = [range(dimInput)]
    ratios = []
    for j, point in enumerate(critical_points):
        ratio = []
        for i in N[j]:
            ratio.append(
                get_second_grad_unsigned(
                    args, point, basis(i, dimInput), weights, biases, eps, eps / 3
                )
            )

        # Row sign recovery (a_1,a_2,...,a_n) -> recovery of sign of each a_i
        if with_sign:
            both_ratio = []
            for i in N[j]:
                both_ratio.append(
                    get_second_grad_unsigned(
                        args,
                        point,
                        (basis(i, dimInput) + basis(N[j][0], dimInput)) / 2,
                        weights,
                        biases,
                        eps,
                        eps / 3,
                    )
                )

            signed_ratio = []
            for i in range(len(ratio)):
                # When we have at least one y value already we need to orient this one
                # so that they point the same way.
                # We are given |f(x+d1)| and |f(x+d2)|
                # Compute |f(x+d1+d2)|.
                # Then either
                # |f(x+d1+d2)| = |f(x+d1)| + |f(x+d2)|
                # or
                # |f(x+d1+d2)| = |f(x+d1)| - |f(x+d2)|
                # or
                # |f(x+d1+d2)| = |f(x+d2)| - |f(x+d1)|
                positive_error = abs(abs(ratio[0] + ratio[i]) / 2 - abs(both_ratio[i]))
                negative_error = abs(abs(ratio[0] - ratio[i]) / 2 - abs(both_ratio[i]))

                if positive_error > 1e-2 and negative_error > 1e-2:
                    print("[Petential Error] Probably something is borked")
                    print(
                        "d^2(e(i))+d^2(e(j)) != d^2(e(i)+e(j))",
                        positive_error,
                        negative_error,
                    )
                    raise AcceptableFailure()

                if positive_error < negative_error:
                    signed_ratio.append(ratio[i])
                else:
                    signed_ratio.append(-ratio[i])
        else:
            signed_ratio = ratio

        ratio = np.array(signed_ratio)

        # print(ratio)
        ratios.append(ratio)

    return ratios


input_diff = []
input_diff_relative = []


def get_ratios_lstsq(args, model, critical_points, A, B, eps=1e-5):
    """
    Do the same thing as get_ratios, but works when we can't directly control where we want to query.

    This means we can't directly choose orthogonal directions, and so we're going
    to just pick random ones and then use least-squares to do it
    """
    print("EPS", eps)
    weights, biases = getAllWeightsAndBiases(model)
    ratios = []
    h_matrices = []

    for i, point in enumerate(critical_points):
        # We're going to create a system of linear equations
        # d_matrix is going to hold the inputs,
        # ys is going to hold the resulting learned outputs
        d_matrix = []
        ys = []

        # Query on N+2 random points, so that we have redundency
        # for the least squares solution.
        upper_bound = np.sum(forward(point, A, B) != 0) + 2
        print("Upper bound for random directions is", upper_bound)

        for i in range(upper_bound):
            # for i in range(np.sum(forward(point, A, B) != 0) * 100):
            # 1. Choose a random direction
            d = np.sign(np.random.normal(0, 1, point.shape))
            d_matrix.append(d)

            # 2. See what the second partial derivitive at this value is
            ratio_val = get_second_grad_unsigned(
                args, point, d, weights, biases, eps, eps / 3
            )

            # 3. Get the sign correct
            if len(ys) > 0:
                # When we have at least one y value already we need to orient this one
                # so that they point the same way.
                # We are given |f(x+d1)| and |f(x+d2)|
                # Compute |f(x+d1+d2)|.
                # Then either
                # |f(x+d1+d2)| = |f(x+d1)| + |f(x+d2)|
                # or
                # |f(x+d1+d2)| = |f(x+d1)| - |f(x+d2)|
                # or
                # |f(x+d1+d2)| = |f(x+d2)| - |f(x+d1)|
                both_ratio_val = get_second_grad_unsigned(
                    args, point, (d + d_matrix[0]) / 2, weights, biases, eps, eps / 3
                )

                positive_error = abs(abs(ys[0] + ratio_val) / 2 - abs(both_ratio_val))
                negative_error = abs(abs(ys[0] - ratio_val) / 2 - abs(both_ratio_val))
                if positive_error > 1e-2 and negative_error > 1e-2:
                    print("Probably something is borked")
                    print(
                        "d^2(e(i))+d^2(e(j)) != d^2(e(i)+e(j))",
                        positive_error,
                        negative_error,
                    )
                    raise AcceptableFailure()

                if negative_error < positive_error:
                    ratio_val *= -1

            ys.append(ratio_val)

        # Now we need to compute the system of equations
        # We have to figure out what the vectors look like in hidden space,
        # so compute that precisely
        d_matrix = np.array(d_matrix)
        h_matrix = np.array(forward_at(point, A, B, d_matrix))

        if len(A) != 0:
            # Which dimensions do we lose?
            column_is_zero = np.mean(np.abs(h_matrix) < 1e-8, axis=0) > 0.5
            assert np.all((forward(point, A, B, with_relu=True) == 0) == column_is_zero)

            if args.real_attack == 1:
                # normalize the matrix h_matrix
                normalize_baseline = 0
                abs_sum = []
                for i in range(h_matrix.shape[0]):
                    abs_sum.append(np.sum(np.abs(h_matrix[i])))
                median_value = np.median(abs_sum)
                for i in range(len(abs_sum)):
                    if abs_sum[i] == median_value:
                        normalize_baseline = i
                        break
                print("The normalize baseline is", normalize_baseline)

                h_matrix_non_zero_first_row = []
                h_matrix_non_zero_column_index = []
                for i in range(len(h_matrix[normalize_baseline])):
                    if (
                        abs(h_matrix[normalize_baseline][i]) != 0
                        and h_matrix[normalize_baseline][i] is not None
                    ):
                        h_matrix_non_zero_first_row.append(h_matrix[normalize_baseline][i])
                        h_matrix_non_zero_column_index.append(i)
                print("The non-zero first row of h_matrix is ", h_matrix_non_zero_first_row)
                print(
                    "The non-zero column index of h_matrix is ",
                    h_matrix_non_zero_column_index,
                )

                if len(h_matrix_non_zero_first_row) == 1:
                    normalized_h_matrix_non_zero_first_row = np.array(
                        h_matrix_non_zero_first_row
                    )
                else:
                    normalized_h_matrix_non_zero_first_row = (
                        2
                        * (
                            h_matrix_non_zero_first_row
                            - np.min(h_matrix_non_zero_first_row)
                        )
                        / (
                            np.max(h_matrix_non_zero_first_row)
                            - np.min(h_matrix_non_zero_first_row)
                        )
                        - 1
                    )
                print(
                    "The normalized non-zero first row is ",
                    normalized_h_matrix_non_zero_first_row,
                )
                normalized_ratio_h_matrix_non_zero_first_row = (
                    normalized_h_matrix_non_zero_first_row / h_matrix_non_zero_first_row
                )
                print(
                    "The normalized ratio of the non-zero first row of h_matrix is ",
                    normalized_ratio_h_matrix_non_zero_first_row,
                )

                normalization_ratio = np.zeros(h_matrix.shape[1])
                for i in range(len(normalization_ratio)):
                    if i in h_matrix_non_zero_column_index:
                        normalization_ratio[i] = (
                            normalized_ratio_h_matrix_non_zero_first_row[
                                h_matrix_non_zero_column_index.index(i)
                            ]
                        )

                # multiply the whole matrix B by the normalized ratio
                normalized_h_matrix = h_matrix.copy()
                for i in range(normalized_h_matrix.shape[1]):
                    if i in h_matrix_non_zero_column_index:
                        normalized_h_matrix[:, i] = (
                            normalized_h_matrix[:, i]
                            * normalized_ratio_h_matrix_non_zero_first_row[
                                h_matrix_non_zero_column_index.index(i)
                            ]
                        )
                # print("The normalized matrix h_matrix is \n", normalized_h_matrix)

                # solve the linear equation with the normalized matrix h_matrix
                soln_normalized_h_matrix, *rest = np.linalg.lstsq(
                    np.array(normalized_h_matrix, dtype=np.float64),  # float64 #float32
                    np.array(ys, dtype=np.float64),
                    1e-5,
                )
                soln_normalized_h_matrix[column_is_zero] = np.nan
                soln = soln_normalized_h_matrix
                if args.debug:
                    print("rank of h_matrix is: ", np.linalg.matrix_rank(h_matrix))
                    print(
                        "The non-zero columns are: ",
                        h_matrix.shape[1] - np.sum(column_is_zero),
                    )

                # Just for comparison
                # print("-------------------------------")
                # print("The scaled matrix h_matrix is \n", normalized_h_matrix)
                # print("The scaled ys is \n", ys)
                # print("The solution with scaled h_matrix is \n", soln)

                for i in range(len(soln)):
                    if i in h_matrix_non_zero_column_index:
                        soln[i] = (
                            soln[i]
                            * normalized_ratio_h_matrix_non_zero_first_row[
                                h_matrix_non_zero_column_index.index(i)
                            ]
                        )
                # print("The final solution with scaled h_matrix is \n", soln)
                # breakpoint()
                # print("The solved partial signature is", soln)
                # print("residuals are", ys - h_matrix @ soln)
            if args.real_attack == 0:
                # solve the linear equation with the original matrix h_matrix
                soln, *rest = np.linalg.lstsq(
                    np.array(h_matrix, dtype=np.float64),  # float64 #float32
                    np.array(ys, dtype=np.float64),
                    1e-5,
                )  # float64 #float32
                for i in range(len(soln)):
                    if column_is_zero[i]:
                        soln[i] = np.nan

        else:
            # Which dimensions do we lose?
            if args.layerID > 1:
                column_is_zero = np.mean(np.abs(h_matrix) < 1e-8, axis=0) > 0.5
                assert np.all(
                    (forward(point, A, B, with_relu=True) == 0) == column_is_zero
                )

            # Solve the least squares problem and get the solution
            # This is equal to solving for the ratios of the weight vector
            soln, *rest = np.linalg.lstsq(
                np.array(h_matrix, dtype=np.float64),  # float64 #float32
                np.array(ys, dtype=np.float64),
                1e-5,
            )  # float64 #float32

            if args.debug:
                print("rank of h_matrix is: ", np.linalg.matrix_rank(h_matrix))
                print(
                    "The non-zero columns are: ",
                    h_matrix.shape[1] - np.sum(column_is_zero),
                )

        ratios.append(soln)
        h_matrices.append(h_matrix)

    return (
        ratios,
        h_matrices,
    )


def get_ratios_lstsq_with_filter(args, model, critical_points, A, B, eps=1e-5):
    """
    Do the same thing as get_ratios, but works when we can't directly control where we want to query.

    This means we can't directly choose orthogonal directions, and so we're going
    to just pick random ones and then use least-squares to do it
    """
    print("EPS", eps)
    weights, biases = getAllWeightsAndBiases(model)
    ratios = []
    h_matrices = []
    filtered = False

    for p_index, point in enumerate(critical_points):
        # We're going to create a system of linear equations
        # d_matrix is going to hold the inputs,
        # ys is going to hold the resulting learned outputs
        d_matrix = []
        ys = []

        # Query on N+2 random points, so that we have redundency
        # for the least squares solution.
        if args.layerID == 1 and args.real_attack == 0:
            upper_bound = len(point)
        else:
            upper_bound = np.sum(forward(point, A, B) != 0) + 2
        print("Upper bound for random directions is", upper_bound)

        layer_id, neuron_id = on_which_hidden_layer(point, model)
        print("This point is on layer", layer_id, "and neuron", neuron_id)

        min_outputs = []
        second_derivatives = []
        bad_quality_points_count = 0
        for i in range(upper_bound):
            # for i in range(np.sum(forward(point, A, B) != 0) * 100):
            # 1. Choose a random direction
            d = np.sign(np.random.normal(0, 1, point.shape))

            if args.layerID == 1 and args.real_attack == 1:
                # If we are on the first layer, we need to make sure that direction is related to the input
                for j in range(len(point)):
                    if point[j] == 0:
                        d[j] = 0
            if args.layerID == 1 and args.real_attack == 0:
                # set all elements to 0 except the i-th element
                d_zeros = np.zeros(point.shape, dtype=np.float64)
                d_zeros[i] = d[i]
                d = d_zeros

            # d_matrix.append(d)

            # 2. See what the second partial derivitive at this value is
            ratio_val = get_second_grad_unsigned(
                args, point, d, weights, biases, eps, eps / 3
            )

            # Check if the ratio_val is too small
            layer_id, neuron_id = on_which_hidden_layer(point, model)
            if layer_id == 0:
                # print(
                #     "We have identified a critical point on layer",
                #     layer_id + 1,
                #     "and neuron",
                #     neuron_id + 1,
                # )
                output = np.matmul(point, weights[0]) + biases[0]

                min_outputs.append(abs(output[neuron_id]))
            second_derivatives.append(abs(ratio_val))

            if args.dataset == "mnist":
                dimOfLayer = np.array(weights[0]).shape[1]
                if dimOfLayer == 8:
                    flat_threshold = 1e-4
                if dimOfLayer == 16:
                    flat_threshold = 1e-5
                flat_num = 2
            if args.dataset == "cifar10":
                flat_threshold = 1e-5
                flat_num = 1

            if abs(ratio_val) < flat_threshold:
                if layer_id == 0:
                    print("This bad critical point is on the first hidden layer")
                print(
                    "Ratio value is small",
                    ratio_val,
                    "at point ",
                    p_index,
                    "direction",
                    i,
                )
                real_output = np.matmul(point, weights[0]) + biases[0]
                print("The real output is", real_output[neuron_id])

                bad_quality_points_count += 1
                if bad_quality_points_count >= flat_num:
                    print(
                        "Too many bad quality directions, stopping the search for this critical point"
                    )
                    filtered = True
                    break
                # filtered = True
                # break
            else:
                d_matrix.append(d)

                # 3. Get the sign correct
                if len(ys) > 0:
                    # When we have at least one y value already we need to orient this one
                    # so that they point the same way.
                    # We are given |f(x+d1)| and |f(x+d2)|
                    # Compute |f(x+d1+d2)|.
                    # Then either
                    # |f(x+d1+d2)| = |f(x+d1)| + |f(x+d2)|
                    # or
                    # |f(x+d1+d2)| = |f(x+d1)| - |f(x+d2)|
                    # or
                    # |f(x+d1+d2)| = |f(x+d2)| - |f(x+d1)|
                    both_ratio_val = get_second_grad_unsigned(
                        args,
                        point,
                        (d + d_matrix[0]) / 2,
                        weights,
                        biases,
                        eps,
                        eps / 3,
                    )

                    positive_error = abs(
                        abs(ys[0] + ratio_val) / 2 - abs(both_ratio_val)
                    )
                    negative_error = abs(
                        abs(ys[0] - ratio_val) / 2 - abs(both_ratio_val)
                    )
                    if positive_error > 1e-2 and negative_error > 1e-2:
                        print("Probably something is borked")
                        print(
                            "d^2(e(i))+d^2(e(j)) != d^2(e(i)+e(j))",
                            positive_error,
                            negative_error,
                        )
                        raise AcceptableFailure()

                    if negative_error < positive_error:
                        ratio_val *= -1

                ys.append(ratio_val)

        if filtered:
            print(
                "Filtered out the point",
                p_index,
                "because the ratio value was too small",
            )
            continue

        # Now we need to compute the system of equations
        # We have to figure out what the vectors look like in hidden space,
        # so compute that precisely
        d_matrix = np.array(d_matrix)
        h_matrix = np.array(forward_at(point, A, B, d_matrix))
        print("Shape of d_matrix:", d_matrix.shape)

        if len(A) != 0:
            # Which dimensions do we lose?
            column_is_zero = np.mean(np.abs(h_matrix) < 1e-8, axis=0) > 0.5
            assert np.all((forward(point, A, B, with_relu=True) == 0) == column_is_zero)

            # normalize the matrix h_matrix
            normalize_baseline = 0
            abs_sum = []
            for i in range(h_matrix.shape[0]):
                abs_sum.append(np.sum(np.abs(h_matrix[i])))
            median_value = np.median(abs_sum)
            for i in range(len(abs_sum)):
                if abs_sum[i] == median_value:
                    normalize_baseline = i
                    break
            print("The normalize baseline is", normalize_baseline)

            h_matrix_non_zero_first_row = []
            h_matrix_non_zero_column_index = []
            for i in range(len(h_matrix[normalize_baseline])):
                if (
                    abs(h_matrix[normalize_baseline][i]) != 0
                    and h_matrix[normalize_baseline][i] is not None
                ):
                    h_matrix_non_zero_first_row.append(h_matrix[normalize_baseline][i])
                    h_matrix_non_zero_column_index.append(i)
            print("The non-zero first row of h_matrix is ", h_matrix_non_zero_first_row)
            print(
                "The non-zero column index of h_matrix is ",
                h_matrix_non_zero_column_index,
            )

            if len(h_matrix_non_zero_first_row) == 1:
                normalized_h_matrix_non_zero_first_row = np.array(
                    h_matrix_non_zero_first_row
                )
            else:
                normalized_h_matrix_non_zero_first_row = (
                    2
                    * (
                        h_matrix_non_zero_first_row
                        - np.min(h_matrix_non_zero_first_row)
                    )
                    / (
                        np.max(h_matrix_non_zero_first_row)
                        - np.min(h_matrix_non_zero_first_row)
                    )
                    - 1
                )
            print(
                "The normalized non-zero first row of B is ",
                normalized_h_matrix_non_zero_first_row,
            )
            normalized_ratio_h_matrix_non_zero_first_row = (
                normalized_h_matrix_non_zero_first_row / h_matrix_non_zero_first_row
            )
            print(
                "The normalized ratio of the non-zero first row of h_matrix is ",
                normalized_ratio_h_matrix_non_zero_first_row,
            )

            normalization_ratio = np.zeros(h_matrix.shape[1])
            for i in range(len(normalization_ratio)):
                if i in h_matrix_non_zero_column_index:
                    normalization_ratio[i] = (
                        normalized_ratio_h_matrix_non_zero_first_row[
                            h_matrix_non_zero_column_index.index(i)
                        ]
                    )

            # multiply the whole matrix B by the normalized ratio
            normalized_h_matrix = h_matrix.copy()
            for i in range(normalized_h_matrix.shape[1]):
                if i in h_matrix_non_zero_column_index:
                    normalized_h_matrix[:, i] = (
                        normalized_h_matrix[:, i]
                        * normalized_ratio_h_matrix_non_zero_first_row[
                            h_matrix_non_zero_column_index.index(i)
                        ]
                    )

            # print("The normalized matrix h_matrix is \n", normalized_h_matrix)

            # solve the linear equation with the normalized matrix h_matrix
            soln_normalized_h_matrix, *rest = np.linalg.lstsq(
                np.array(normalized_h_matrix, dtype=np.float64),  # float64 #float32
                np.array(ys, dtype=np.float64),
                1e-5,
            )
            soln_normalized_h_matrix[column_is_zero] = np.nan
            soln = soln_normalized_h_matrix
            if args.debug:
                print("rank of h_matrix is: ", np.linalg.matrix_rank(h_matrix))
                print(
                    "The non-zero columns are: ",
                    h_matrix.shape[1] - np.sum(column_is_zero),
                )

            for i in range(len(soln)):
                if i in h_matrix_non_zero_column_index:
                    soln[i] = (
                        soln[i]
                        * normalized_ratio_h_matrix_non_zero_first_row[
                            h_matrix_non_zero_column_index.index(i)
                        ]
                    )
            # print("The solved partial signature is", soln)
            # print("residuals are", ys - h_matrix @ soln)

        else:
            # Which dimensions do we lose?
            if args.layerID > 1:
                column_is_zero = np.mean(np.abs(h_matrix) < 1e-8, axis=0) > 0.5
                assert np.all(
                    (forward(point, A, B, with_relu=True) == 0) == column_is_zero
                )

            # Solve the least squares problem and get the solution
            # This is equal to solving for the ratios of the weight vector
            soln, *rest = np.linalg.lstsq(
                np.array(h_matrix, dtype=np.float64),  # float64 #float32
                np.array(ys, dtype=np.float64),
                1e-5,
            )  # float64 #float32

            if args.debug:
                print("rank of h_matrix is: ", np.linalg.matrix_rank(h_matrix))
                print(
                    "The non-zero columns are: ",
                    h_matrix.shape[1] - np.sum(column_is_zero),
                )

        ratios.append(soln)
        h_matrices.append(h_matrix)

    return (
        ratios,
        h_matrices,
        filtered,
    )


def gather_ratios(
    args,
    this_layer_critical_point_count,
    critical_points_yielder,
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
    eps=1e-6,
):
    """
    Calculate the ratio values/partial signatures that can be deduced from each critical point.
    But if the rank of h matrix is not enough, the obtained signature is not reliable.
    """
    if LAYER == 0:
        print("We are on the first layer")

    point_index = this_layer_critical_point_count
    new_this_layer_critical_points = []
    new_remaining_critical_points = []

    critical_point_imprecision = []
    partial_weights_imprecision = []
    bad_critical_point_indices = []

    # they are only for visualization
    # num_of_cps_on_target_layer = 0
    # num_of_cps_on_target_layer_full_rank = 0
    # num_of_cps_on_target_layer_small_rank = 0

    for index in tqdm(
        range(len(critical_points_yielder)), desc="Analyzing critical points"
    ):
        print("--------------------------------")
        print("Got a new critical point")
        point = critical_points_yielder[index]

        if LAYER > 0:
            # Check if the output of any hidden layer before relu is less than tolerance, if yes we move on
            # i.e., the output is very small, this means further computations might fail due to limitation in precision?
            if args.real_attack == 0:
                if any(
                    np.any(np.abs(x) < 1e-5) for x in get_hidden_layers(point, A, B)
                ):  # Maybe adjust to float64 to 1e-12?
                    print(
                        "This critical point is caused by a recovered neuron before the target layer"
                    )
                    continue

        if LAYER > 0 and np.sum(forward(point, A, B) != 0) <= 1:
            print("Not enough hidden values are active to get meaningful data")
            continue

        if not check_fn(point):
            # This critical point is caused by a recovered neuron on the target layer
            # print("Check function rejected it")
            print(
                "This critical point is caused by a recovered neuron on the target layer"
            )
            continue

        from .solve_noise_issue.filter_out_noise_critical_points import (
            is_on_target_layer,
        )
        from .signature_recovery import CriticalPoint

        filtered = False

        if len(model.layers) == 3:
            GRAD_EPS = 1e1
        else:
            GRAD_EPS = 1e-4  # 1e-6#1e-4
        for EPS in [GRAD_EPS, GRAD_EPS / 10, GRAD_EPS / 100]:
            try:
                if args.real_attack == 1 and args.layerID > 1:
                    ratios, h_matrices = get_ratios_lstsq(
                        args, model, [point], A, B, eps=EPS
                    )
                elif args.real_attack == 1 and args.layerID == 1:
                    # ratios, h_matrices, filtered = get_ratios_lstsq_with_filter(
                    #     args, model, [point], A, B, eps=EPS
                    # )
                    ratios, h_matrices = get_ratios_lstsq(
                        args, model, [point], A, B, eps=EPS
                    )
                else:
                    ratios, h_matrices = get_ratios_lstsq(
                        args, model, [point], A, B, eps=EPS
                    )

                if filtered:
                    print(
                        "Filtered out the critical point because the ratio value is too small"
                    )
                    break
                ratio = ratios[0].flatten()
                h_matrix = h_matrices[0]

                # just for sloving the imprecision issue
                layer_id, neuron_id = on_which_hidden_layer(point, model)
                print("*****************Precision Checking********************")
                weights_all, biases_all = getAllWeightsAndBiases(model)
                hidden_output = get_hidden_layers(
                    point,
                    weights_all,
                    biases_all,
                )
                print(
                    "[Precision Checkpoint - critical point] Point ID: ",
                    index,
                    "Layer ID: ",
                    layer_id,
                    "Neuron ID: ",
                    neuron_id,
                    "Output: ",
                    hidden_output[layer_id][neuron_id],
                )
                print("The point index is", point_index)

                if hidden_output[layer_id][neuron_id] > 1e-6:
                    print(
                        "[Precision Checkpoint - critical point] warning!!! the critical point is bad!!"
                    )

                if layer_id == args.layerID - 1:
                    print("The critical point is on the target layer")
                    critical_point_imprecision.append(
                        abs(hidden_output[layer_id][neuron_id])
                    )

                    real_weight = []
                    for i in range(len(weights_all[layer_id])):
                        real_weight.append(weights_all[layer_id][i][neuron_id])
                    ratio = np.array(ratio)
                    real_weight = np.array(real_weight)
                    extracted_ratio = ratio.copy()

                    previous_extracted_sequence, previous_alpha = get_prev_ratio(
                        args, model, biases_all
                    )
                    real_weight = real_weight[previous_extracted_sequence]
                    # print("The real weight is", real_weight)
                    for weight_index in range(len(extracted_ratio)):
                        extracted_ratio[weight_index] = (
                            extracted_ratio[weight_index] * previous_alpha[weight_index]
                        )
                    # print("previous_extracted_sequence is", previous_extracted_sequence)
                    # print("previous_alpha is", previous_alpha)
                    # print("The scaled recovered weight is", extracted_ratio)
                    # breakpoint()

                    # Which dimensions do we lose?
                    column_is_zero = np.mean(np.abs(h_matrix) < 1e-8, axis=0) > 0.5
                    if np.linalg.matrix_rank(h_matrix) >= h_matrix.shape[1] - np.sum(
                        column_is_zero
                    ):
                        print("With enough rank")
                        print("Compare the extracted ratio with the real weight")
                        diff = compute_weight_difference(extracted_ratio, real_weight)
                        partial_weights_imprecision.append(diff)
                        print("The difference is", diff)
                break
            except AcceptableFailure:
                print("Try again with smaller eps")
                continue
        try:
            if LAYER == 0:
                if filtered:
                    bad_critical_point_indices.append(index)
                    new_remaining_critical_points.append(
                        CriticalPoint(index, ratio, point, h_matrix)
                    )
                else:
                    new_this_layer_critical_points.append(
                        CriticalPoint(point_index, ratio, point, h_matrix)
                    )
                    point_index += 1
                continue

            # With filtering out noise critical points
            if not is_on_target_layer(
                args, model, A[:-1], B[:-1], A[-1], B[-1], point, dimInput
            ):
                # Filter out critical points that are not on the target layer
                new_remaining_critical_points.append(
                    CriticalPoint(index, ratio, point, h_matrix)
                )
            else:
                new_this_layer_critical_points.append(
                    CriticalPoint(point_index, ratio, point, h_matrix)
                )
                point_index += 1
        except:
            continue

    gc.collect()

    print("-----------------------------------------------")
    critical_point_imprecision = [
        value for value in critical_point_imprecision if value != 0
    ]
    if critical_point_imprecision:
        print(
            "[Precision Checkpoint] The largest imprecision of the critical points in the target layer is",
            np.max(critical_point_imprecision),
            "the smallest imprecision is",
            np.min(critical_point_imprecision),
            "and the average imprecision is",
            np.mean(critical_point_imprecision),
        )
    else:
        print("[Precision Checkpoint] No diagnostic target-layer labels were available")
    partial_weights_imprecision = [
        value for value in partial_weights_imprecision if value != 0
    ]
    if len(partial_weights_imprecision) != 0:
        print(
            "[Precision Checkpoint] The largest imprecision of the partial weights in the target layer is",
            np.max(partial_weights_imprecision),
            "the smallest imprecision is",
            np.min(partial_weights_imprecision),
            "and the average imprecision is",
            np.mean(partial_weights_imprecision),
        )

    global input_diff
    global input_diff_relative
    print("[Precision Checkpoint] The average input difference is", np.mean(input_diff))
    print(
        "[Precision Checkpoint] The average input relative difference is",
        np.mean(input_diff_relative),
    )

    # print("--------------------------------")
    # print("Nuber of critical points on the target layer:", num_of_cps_on_target_layer)
    # print(
    #     "Number of critical points on the target layer with full rank:",
    #     num_of_cps_on_target_layer_full_rank,
    # )
    # print(
    #     "Number of critical points on the target layer with small rank:",
    #     num_of_cps_on_target_layer_small_rank,
    # )
    # breakpoint()

    print("len of new_this_layer_critical_points", len(new_this_layer_critical_points))
    print("len of new_remaining_critical_points", len(new_remaining_critical_points))
    print("len of bad_critical_point_indices", len(bad_critical_point_indices))
    return (
        new_this_layer_critical_points,
        new_remaining_critical_points,
        bad_critical_point_indices,
    )


def gather_ratios_without_filter(
    args,
    this_layer_critical_point_count,
    critical_points_yielder,
    A,
    B,
    check_fn,
    LAYER,
    COUNT,
    model,
    eps=1e-6,
):
    """
    Calculate the ratio values/partial signatures that can be deduced from each critical point.
    But if the rank of h matrix is not enough, the obtained signature is not reliable.
    """
    point_index = this_layer_critical_point_count
    new_critical_points = []
    print("Gathering", COUNT, "critical points")

    for index, point in enumerate(critical_points_yielder):
        print("--------------------------------")
        print("Got a new critical point")
        if LAYER > 0:
            if args.real_attack == 0:
                # Check if the output of any hidden layer before relu is less than tolerance, if yes we move on
                # i.e., the output is very small, this means further computations might fail due to limitation in precision?
                if any(
                    np.any(np.abs(x) < 1e-5) for x in get_hidden_layers(point, A, B)
                ):  # Maybe adjust to float64 to 1e-12?
                    print(
                        "This critical point is caused by a recovered neuron before the target layer"
                    )
                    continue
            else:
                dead_neurons = []
                if args.real_attack == 1 and args.layerID >= 2:
                    # get the dead neurons of the previous layer
                    for i in range(1, args.layerID):
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

                        dead_neuron = []
                        for index, b in enumerate(
                            recovered_parameters["extracted_bias"]
                        ):
                            if abs(b) == 0:
                                dead_neuron.append(index)
                        dead_neurons.append(dead_neuron)
                print("The dead neurons from the previous layer are", dead_neurons)

                vs = get_hidden_layers(point, A, B)
                recovered = False
                for i in range(len(vs)):
                    x = vs[i]
                    for j in range(len(x)):
                        if np.abs(x[j]) < 1e-5 and j not in dead_neurons[i]:
                            print(
                                "This critical point is caused by a recovered neuron before the target layer"
                            )
                            recovered = True
                            break
                    if recovered:
                        break
                if recovered:
                    continue

        if LAYER > 0 and np.sum(forward(point, A, B) != 0) <= 1:
            print("Not enough hidden values are active to get meaningful data")
            continue

        if not check_fn(point):
            # This critical point is caused by a recovered neuron on the target layer
            # print("Check function rejected it")
            print(
                "This critical point is caused by a recovered neuron on the target layer"
            )
            continue

        from .signature_recovery import CriticalPoint

        if len(model.layers) == 3:
            GRAD_EPS = 1e1
        else:
            GRAD_EPS = 1e-4  # 1e-6#1e-4
        for EPS in [GRAD_EPS, GRAD_EPS / 10, GRAD_EPS / 100]:
            try:
                ratios, h_matrices = get_ratios_lstsq(
                    args, model, [point], A, B, eps=EPS
                )
                ratio = ratios[0].flatten()
                h_matrix = h_matrices[0]
                break
            except AcceptableFailure:
                print("Try again with smaller eps")
                continue
        try:
            new_critical_points.append(
                CriticalPoint(point_index, ratio, point, h_matrix)
            )
            point_index += 1
        except:
            continue

        # coupon collector: we need nlogn points.
        print("Up to", len(new_critical_points), "of", COUNT)
        if len(new_critical_points) >= COUNT:
            break
    gc.collect()

    return new_critical_points
