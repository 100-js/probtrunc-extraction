import numpy as np
from tqdm import tqdm
from signature_recovery.critical_point_search import (
    sweep_for_critical_points,
    do_better_sweep,
)
from signature_recovery.utils import (
    forward,
    matmul,
    get_hidden_layers,
    getAllWeightsAndBiases,
    on_which_hidden_layer,
    get_query_counts,
    check_activation,
    compute_weight_difference,
    predict_manual_fast,
    AcceptableFailure,
    GatherMoreData,
)
from .sign_recovery import recoverSign


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


def get_activation_pattern(args, model, point):
    weights_all, biases_all = getAllWeightsAndBiases(model)

    output = np.matmul(point, weights_all[0]) + biases_all[0]
    output = output * (output > 0)
    print("The active neurons in layer 1 are", np.where(output != 0)[0])

    for i in range(1, len(weights_all) - 1):
        output = np.matmul(output, weights_all[i]) + biases_all[i]
        output = output * (output > 0)
        print("The active neurons in layer", i + 1, "are", np.where(output != 0)[0])


def on_which_recovered_layer(args, point, recovered_weights, recovered_biases):
    x = point.copy()
    for i in range(len(recovered_weights)):
        x = np.matmul(x, recovered_weights[i]) + recovered_biases[i]
        for j in range(len(x)):
            if abs(x[j]) < 1e-5 and recovered_biases[i][j] != 0:
                # from a really recovered neuron
                return i
        x = x * (x > 0)

    return np.inf


def detect_activation_change(
    args,
    model,
    prev_point,
    point,
    target_layer,
    recovered_weights,
    recovered_biases,
    target_weight,
    target_bias,
):
    # This function is used to check whether the prev_point and point have the same activation pattern on the target layer
    # this works since we have recovered signatures of the target layer
    # we can check whether they have the same activation pattern without the knowledge of signs
    prev_output = forward(prev_point, recovered_weights, recovered_biases, True)
    curr_output = forward(point, recovered_weights, recovered_biases, True)

    prev_target_output = np.matmul(prev_output, target_weight) + target_bias
    curr_target_output = np.matmul(curr_output, target_weight) + target_bias

    prev_sign = np.sign(prev_target_output)
    curr_sign = np.sign(curr_target_output)

    for i in range(len(prev_sign)):
        if prev_sign[i] == 0:
            prev_sign[i] = -1
        if curr_sign[i] == 0:
            curr_sign[i] = -1

    if np.array_equal(prev_sign, curr_sign) == False:
        print(prev_sign)
        print(curr_sign)
    return np.array_equal(prev_sign, curr_sign)


def detect_dead_weight_activation(
    args,
    model,
    point,
    recovered_weights,
    recovered_biases,
    target_weight,
    target_bias,
):
    # return False
    weights_all, biases_all = getAllWeightsAndBiases(model)
    # This function is used to check whether some dead weights are activated by a point
    x = point.copy()

    # The first layer must be fully recovered
    x = np.matmul(x, recovered_weights[0]) + recovered_biases[0]
    x = x * (x > 0)

    for i in range(1, len(recovered_weights)):
        prev_active_index = np.where(x != 0)[0]
        recovered_weight_matrix = np.array(recovered_weights[i])

        x = np.matmul(x, recovered_weights[i]) + recovered_biases[i]
        x = x * (x > 0)
        post_active_index = np.where(x != 0)[0]

        for j in range(recovered_weight_matrix.shape[1]):
            weight_array = recovered_weight_matrix[:, j].flatten()
            if np.any(weight_array[prev_active_index] == 0) and j in post_active_index:
                # This neuron is activated by the point
                return True

    # For the target layer
    if args.dataset == "cifar10" and args.real_attack == 1:
        prev_active_index = np.where(x != 0)[0]
        target_weight = np.array(target_weight)
        for j in range(target_weight.shape[1]):
            weight_array = target_weight[:, j].flatten()
            if np.any(weight_array[prev_active_index] == 0) and target_bias[j] != 0:
                print("prev_active_index", np.where(x != 0)[0], "j", j)
                # This neuron is activated by the point
                return True

    return False


def check_all_activation_patterns(args, model, points, target_layer):
    weights_all, biases_all = getAllWeightsAndBiases(model)
    recovered_weights = weights_all[:target_layer]
    recovered_biases = biases_all[:target_layer]
    target_weight = weights_all[target_layer]
    target_bias = biases_all[target_layer]

    sign = []
    for index, point in enumerate(points):
        output = forward(point, recovered_weights, recovered_biases, True)
        target_output = np.matmul(output, target_weight) + target_bias

        curr_sign = np.sign(target_output)
        for i in range(len(curr_sign)):
            if curr_sign[i] == 0:
                curr_sign[i] = -1

        if len(sign) == 0:
            sign = curr_sign
        else:
            # compare sign and curr_sign
            if not np.array_equal(sign, curr_sign):
                print("Failed at index", index)
                return False
    print("The common activation is", sign)
    return True


def get_mergeable_critical_points(
    args,
    model,
    start_point,
    target_layer,
    recovered_weights,
    recovered_biases,
    target_weight,
    target_bias,
    dead_neurons,
):
    """
    This function identifies mergeable critical points that have the same
    activation pattern on deeper layers.
    They can be stacked together to increase the rank of the SOE matrix.

    This is the advanced version with efficiency consideration.
    Given a start point with many inactive neurons with high-confidence,
    we find many mergeable critical points.
    """
    dimOfLayer = model.layers[target_layer].output_shape[-1]
    shape = model.input_shape[1:]
    # 50 for small, 20 for large
    direction_number = 50

    # Do the sphere search for mergeable critical points based on the start point
    print("Find mergeable critical points based on the longest subarray")
    mergeable_critical_points = []
    mergeable_critical_points.append(start_point)

    # Use the start_point as the center of the sphere
    # then search for more mergeable critical points
    for dir_index in range(direction_number):
        # Generate a random direction
        random_direction = np.random.normal(0, 1, size=shape).flatten()

        critical_points = do_better_sweep(
            args,
            model,
            offset=start_point,
            direction=random_direction,
            low=-0,
            high=1e1,
            dataset=None,
        )
        prev_point = start_point

        # print("Searching in direction", dir_index + 1)
        # print("Number of critical points found:", len(critical_points))

        for point in critical_points:
            layer_id = on_which_recovered_layer(
                args, point, recovered_weights, recovered_biases
            )
            # print("layer id of this point is", layer_id)
            if 0 <= layer_id < target_layer:
                new_point = (prev_point + point) / 2
                activation_flag = detect_activation_change(
                    args,
                    model,
                    # prev_point,
                    mergeable_critical_points[-1],
                    new_point,
                    target_layer,
                    recovered_weights,
                    recovered_biases,
                    target_weight,
                    target_bias,
                )
                weight_flag = detect_dead_weight_activation(
                    args,
                    model,
                    new_point,
                    recovered_weights,
                    recovered_biases,
                    target_weight,
                    target_bias,
                )
                # if activation_flag == True and weight_flag == False:
                #     mergeable_critical_points.append(new_point)
                #     prev_point = point
                if activation_flag == False:
                    print("find a bad new point")
                    break
                elif weight_flag == False:
                    mergeable_critical_points.append(new_point)
                    prev_point = point
                elif weight_flag == True:
                    print("find a bad new point activating dead weights")
                    prev_point = point
            else:
                break

    print("*****************[Step 2] Result************************")
    print("The activation pattern for points on this search line is:")
    common_activation = check_all_activation_patterns(
        args, model, mergeable_critical_points, target_layer
    )
    if common_activation:
        print("All mergeable_critical_points share the same activation pattern")
    else:
        print("mergeable_critical_points are not good")

    print(
        "We get",
        len(mergeable_critical_points),
        "mergeable critical points for target layer",
    )
    rank_Y, matrix_Y, vector_Z = SOE_for_signs_merged(
        args,
        model,
        mergeable_critical_points,
        target_layer,
        recovered_weights,
        recovered_biases,
        target_weight,
        target_bias,
        dead_neurons,
    )

    return mergeable_critical_points, matrix_Y, vector_Z


def find_mergeable_critical_points(
    args,
    model,
    target_layer,
    recovered_weights,
    recovered_biases,
    target_weight,
    target_bias,
    dead_neurons,
):
    """
    This function identifies mergeable critical points that have the same
    activation pattern on deeper layers.
    They can be stacked together to increase the rank of the SOE matrix.

    This is the simple version without efficiency consideration.
    It works well for small models
    """
    # 50 for small, 10/20/30 for large
    total_search_line_num = 50
    # 50 for small, 20 for large
    direction_number = 50
    dimOfLayer = model.layers[target_layer].output_shape[-1]

    rank_Ys = []
    final_mergeable_critical_points = []
    final_matrix_Y = []
    final_vector_Z = []

    if args.real_attack == 3:
        # For sign recovery testing only
        final_mergeable_critical_points_list = []
        final_matrix_Y_list = []
        final_vector_Z_list = []
    # Step 1. Find the longest part of the random search line
    print(
        "Find the longest part of the line for the target layer",
        target_layer + 1,
        "based on",
        total_search_line_num,
        "random lines",
    )
    for line_number in tqdm(range(total_search_line_num)):
        print("*****************************************")
        print(f"Processing line {line_number + 1} of {total_search_line_num}")
        max_subarray_length = 0
        max_subarray_points = []
        critical_points = do_better_sweep(args, model, low=-1e3, high=1e3, dataset=None)

        layers = []
        for point in critical_points:
            # This function cheats on the previous layer
            layer_id_cheat, neuron_id = on_which_hidden_layer(point, model)

            layer_id = on_which_recovered_layer(
                args, point, recovered_weights, recovered_biases
            )
            # print("layer ids are", layer_id, layer_id_cheat)
            layers.append(layer_id)

        sub_points, sub_points_index = find_subarrays_less_than_target(
            layers, target_layer
        )
        max_subarray = max(sub_points, key=len)
        max_subarray_index = sub_points_index[sub_points.index(max_subarray)]

        max_subarray_length = len(max_subarray)
        if max_subarray_index[0] - 1 >= 0:
            # max_subarray_points.append((critical_points[max_subarray_index[0] - 1] + critical_points[max_subarray_index[0]]) / 2)
            max_subarray_points.append(critical_points[max_subarray_index[0] - 1])
        for i in max_subarray_index:
            max_subarray_points.append(critical_points[i])
        if max_subarray_index[-1] + 1 < len(critical_points):
            # max_subarray_points.append((critical_points[max_subarray_index[-1] + 1] + critical_points[max_subarray_index[-1]]) / 2)
            max_subarray_points.append(critical_points[max_subarray_index[-1] + 1])
        print("max_subarray_index", max_subarray_index)
        print("max_subarray is", max_subarray)
        print("len of max_subarray_points", len(max_subarray_points))

        # For large networks, 256*8 
        # rank_of_array = get_rank_of_point(
        #     max_subarray_points[0], recovered_weights, recovered_biases
        # )
        # if rank_of_array < 78:
        #     print("The rank of the first point is too small, skipping this line")
        #     continue

        print("*****************[Step 1] Result************************")
        print(
            "For layer",
            target_layer + 1,
            "The longest subarray length for each layer is",
        )
        print(max_subarray_length)

        # Step 2. Do the sphere search for mergeable critical points based on the longest subarray
        print("Find mergeable critical points based on the longest subarray")
        mergeable_critical_points = []
        shape = model.input_shape[1:]

        critical_points_good = True
        for point_index in range(len(max_subarray_points) - 1):
            mid_point = (
                max_subarray_points[point_index] + max_subarray_points[point_index + 1]
            ) / 2
            # if point_index > 0:
            if len(mergeable_critical_points) > 0:
                # For other points, check both activation change and dead weight activation
                if (
                    detect_activation_change(
                        args,
                        model,
                        mergeable_critical_points[-1],
                        mid_point,
                        target_layer,
                        recovered_weights,
                        recovered_biases,
                        target_weight,
                        target_bias,
                    )
                    == False
                ):
                    print("Find a bad critical point")
                    critical_points_good = False
                    break

                if (
                    detect_dead_weight_activation(
                        args,
                        model,
                        mid_point,
                        recovered_weights,
                        recovered_biases,
                        target_weight,
                        target_bias,
                    )
                    == True
                ):
                    print("Find a bad critical point activating dead weights")
                    continue
            else:
                # For the first point, only check dead weight activation
                if (
                    detect_dead_weight_activation(
                        args,
                        model,
                        mid_point,
                        recovered_weights,
                        recovered_biases,
                        target_weight,
                        target_bias,
                    )
                    == True
                ):
                    print("Find a bad critical point activating dead weights")
                    continue
            mergeable_critical_points.append(mid_point)
            # This position

        if critical_points_good == False or len(mergeable_critical_points) == 0:
            continue

        print("The start length is", len(mergeable_critical_points))
        # Do the sphere search for more mergeable critical points
        first_point = max_subarray_points[0]
        last_point = max_subarray_points[-1]
        start_point = (first_point + last_point) / 2

        # Use the start_point as the center of the sphere
        # then search for more mergeable critical points
        for dir_index in range(direction_number):
            # Generate a random direction
            random_direction = np.random.normal(0, 1, size=shape).flatten()

            critical_points = do_better_sweep(
                args,
                model,
                offset=start_point,
                direction=random_direction,
                low=0,
                high=1e1,
                dataset=None,
            )
            prev_point = start_point

            for point in critical_points:
                layer_id = on_which_recovered_layer(
                    args, point, recovered_weights, recovered_biases
                )
                if 0 <= layer_id < target_layer:
                    new_point = (prev_point + point) / 2
                    activation_flag = detect_activation_change(
                        args,
                        model,
                        # prev_point,
                        mergeable_critical_points[-1],
                        new_point,
                        target_layer,
                        recovered_weights,
                        recovered_biases,
                        target_weight,
                        target_bias,
                    )
                    weight_flag = detect_dead_weight_activation(
                        args,
                        model,
                        new_point,
                        recovered_weights,
                        recovered_biases,
                        target_weight,
                        target_bias,
                    )
                    # if activation_flag == True and weight_flag == False:
                    #     mergeable_critical_points.append(new_point)
                    #     prev_point = point
                    if activation_flag == False:
                        print("find a bad new point")
                        break
                    elif weight_flag == False:
                        mergeable_critical_points.append(new_point)
                        prev_point = point
                    elif weight_flag == True:
                        print("find a bad new point activating dead weights")
                        prev_point = point
                else:
                    break

        print("*****************[Step 2] Result************************")
        print("The activation pattern for points on this search line is:")
        common_activation = check_all_activation_patterns(
            args, model, mergeable_critical_points, target_layer
        )
        if common_activation:
            print("All mergeable_critical_points share the same activation pattern")
        else:
            print("mergeable_critical_points are not good")

        rank_Y, matrix_Y, vector_Z = SOE_for_signs_merged(
            args,
            model,
            mergeable_critical_points,
            target_layer,
            recovered_weights,
            recovered_biases,
            target_weight,
            target_bias,
            dead_neurons,
        )

        U, S, Vt = np.linalg.svd(matrix_Y)
        print("The singular values of matrix_Y are", S)

        if rank_Y == dimOfLayer - len(dead_neurons):
            print("[Final Result] The rank of the matrix Y is", rank_Y)
            soln, *rest = np.linalg.lstsq(
                np.array(matrix_Y, dtype=np.float64),  # float64 #float32
                np.array(vector_Z, dtype=np.float64),
                1e-5,
            )  # float64 #float32
            print("The soln from the SOE (Y, 1e-5) is", soln)

            # check whether all items in soln are larger than 1
            all_large = True
            for item in soln:
                if abs(item) < 1e-3:
                    all_large = False
            if all_large:
                print("All items in soln are larger than 1e-3, stopping")
                print("A bad line")
                continue

        if len(rank_Ys) == 0:
            final_mergeable_critical_points = mergeable_critical_points
            final_matrix_Y = matrix_Y
            final_vector_Z = vector_Z

            if args.real_attack == 3:
                final_mergeable_critical_points_list.append(mergeable_critical_points)
                final_matrix_Y_list.append(matrix_Y)
                final_vector_Z_list.append(vector_Z)
        else:
            if rank_Y > max(rank_Ys):
                final_mergeable_critical_points = mergeable_critical_points
                final_matrix_Y = matrix_Y
                final_vector_Z = vector_Z

                if args.real_attack == 3:
                    final_mergeable_critical_points_list = []
                    final_matrix_Y_list = []
                    final_vector_Z_list = []

                    final_mergeable_critical_points_list.append(
                        mergeable_critical_points
                    )
                    final_matrix_Y_list.append(matrix_Y)
                    final_vector_Z_list.append(vector_Z)
            elif rank_Y == max(rank_Ys) and args.real_attack == 3:
                final_mergeable_critical_points_list.append(mergeable_critical_points)
                final_matrix_Y_list.append(matrix_Y)
                final_vector_Z_list.append(vector_Z)

        print("[Final Result] The rank of the matrix Y is", rank_Y)
        # print("matrix_Y", matrix_Y)
        # print("vector_Z", vector_Z)
        rank_Ys.append(rank_Y)

        print("The target rank is", dimOfLayer - len(dead_neurons))
        if rank_Y == dimOfLayer - len(dead_neurons):
            break

    if len(rank_Ys) == 0:
        print("No mergeable critical points found")
        return [], [], []

    print("[Final Result] The largest rank of the matrix Y is", max(rank_Ys))

    if args.real_attack == 3:
        # For sign recovery testing only
        return (
            final_mergeable_critical_points_list,
            final_matrix_Y_list,
            final_vector_Z_list,
        )
    return final_mergeable_critical_points, final_matrix_Y, final_vector_Z


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
    print("The rank of the point is", rank)

    return rank


def SOE_for_signs_merged(
    args,
    model,
    continuous_input_points,
    target_layer,
    previous_weights,
    previous_biases,
    target_weights,
    target_biases,
    dead_neurons,
):
    print("The target layer is", target_layer + 1)

    weights_all, biases_all = getAllWeightsAndBiases(model)
    dimOfInput = weights_all[0].shape[0]
    dimOfLayer = weights_all[target_layer].shape[1]
    print("The dimension of the input is", dimOfInput)
    print("The dimension of the target layer is", dimOfLayer)

    # Construct the system of equations
    vector_Z = []
    matrix_Y = []

    for point_index, point in enumerate(continuous_input_points):
        print("\n***************************************")
        print("Processing point", point_index + 1, "of", len(continuous_input_points))
        get_activation_pattern(args, model, point)

        # Get the rank of this point
        point_rank = get_rank_of_point(point, previous_weights, previous_biases)
        if point_rank == 0:
            print("The point has rank 0, skipping")
            continue
        eps = 1e-5
        print("The rank of the point is", point_rank)

        for i in range(point_rank):
            # Select random directions and compute the output
            random_direction = np.sign(np.random.normal(0, 1, point.shape))

            output_point = forward(point, previous_weights, previous_biases, True)
            output_point = np.matmul(output_point, target_weights) + target_biases
            output_point = np.abs(output_point)
            final_output_point = predict_manual_fast(point, weights_all, biases_all)
            if args.dataset == "cifar10":
                final_output_point = final_output_point[args.output_index]

            point_eps = point + eps * random_direction
            output_point_eps = forward(
                point_eps, previous_weights, previous_biases, True
            )
            output_point_eps = (
                np.matmul(output_point_eps, target_weights) + target_biases
            )
            output_point_eps = np.abs(output_point_eps)
            final_output_point_eps = predict_manual_fast(
                point_eps, weights_all, biases_all
            )
            if args.dataset == "cifar10":
                final_output_point_eps = final_output_point_eps[args.output_index]

            output_delta = output_point_eps - output_point
            final_output_delta = final_output_point_eps - final_output_point
            output_delta = np.delete(output_delta, dead_neurons)

            # multiply by 1/eps, otherwise singular values are small 1e3
            output_delta *= 1e3
            final_output_delta *= 1e3

            # Check if the rank of matrix_Y increases
            new_matrix_Y = matrix_Y.copy()
            new_matrix_Y.append(output_delta)
            
            new_rank = np.linalg.matrix_rank(new_matrix_Y, tol=1e-5)
            # new_rank = np.linalg.matrix_rank(new_matrix_Y, tol=1e-4)
            # print("The new rank of the matrix Y is", new_rank)

            prev_rank = np.linalg.matrix_rank(matrix_Y, tol=1e-5)
            # prev_rank = np.linalg.matrix_rank(matrix_Y, tol=1e-4)
            # print("The previous rank of the matrix Y is", prev_rank)
            if new_rank > prev_rank or len(matrix_Y) < point_rank:
                matrix_Y.append(output_delta)
                vector_Z.append(final_output_delta)

        print("The shape of matrix_Y is", np.array(matrix_Y).shape)
        print(
            "The rank of the matrix Y is(default tol)", np.linalg.matrix_rank(matrix_Y)
        )
        print(
            "The rank of the matrix Y is(tol=1e-5)",
            np.linalg.matrix_rank(matrix_Y, tol=1e-5),
        )
        print(
            "The rank of the matrix Y is(tol=1e-4)",
            np.linalg.matrix_rank(matrix_Y, tol=1e-4),
        )

        if np.linalg.matrix_rank(matrix_Y, tol=1e-5) >= dimOfLayer - len(dead_neurons):
            break
        # if np.linalg.matrix_rank(matrix_Y, tol=1e-4) >= dimOfLayer - len(dead_neurons):
        #     break

    rank_Y = np.linalg.matrix_rank(matrix_Y, tol=1e-5)
    # rank_Y = np.linalg.matrix_rank(matrix_Y, tol=1e-4)
    print("The rank of the matrix Y is", rank_Y)

    # Solve the system of equations
    # print("Solving the system of equations")
    # soln, *rest = np.linalg.lstsq(
    #     np.array(matrix_Y, dtype=np.float64),  # float64 #float32
    #     np.array(vector_Z, dtype=np.float64),
    #     1e-5,
    # )  # float64 #float32
    # print("The soln from the SOE (Y, 1e-5) is", soln)
    # print("Residual is", vector_Z - np.matmul(np.array(matrix_Y), soln))

    return rank_Y, matrix_Y, vector_Z
