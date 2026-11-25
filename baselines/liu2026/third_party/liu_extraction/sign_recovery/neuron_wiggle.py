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


def on_which_recovered_layer_and_neuron(
    args,
    point,
    recovered_weights,
    recovered_biases,
    target_weights,
    target_biases,
):
    x = point.copy()
    for i in range(len(recovered_weights)):
        x = np.matmul(x, recovered_weights[i]) + recovered_biases[i]
        for j in range(len(x)):
            if abs(x[j]) < 1e-5 and recovered_biases[i][j] != 0:
                # from a really recovered neuron
                return i, j
        x = x * (x > 0)

    x = matmul(x, target_weights, target_biases, np)
    for j in range(len(x)):
        if abs(x[j]) < 1e-5 and target_biases[j] != 0:
            # from a target neuron
            return len(recovered_weights), j

    return -1, -1


def neuron_wiggle_to_find_removable_columns(
    args,
    model,
    mergeable_critical_points,
    rank_Y,
    target_rank,
    target_weights,
    target_biases,
    target_layer,
    previous_weights,
    previous_biases,
    dead_neurons,
):
    # Use neuron wiggle to decide which column can be removed
    PARAM_SEARCH_AT_LOCATION = 1e2

    sign_confidence = [0] * target_weights.shape[1]
    guessed_sign_neuron_wiggle = [0] * target_weights.shape[1]
    remove_columns = []

    for neuron_id in range(target_weights.shape[1]):
        print("Processing neuron", neuron_id)
        target_weight = target_weights[:, neuron_id]
        target_bias = target_biases[neuron_id]
        print("The target weight is", target_weight)

        if neuron_id in dead_neurons:
            print("This neuron is dead, cannot be recovered")
            continue

        # Find 100 critical points for this neuron
        critical_points_yielder = sweep_for_critical_points(
            args, model, PARAM_SEARCH_AT_LOCATION, None
        )
        critical_points = []
        for point in critical_points_yielder:
            # This is the cheat function
            # layer_index, neuron_index = on_which_hidden_layer(point, model)

            layer_index, neuron_index = on_which_recovered_layer_and_neuron(
                args,
                point,
                previous_weights,
                previous_biases,
                target_weights,
                target_biases,
            )
            if layer_index == target_layer and neuron_index == neuron_id:
                critical_points.append(point)
            if len(critical_points) >= 100:
                break

        # print("The random sign is ", random_signs[neuron_id])
        pos_exp_num, sign = recoverSign(
            args,
            model,
            previous_weights,
            previous_biases,
            target_weight,
            target_layer,
            critical_points,
        )

        sign_confidence[neuron_id] = max(pos_exp_num / 100, 1 - pos_exp_num / 100)
        guessed_sign_neuron_wiggle[neuron_id] = sign
        print(
            "The sign confidence for neuron",
            neuron_id,
            "is",
            sign_confidence[neuron_id],
            "and the guessed sign is",
            guessed_sign_neuron_wiggle[neuron_id],
        )

        if len(mergeable_critical_points) == 0:
            print("No mergeable critical points, cannot decide removability")
            continue

        # Just for plotting, we use the neuron wiggle on all neurons
        # if args.layerID == 3 and len(remove_columns) >= len(target_weight) - rank_Y:
        #     continue

        if target_weights.shape[1] == 8:
            # We need a strict confidence level for small layers
            confidence_threshold = 0.75
        if target_weights.shape[1] == 16:
            confidence_threshold = 0.7

        if sign_confidence[neuron_id] >= confidence_threshold:
            removeable = True
            # Check whether this neuron can be removed
            print(
                "The sign confidence for neuron",
                neuron_id,
                "is",
                sign_confidence[neuron_id],
                "and the guessed sign is",
                guessed_sign_neuron_wiggle[neuron_id],
            )

            for point_index in range(0, 10):
                # The number 10 here is the number of experiments
                point = mergeable_critical_points[point_index]
                output = forward(point, previous_weights, previous_biases, True)
                output = np.matmul(output, target_weights) + target_biases
                output_at_neuron = (
                    output[neuron_id] * guessed_sign_neuron_wiggle[neuron_id]
                )
                if output_at_neuron > 0:
                    print("We are pretty sure that this neuron is active")
                    removeable = False
                    break
            if removeable:
                print("We are pretty sure that this neuron is inactive")
                remove_columns.append(neuron_id)
            print("The removable status is", removeable)
        else:
            print("The removable status is False")

        if len(remove_columns) >= target_rank - rank_Y:
            print(
                "We have enough columns to remove, stopping the search. The columns to remove are",
                remove_columns,
            )
            return remove_columns, guessed_sign_neuron_wiggle
        # break
    return [], guessed_sign_neuron_wiggle

    # Just for layer 3 plotting
    # return remove_columns, guessed_sign_neuron_wiggle


def neuron_wiggle_to_get_all_signs(
    args,
    model,
    target_weights,
    target_biases,
    target_layer,
    previous_weights,
    previous_biases,
    dead_neurons,
):
    # Use neuron wiggle to decide which column can be removed
    PARAM_SEARCH_AT_LOCATION = 1e2

    sign_confidence = [0] * target_weights.shape[1]
    guessed_sign_neuron_wiggle = [0] * target_weights.shape[1]

    # Just for 256*8
    critical_points_total_num = 500000
    dimOfInput = previous_weights[0].shape[0]
    # """
    # We first find 1,000,000 critical points and store them
    # Then for each neuron, we check which critical points correspond to it
    # This saves a lot of time for sweeping critical points
    print("Sweeping for 500,000 critical points for later use...")

    mm = np.memmap(
        "/home/haolin/workspace/results/model_extraction/attack_logs/sign_results_rank/cifar256/criticalPoints5e5.dat",
        dtype=np.float64,
        mode="w+",
        shape=(critical_points_total_num, dimOfInput),
    )

    critical_points_yielder = sweep_for_critical_points(
        args, model, PARAM_SEARCH_AT_LOCATION, None
    )
    critical_points = []
    point_counter = 0
    section_counter = 0
    for point_counter in tqdm(range(critical_points_total_num)):
        point = next(critical_points_yielder)
        critical_points.append(point)
        if point_counter % 10000 == 0 and point_counter > 0:
            print("Collected", point_counter, "critical points...")
            # Store intermediate results to avoid losing progress
            mm[section_counter : section_counter + len(critical_points)] = (
                critical_points
            )
            section_counter += len(critical_points)
            critical_points.clear()

    # Flush remaining points that didn't reach the next batch
    if len(critical_points) > 0:
        print("Flushing remaining", len(critical_points), "critical points...")
        mm[section_counter : section_counter + len(critical_points)] = critical_points
    # breakpoint()
    # """

    dead_neurons = []
    almost_dead_neurons = []
    # Read the stored critical points for later use
    print("Reading back the stored critical points...")
    mm = np.memmap(
        "/home/haolin/workspace/results/model_extraction/attack_logs/sign_results_rank/cifar256/criticalPoints5e5.dat",
        dtype=np.float64,
        mode="r",
        shape=(critical_points_total_num, dimOfInput),
    )

    critical_points_per_neuron = [[] for _ in range(target_weights.shape[1])]
    for point_index in tqdm(range(critical_points_total_num)):
        point = mm[point_index]
        layer_index, neuron_index = on_which_recovered_layer_and_neuron(
            args,
            point,
            previous_weights,
            previous_biases,
            target_weights,
            target_biases,
        )

        if layer_index == target_layer:
            if len(critical_points_per_neuron[neuron_index]) >= 100:
                continue
            critical_points_per_neuron[neuron_index].append(point)

    for neuron_id in tqdm(range(target_weights.shape[1])):
        if (
            len(critical_points_per_neuron[neuron_id]) > 0
            and len(critical_points_per_neuron[neuron_id]) < 5
        ):
            almost_dead_neurons.append(neuron_id)
    print("Almost dead neurons are", almost_dead_neurons)

    for neuron_id in tqdm(range(target_weights.shape[1])):
        print("Processing neuron", neuron_id)
        print(
            "Neuron",
            neuron_id,
            "has",
            len(critical_points_per_neuron[neuron_id]),
            "critical points.",
        )

        if len(critical_points_per_neuron[neuron_id]) == 0:
            print("No critical points found for this neuron, skipping...")
            dead_neurons.append(neuron_id)
            continue

        critical_point_num = len(critical_points_per_neuron[neuron_id])
        target_weight = target_weights[:, neuron_id]
        target_bias = target_biases[neuron_id]

        pos_exp_num, sign = recoverSign(
            args,
            model,
            previous_weights,
            previous_biases,
            target_weight,
            target_layer,
            critical_points_per_neuron[neuron_id],
        )

        sign_confidence[neuron_id] = max(
            pos_exp_num / critical_point_num, 1 - pos_exp_num / critical_point_num
        )
        guessed_sign_neuron_wiggle[neuron_id] = sign
        print(
            "The sign confidence for neuron",
            neuron_id,
            "is",
            sign_confidence[neuron_id],
            "and the guessed sign is",
            guessed_sign_neuron_wiggle[neuron_id],
        )

    # ############################
    # original code without pre-stored critical points
    #############################
    """ 
    for neuron_id in tqdm(range(target_weights.shape[1])):
        print("Processing neuron", neuron_id)

        if neuron_id in dead_neurons:
            print("This neuron is dead, cannot be recovered")
            continue

        target_weight = target_weights[:, neuron_id]
        target_bias = target_biases[neuron_id]

        # Find 100 critical points for this neuron
        critical_points_yielder = sweep_for_critical_points(
            args, model, PARAM_SEARCH_AT_LOCATION, None
        )
        critical_points = []
        for point in critical_points_yielder:
            # This is the cheat function
            # layer_index, neuron_index = on_which_hidden_layer(point, model)

            layer_index, neuron_index = on_which_recovered_layer_and_neuron(
                args,
                point,
                previous_weights,
                previous_biases,
                target_weights,
                target_biases,
            )
            if layer_index == target_layer and neuron_index == neuron_id:
                critical_points.append(point)
            if len(critical_points) >= 100:
                break

        # print("The random sign is ", random_signs[neuron_id])
        pos_exp_num, sign = recoverSign(
            args,
            model,
            previous_weights,
            previous_biases,
            target_weight,
            target_layer,
            critical_points,
        )

        sign_confidence[neuron_id] = max(pos_exp_num / 100, 1 - pos_exp_num / 100)
        guessed_sign_neuron_wiggle[neuron_id] = sign
        print(
            "The sign confidence for neuron",
            neuron_id,
            "is",
            sign_confidence[neuron_id],
            "and the guessed sign is",
            guessed_sign_neuron_wiggle[neuron_id],
        )
    """

    return sign_confidence, guessed_sign_neuron_wiggle, dead_neurons
