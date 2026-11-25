from whitebox import getWhiteboxRealSigns, getWhiteboxSignatures
from ..utils import (
    on_which_hidden_layer,
)
from ..follow_critical_hyperplane import is_on_following_layer


def visualize_filter_performance(args, model, dimOfInput, critical_points):
    # This function is used for the performance visualization

    cp_array = []
    for i in range(200):
        cp_array.append(next(critical_points))

    hidden_layer_num = model.layers.__len__() - 2
    neuron_num = [0] * hidden_layer_num
    for i in range(hidden_layer_num):
        neuron_num[i] = model.layers[i + 1].output_shape[1]

    for target in range(1, hidden_layer_num):
        print("Target layer:", target + 1)
        layer_count = [0] * hidden_layer_num
        cp_count_on_target_layer = [0] * neuron_num[target]
        layer_count_filter = [0] * hidden_layer_num
        cp_count_on_target_layer_filter = [0] * neuron_num[target]

        weights, biases = getWhiteboxSignatures(model, target + 1)
        A = weights[:-1]
        B = biases[:-1]

        args.layerID = target + 1
        for witness in cp_array:
            layer_id, neuron_id = on_which_hidden_layer(witness, model)
            layer_count[layer_id] += 1
            print("Layer ID:", layer_id + 1, "Neuron ID:", neuron_id + 1)

            if layer_id == args.layerID - 1:
                cp_count_on_target_layer[neuron_id] += 1
                print("[Cheat] This critical point is in the target layer")
            else:
                print("[Cheat] This critical point is not in the target layer")

            on_the_target_layer = is_on_following_layer(
                args,
                model,
                A[:-1],
                B[:-1],
                A[-1],
                B[-1],
                witness,
                dimOfInput,
            )
            print(on_the_target_layer)
            if on_the_target_layer:
                layer_count_filter[layer_id] += 1
                if layer_id == args.layerID - 1:
                    cp_count_on_target_layer_filter[neuron_id] += 1

        print("Analysis for target layer", target + 1)
        print("Layer count:\n", layer_count)
        print("CP count on target layer:\n", cp_count_on_target_layer)
        print("Layer count filter:\n", layer_count_filter)
        print("CP count on target layer filter:\n", cp_count_on_target_layer_filter)
