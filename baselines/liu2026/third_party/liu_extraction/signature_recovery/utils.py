import numpy as np
import jax.numpy as jnp
import random
import os
from global_vars import GlobalConfig

np.set_printoptions(precision=20)


# ==========
#  Possible failure modes defined
# ==========
class AcceptableFailure(Exception):
    """
    Sometimes things fail for entirely acceptable reasons (e.g., we haven't
    queried enough points to have seen all the hyperplanes, or we get stuck
    in a constant zero region). When that happens we throw an AcceptableFailure
    because life is tough but we should just back out and try again after
    making the appropriate correction.
    """

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class GatherMoreData(AcceptableFailure):
    """
    When gathering witnesses to hyperplanes, sometimes we don't have
    enough and need more witnesses to *this particular neuron*.
    This error says that we should gather more examples of that one.
    """

    def __init__(self, data, **kwargs):
        super(GatherMoreData, self).__init__(data=data, **kwargs)

    def __str__(self):
        return "GatherMoreData!!"


# ----------
# Most basic Helper functions
# ----------


def basis(i, N):
    """
    Standard basis vector along dimension i
    """
    a = np.zeros(N, dtype=np.float64)  # float64
    a[i] = 1
    return a


def matmul(a, b, c, np=np):
    if c is None:
        c = np.zeros(1)
    return np.dot(a, b) + c


def forward(x, A, B, with_relu=False, np=np):
    for i, (a, b) in enumerate(zip(A, B)):
        x = matmul(x, a, b, np)
        if (i < len(A) - 1) or with_relu:
            x = x * (x > 0)
        # print("forward pass layer", i, "output: ", x)
    return x


def forward_at(point, A, B, d_matrix):
    if len(A) == 0:
        return d_matrix

    mask_vectors = [layer > 0 for layer in get_hidden_layers(point, A, B)]
    # print("mask_vectors: ", mask_vectors)

    h_matrix = np.array(d_matrix)
    for i, (matrix, mask) in enumerate(zip(A, mask_vectors)):
        h_matrix = matmul(h_matrix, matrix, None) * mask

    return h_matrix


def get_hidden_layers(
    x, A, B, flat=False, np=np
):  # forward pass through all layers that have been found
    """
    Given the weights and biases up to a certain layer, compute a forward pass
    around the vicinity of an input x.
    Kind of like getLocalMatrixAndBias(weights, biases, x0) in blackbox.py
    Returns the output calculation from each layer
    """
    if len(A) == 0:
        return []
    region = []
    for i, (a, b) in enumerate(zip(A, B)):
        x = matmul(x, a, b, np)  # forward pass
        if np == jnp:
            region.append(x)
        else:
            region.append(np.copy(x))
        if i < len(A) - 1:  # relu
            x = x * (x > 0)
    if flat:
        region = np.concatenate(region, axis=0)
    return region


def get_hidden_at(A, B, known_A, known_B, LAYER, x, prior=True):
    """
    Get the hidden value for an input using the known transform and known A.
    This function IS NOT CHEATING.
    """
    if prior:
        which_activation = [y for x in get_hidden_layers(x, A, B) for y in x]
    else:
        which_activation = []
    which_activation += list(
        matmul(forward(x[np.newaxis, :], A, B, with_relu=True), known_A, known_B)[0]
    )
    return tuple(which_activation)


def get_polytope(x, A, B, flat=False):
    if len(A) == 0:
        return tuple()
    h = get_hidden_layers(x, A, B)  # ,flat
    h = np.concatenate(h, axis=0)
    return tuple(np.int32(np.sign(h)))


def get_polytope_at(A, B, known_A, known_B, x, prior=True):
    """
    Get the polytope for an input using the known transform and known A.
    This function IS NOT CHEATING.
    """
    if prior:
        which_polytope = get_polytope(x, A, B)
    else:
        which_polytope = tuple()
    hidden = forward(x[np.newaxis, :], A, B, with_relu=True)
    which_polytope += tuple(np.int32(np.sign(matmul(hidden, known_A, known_B)))[0])
    return which_polytope


def get_grad(x, direction, weights, biases, eps=1e-6):
    """
    Finite differences to estimate the gradient.
    Uses just two coordinates---that's sufficient for most of the code.

    Can fail if we're right at a critical point and we get the left and right side.
           /
          X
         /
    -X--/

    """
    x = x[np.newaxis, :]
    a = predict_manual_fast(x - eps * direction, weights, biases)
    b = predict_manual_fast(x, weights, biases)
    g1 = (b - a) / eps
    return g1


def get_second_grad_unsigned(args, x, direction, weights, biases, eps, eps2):
    """
    Compute the second derivative by computing the first derivative twice.
    """
    grad_value = get_grad(
        x + direction * eps, direction, weights, biases, eps2
    ) + get_grad(x - direction * eps, -direction, weights, biases, eps2)

    if args.dataset == "cifar10":
        return grad_value[0][args.output_index]

    return grad_value[0]


def getAllWeightsAndBiases(model):
    weights = []
    biases = []
    for l in model.layers:
        if len(l.get_weights()) > 0:
            w, b = l.get_weights()
            weights.append(np.copy(w))
            biases.append(np.copy(b))
    return weights, biases


def predict_manual(x, model):
    """Manual prediction"""
    weights, biases = getAllWeightsAndBiases(model)
    for i in range(len(weights)):
        x = np.matmul(x, weights[i]) + biases[i]
        if i < len(weights) - 1:
            x = x * (x > 0)
    return x


def predict_manual_fast(x, weights, biases):
    """Manual prediction faster"""
    GlobalConfig.query_count += x.shape[0]  # 1

    orig_x = x
    for i in range(len(weights)):
        x = matmul(x, weights[i], biases[i])  # np.matmul(x, weights[i]) + biases[i]
        if i < len(weights) - 1:
            x = x * (x > 0)
        # print("predict_manual_fast layer", i, "output: ", x)
    if GlobalConfig.set_save_queries:
        GlobalConfig.SAVED_QUERIES.extend(zip(orig_x, x))
    return x


def get_saved_queries(weights, biases, inputShape):
    print("len saved queries: ", len(GlobalConfig.SAVED_QUERIES))
    if len(GlobalConfig.SAVED_QUERIES) < 10:
        points = []
        for i in range(100):
            # inputShape = model.input_shape[1:]
            offset = np.random.normal(0, 1, size=inputShape).flatten()
            direction = np.random.normal(0, 1, size=inputShape).flatten()
            x = random.uniform(-1000, 1000)
            points.append((offset + direction * x))
        points = np.array(points)
        predict_manual_fast(points, weights, biases)
    return GlobalConfig.SAVED_QUERIES


def get_query_counts():
    print("Query count: ", GlobalConfig.query_count)
    return GlobalConfig.query_count, GlobalConfig.crit_query_count


def softmax(x):
    """Compute the softmax of vector x."""
    exp_x = np.exp(x - np.max(x))  # Subtracting np.max(x) for numerical stability
    return exp_x / exp_x.sum()


# This function is used to filter Not Relevant Critical Points
# It is used in the function do_better_sweep() in critical_point_search.py
def on_which_hidden_layer(point, model):
    weights, biases = getAllWeightsAndBiases(model)

    x = point
    y = point
    for i in range(0, len(weights)):
        x = np.matmul(x, weights[i]) + biases[i]
        # print("output of layer: ", x)
        for j in range(len(x)):
            if abs(x[j]) < 1e-6:
                # print("output of first layer: ", y)
                # for k in range(0, i - 1):
                #     y = np.matmul(y, weights[k]) + biases[k]
                #     if k < len(weights) - 1:
                #         y = y * (y > 0)
                #     print("Layer ", k + 1, "Active Neurons: ", np.sum(y > 0))
                # print("output of last layer: ", y)
                return i, j
        if i < len(weights) - 1:
            x = x * (x > 0)

    return -1, -1


def check_multiple_intersection_point(point, model, layerID):
    weights, biases = getAllWeightsAndBiases(model)

    target_layer_status = False
    previous_layer_status = False
    previous_layer_id = -1

    x = point
    for i in range(0, layerID):
        x = np.matmul(x, weights[i]) + biases[i]
        print(
            "[check_multiple_intersection_point] Layer",
            i + 1,
            "output (without ReLU): ",
            x,
        )
        for j in range(len(x)):
            if abs(x[j]) < 1e-4:
                if i == layerID - 1:
                    target_layer_status = True
                else:
                    previous_layer_status = True
                    previous_layer_id = i
        if i < len(weights) - 1:
            x = x * (x > 0)

    if target_layer_status and previous_layer_status:
        return True, previous_layer_id
    return False, -1


def compute_difference(value1, value2):
    # The values in the two lists maybe nan
    # The values in the two lists at the same index have the same factor

    # The difference is computed as the sum of the absolute difference of the values at the same index
    factor = 0.0
    diff = 0.0
    sequence = 0

    for i in range(len(value1)):
        if np.isnan(value1[i]) or np.isnan(value2[i]):
            continue
        if value1[i] == 0 or value2[i] == 0:
            continue

        if factor == 0:
            if abs(value1[i]) > abs(value2[i]):
                factor = value2[i] / value1[i]
                sequence = 1
            else:
                factor = value1[i] / value2[i]
                sequence = 2

            # print("value1: ", value1[:10])
            # print("value2: ", value2[:10])
            # print("factor: ", factor)
            # if sequence == 1:
            #     print(factor * value1[:10])
            # else:
            #     print(factor * value2[:10])
        else:
            if sequence == 1:
                diff += abs(value2[i] - factor * value1[i])
            if sequence == 2:
                diff += abs(value1[i] - factor * value2[i])

    return diff, 0, factor, sequence


def compute_weight_difference(extracted_weight, real_weight):
    # The values in the two lists maybe nan
    # The values in the two lists at the same index have the same factor

    # The difference is computed as the sum of the absolute difference of the values at the same index
    factor = 0.0
    diff = 0.0
    difference = []
    relative_diff = 0.0
    relative_difference = []

    for i in range(len(extracted_weight)):
        if np.isnan(extracted_weight[i]) or np.isnan(real_weight[i]):
            continue
        if extracted_weight[i] == 0 or real_weight[i] == 0:
            continue

        if factor == 0:
            factor = real_weight[i] / extracted_weight[i]
        diff += abs(real_weight[i] - factor * extracted_weight[i])
        difference.append(abs(real_weight[i] - factor * extracted_weight[i]))
        relative_diff += abs(real_weight[i] - factor * extracted_weight[i]) / abs(
            real_weight[i]
        )
        relative_difference.append(
            abs(real_weight[i] - factor * extracted_weight[i]) / abs(real_weight[i])
        )
    # only for the imprecision issue
    # print("[Precision Checkpoin - signature] The real weight is:", real_weight[:10])
    # print(
    #     "[Precision Checkpoint - signature] The extracted weight is: ",
    #     extracted_weight[:10],
    # )
    # print(
    #     "[Precision Checkpoint - signature] The scaling extracted weight is: ",
    #     extracted_weight[:10] * factor,
    # )
    if abs(diff) > 1e-3:
        print("this point is not good")

    non_nan_count = 0
    for i in range(len(extracted_weight)):
        if np.isnan(extracted_weight[i]):
            continue
        non_nan_count += 1
    print(
        "[Precision Checkpoint - signature] The average difference is: ",
        diff / non_nan_count,
    )
    print(
        "[Precision Checkpoint - signature] The average relative difference is: ",
        relative_diff / non_nan_count,
    )

    print("The factor is: ", factor)
    print("[Precision Checkpoint - signature] The sum difference is: ", diff)
    print("[Precision Checkpoint - signature] The difference is: ", difference)
    print(
        "[Precision Checkpoint - signature] The relative difference is: ",
        relative_difference,
    )

    return diff


def check_activation(args, point, A, B, dead_neurons):
    # Check the activation of the neurons in the model
    #  when we do the end-to-end attack, some weights cannot be recovered
    # when we attack the following layers, these neurons must be inactive
    if len(A) == 0:
        # print("This is the first layer. No layers found in the model.")
        return True

    output_extracted = np.matmul(point, A[0]) + B[0]
    output_extracted = output_extracted * (output_extracted > 0)

    for layer in range(1, args.layerID - 1):
        for j in range(A[layer].shape[1]):
            if j in dead_neurons[layer]:
                continue
            input_extracted = output_extracted.flatten()
            weight_extarcted = A[layer][:, j].flatten()
            for i in range(len(input_extracted)):
                if input_extracted[i] > 0 and weight_extarcted[i] == 0:
                    print("This critical point is not valid")
                    print("input_extracted", input_extracted)
                    print("weight_extarcted", weight_extarcted)
                    return False
        output_extracted = np.matmul(output_extracted, A[layer]) + B[layer]
        output_extracted = output_extracted * (output_extracted > 0)
    return True


def get_prev_ratio(args, model, biases):
    if args.layerID == 1:
        dimOfInput = model.input_shape[1]
        previous_extracted_sequence = np.arange(dimOfInput)
        previous_alpha = np.ones(dimOfInput)
        return previous_extracted_sequence, previous_alpha

    if args.real_attack == 0:
        dimOfLayer = model.layers[args.layerID - 1].output_shape[1]
        previous_extracted_sequence = np.arange(dimOfLayer)
        previous_alpha = np.ones(dimOfLayer)
        return previous_extracted_sequence, previous_alpha

    if args.layerID == 2:
        previous_recoveryPath = os.path.join(
            args.output_path,
            "recovered results",
            f"Seed{args.seed}-Count500-RealAttack{args.real_attack}",
            # f"Seed{args.seed}-Count1000-RealAttack{args.real_attack}",
        )

    if args.layerID > 2:
        previous_recoveryPath = os.path.join(
            args.output_path,
            "recovered results",
            f"Seed{args.seed}-Count{args.Count}-RealAttack{args.real_attack}",
        )

    previous_recovered_parameters = np.load(
        os.path.join(previous_recoveryPath, f"recovered_{args.layerID-1}.npz")
    )
    previous_extracted_sequence = previous_recovered_parameters["extracted_sequence"]
    previous_extracted_bias = previous_recovered_parameters["extracted_bias"]

    previous_bias_real = biases[args.layerID - 2]
    previous_bias_real = np.array(previous_bias_real)
    previous_bias_real = previous_bias_real[previous_extracted_sequence]
    previous_alpha = previous_extracted_bias / previous_bias_real

    return previous_extracted_sequence, previous_alpha
