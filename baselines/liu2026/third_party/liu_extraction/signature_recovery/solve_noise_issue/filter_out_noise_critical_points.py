import numpy as np
import jax
import jax.numpy as jnp

from ..utils import (
    get_hidden_layers,
    forward,
    get_polytope_at,
    on_which_hidden_layer,
    getAllWeightsAndBiases,
    predict_manual_fast,
    AcceptableFailure,
)
from ..compute_critical_point_values import get_ratios
from ..follow_critical_hyperplane import binary_search_towards


# This function is the same as the is_on_following_layer() in follow_critical_hyperplane.py
# It is used to filter out critical points that are not on the target layer
def is_on_target_layer(
    args,
    model,
    A,
    B,
    known_A,
    known_B,
    point,
    dimInput,
):
    if len(model.layers) == 3:
        GRAD_EPS = 1e1
    else:
        GRAD_EPS = 1e-4

    # print("\n[Noise Issue] Check if the critical point is on the target layer")
    layer_id, neuron_id = on_which_hidden_layer(point, model)
    print(
        "[Cheating Result] The critical point is on layer",
        layer_id + 1,
        "neuron",
        neuron_id + 1,
    )

    def is_on_prior_layer(query):
        if any(
            np.min(np.abs(layer)) < 1e-5 for layer in get_hidden_layers(query, A, B)
        ):
            return True

        next_A, next_B = A + [known_A], B + [known_B]
        next_hidden = forward(query, next_A, next_B)
        if np.min(np.abs(next_hidden)) < 1e-6:
            return True
        return False

    if args.real_attack == 0:
        if is_on_prior_layer(point):
            print(
                "[x] By is_on_target_layer(), this critical point is on a previous layer"
            )
            return False

    initial_signs = get_polytope_at(A, B, known_A, known_B, point)

    try:
        normal = get_ratios(args, model, [point], dimInput, eps=GRAD_EPS)[0].flatten()
    except AcceptableFailure:
        print("The normal vector is not found")
        print("[x] By is_on_target_layer(), this critical point is on a previous layer")
        return False

    normal = normal / np.sum(normal**2) ** 0.5

    if args.cheat_noise:
        # Cheating the noise issue
        layer_id, neuron_id = on_which_hidden_layer(point, model)
        if layer_id + 1 != args.layerID:
            print(
                "[x] By is_on_target_layer(), this critical point is on a deeper layer"
            )
            return False

    else:
        # Not cheating the noise issue
        weights, biases = getAllWeightsAndBiases(model)
        layer_id, neuron_id = on_which_hidden_layer(point, model)

        # for tol in range(100):
        for tol in range(100):
            # Try 100 different directions
            random_dir = np.random.normal(size=dimInput)
            perp_component = (
                np.dot(random_dir, normal) / (np.dot(normal, normal)) * normal
            )
            parallel_dir = random_dir - perp_component
            go_direction = parallel_dir / np.sum(parallel_dir**2) ** 0.5

            try:
                _, distance = binary_search_towards(
                    A, B, known_A, known_B, point, initial_signs, go_direction
                )
            except AcceptableFailure:
                continue
            if args.debug:
                print("The distance to the next layer is", distance)

            high_bound = 1e-3
            low_bound = -1e-3

            # Check the point in the same polytope
            point_in_same_polytope = point + (distance * 0.99 - 1e-4) * go_direction
            f_low = predict_manual_fast(
                (point_in_same_polytope + normal * low_bound)[np.newaxis, :],
                weights,
                biases,
            )
            f_high = predict_manual_fast(
                (point_in_same_polytope + normal * high_bound)[np.newaxis, :],
                weights,
                biases,
            )
            f_mid = predict_manual_fast(
                point_in_same_polytope[np.newaxis, :], weights, biases
            )

            if args.dataset == "cifar10":
                f_low = f_low[0][args.output_index]
                f_high = f_high[0][args.output_index]
                f_mid = f_mid[0][args.output_index]

            if np.abs(f_mid - (f_high + f_low) / 2) < 1e-8:
                print(
                    "[x] By is_on_target_layer(), this critical point is on a deeper layer"
                )
                if args.layerID == layer_id + 1:
                    print("A mistake is made in branch 1")
                    print(np.abs(f_mid - (f_high + f_low) / 2))
                return False

            high_bound = 1e-4
            low_bound = -1e-4
            # Check the point in a different polytope
            point_in_different_polytope = point + (distance * 1.1 + 1e-1) * go_direction
            f_low = predict_manual_fast(
                (point_in_different_polytope + normal * low_bound)[np.newaxis, :],
                weights,
                biases,
            )
            f_high = predict_manual_fast(
                (point_in_different_polytope + normal * high_bound)[np.newaxis, :],
                weights,
                biases,
            )
            f_mid = predict_manual_fast(
                point_in_different_polytope[np.newaxis, :], weights, biases
            )

            if args.dataset == "cifar10":
                f_low = f_low[0][args.output_index]
                f_high = f_high[0][args.output_index]
                f_mid = f_mid[0][args.output_index]

            if np.abs(f_mid - (f_high + f_low) / 2) > 1e-1:
                print(
                    "[x] By is_on_target_layer(), this critical point is on a deeper layer"
                )
                if args.layerID == layer_id + 1:
                    print("A mistake is made in branch 2")
                    print(np.abs(f_mid - (f_high + f_low) / 2))
                return False

    print("[y] By is_on_target_layer(), this critical point is on the target layer")
    return True


# Check the quality of the filtered critical points
def check_filtered_critical_points_quality(args, model, critilcal_points, dimOfLayer):
    if len(critilcal_points) == 0:
        print("No critical points to analyze")
        return

    good_count = 0
    bad_count = 0
    neurons = [0] * dimOfLayer
    for index, point in enumerate(critilcal_points):
        layer_id, neuron_id = on_which_hidden_layer(point, model)
        if layer_id + 1 == args.layerID:
            good_count += 1
            neurons[neuron_id] += 1
        else:
            bad_count += 1

    print("--------------------------------")
    print("[Cheating]Analyze the quality of the filtered critical points")
    print("[Cheating]The number of critical points on the target layer:", good_count)
    print("[Cheating]The number of critical points on deeper layers:", bad_count)
    print("[Cheating]The distribution of the critical points on the target layer:")
    print(neurons)
    print("--------------------------------")

    return good_count, bad_count, neurons
