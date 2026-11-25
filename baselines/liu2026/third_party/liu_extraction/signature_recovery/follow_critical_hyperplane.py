import numpy as np
import gc
import jax
import jax.numpy as jnp
import os

from .utils import (
    get_hidden_layers,
    forward,
    get_polytope_at,
    on_which_hidden_layer,
    getAllWeightsAndBiases,
    predict_manual_fast,
    matmul,
    get_hidden_at,
    check_multiple_intersection_point,
    AcceptableFailure,
)
from .compute_critical_point_values import get_ratios, get_ratios_lstsq
from .critical_point_search import do_better_sweep
from global_vars import (
    GlobalConfig,
)


# ----------
# Helper functions for get_more_crit_pts which is the hyperplane following algorithm to find more critical points
# once we have found at least two critical points for each neuron
# ----------


def is_on_following_layer(
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

    print("------------------------------------------------")
    print("Check if the critical point is on the next layer")

    def is_on_prior_layer(query):
        for layer in get_hidden_layers(query, A, B):
            print("Layer", layer)
            for i in range(len(layer)):
                if layer[i] != 0 and np.abs(layer[i]) < 1e-5:
                    print("Layer", i, "has zero")
                    return True

        next_A, next_B = A + [known_A], B + [known_B]
        next_hidden = forward(query, next_A, next_B)
        print("Next hidden", next_hidden)
        for i in range(len(next_hidden)):
            if next_hidden[i] != 0 and np.abs(next_hidden[i]) < 1e-4:
                print("Next hidden has zero")
                return True
        return False

        # if any(
        #     np.min(np.abs(layer[layer != 0])) < 1e-5
        #     for layer in get_hidden_layers(query, A, B)
        # ):
        #     return True

        # next_A, next_B = A + [known_A], B + [known_B]
        # next_hidden = forward(query, next_A, next_B)
        # if np.min(np.abs(next_hidden)) < 1e-4:
        #     return True
        # return False

    if is_on_prior_layer(point):
        return False

    initial_signs = get_polytope_at(A, B, known_A, known_B, point)

    normal = get_ratios(model, [point], dimInput, eps=GRAD_EPS)[0].flatten()
    # normal = get_ratios_lstsq(model, [point], A=[], B=[], eps=GRAD_EPS)[0].flatten()
    normal = normal / np.sum(normal**2) ** 0.5

    layer_id, neuron_id = on_which_hidden_layer(point, model)
    weights, biases = getAllWeightsAndBiases(model)

    for tol in range(100):
        random_dir = np.random.normal(size=dimInput)
        perp_component = np.dot(random_dir, normal) / (np.dot(normal, normal)) * normal
        parallel_dir = random_dir - perp_component

        go_direction = parallel_dir / np.sum(parallel_dir**2) ** 0.5

        try:
            _, distance = binary_search_towards(
                A, B, known_A, known_B, point, initial_signs, go_direction
            )
        except AcceptableFailure:
            continue
        # print("The distance to the next layer is", distance)
        high_bound = 1e-4
        low_bound = -1e-4

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
            # print(
            #     "Difference 1",
            #     np.abs(f_mid - (f_high + f_low) / 2),
            # )
            # x = copy.deepcopy(point_in_same_polytope)
            # for i in range(0, layer_id + 1):
            #     x = np.matmul(x, weights[i]) + biases[i]
            #     if i < layer_id:
            #         x = x * (x > 0)
            #     print(x)
            # print(f_low, f_mid, f_high)
            # print("Branch 1")
            return False

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

        if np.abs(f_mid - (f_high + f_low) / 2) > 1e-4:
            # print(
            #     "Difference 2",
            #     np.abs(f_mid - (f_high + f_low) / 2),
            # )
            # print("Initial output")
            # x = copy.deepcopy(point)
            # for i in range(0, layer_id + 1):
            #     x = np.matmul(x, weights[i]) + biases[i]
            #     if i < layer_id:
            #         x = x * (x > 0)
            #     print(x)
            # print("point_in_same_polytope output")
            # x = copy.deepcopy(point_in_same_polytope)
            # for i in range(0, layer_id + 1):
            #     x = np.matmul(x, weights[i]) + biases[i]
            #     if i < layer_id:
            #         x = x * (x > 0)
            #     print(x)
            # print("point_in_different_polytope output")
            # x = copy.deepcopy(point_in_different_polytope)
            # for i in range(0, layer_id + 1):
            #     x = np.matmul(x, weights[i]) + biases[i]
            #     if i < layer_id:
            #         x = x * (x > 0)
            #     print(x)
            # print(f_low, f_mid, f_high)
            # print("Branch 2")
            return False

    return True


PREV_GRAD = None


def binary_search_towards(
    A, B, known_A, known_B, start_point, initial_signs, go_direction, maxstep=1e6
):
    """
    Compute how far we can walk along the hyperplane until it is in a
    different polytope from a prior layer.

    It is okay if it's in a differnt polytope in a *later* layer, because
    it will still have the same angle.

    (but do it analytically by looking at the signs of the first layer)
    this requires no queries and could be done with math but instead
    of thinking I'm just going to run binary search.
    """
    global PREV_GRAD

    new_A, new_B = A + [known_A], B + [known_B]
    # this is the hidden state
    initial_hidden = np.array(get_hidden_layers(start_point, new_A, new_B, flat=True))
    delta_hidden_np = (
        np.array(
            get_hidden_layers(
                start_point + 1e-6 * go_direction, new_A, new_B, flat=True
            )
        )
        - initial_hidden
    ) * 1e6
    #
    can_go_dist_all = initial_hidden / delta_hidden_np
    # print("[1st]Can go dist all", can_go_dist_all)

    # if (
    #     PREV_GRAD is None
    #     or PREV_GRAD[0] is not A
    #     or PREV_GRAD[1] is not B
    #     or PREV_GRAD[2] is not known_A
    #     or PREV_GRAD[3] is not known_B
    # ):

    #     def get_grad(x, i):
    #         initial_hidden = get_hidden_layers(x, new_A, new_B, flat=True, np=jnp)
    #         return initial_hidden[i]

    #     g = jax.jit(jax.grad(get_grad))

    #     def grads(start_point, go_direction):
    #         return jnp.array(
    #             [
    #                 jnp.dot(g(start_point, i), go_direction)
    #                 for i in range(initial_hidden.shape[0])
    #             ]
    #         )

    #     PREV_GRAD = (A, B, known_A, known_B, jax.jit(grads))
    # else:
    #     grads = PREV_GRAD[4]

    # delta_hidden = grads(start_point, go_direction)

    # can_go_dist_all = np.array(initial_hidden / delta_hidden)
    # print("[2nd]Can go dist all", can_go_dist_all)

    can_go_dist = -can_go_dist_all[can_go_dist_all < 0]

    if len(can_go_dist) == 0:
        # print("Can't go anywhere at all")
        raise AcceptableFailure()

    can_go_dist = np.min(can_go_dist)
    if can_go_dist > 1e6:
        # print("The distance is too large, something is wrong")
        raise AcceptableFailure()

    a_bit_further = start_point + (can_go_dist + 1e-4) * go_direction
    return a_bit_further, can_go_dist


def find_plane_angle(
    args,
    model,
    A,
    B,
    known_A,
    known_B,
    dimInput,
    multiple_intersection_point,
    sign_at_init,
    init_step,
    exponential_base=1.5,
):
    """
    Given an input that's at the multiple intersection point, figure out how
    to continue along the path after it bends.


                /       X    : multiple intersection point
       ......../..      ---- : layer N hyperplane
       .      /  .       |   : layer N+1 hyperplane that bends
       .     /   .
    --------X-----------
       .    |    .
       .    |    .
       .....|.....
            |
            |

    We need to make sure to bend, and not turn onto the layer N hyperplane.

    To do this we will draw a box around the X and intersect with the planes
    and determine the four coordinates. Then draw another box twice as big.

    The first layer plane will be the two points at a consistent angle.
    The second layer plane will have an inconsistent angle.

    Choose the inconsistent angle plane, and make sure we move to a new
    polytope and don't just go backwards to where we've already been.
    """
    success = None
    camefrom = None

    prev_iter_intersections = []

    while True:
        x_dir_base = np.sign(np.random.normal(size=dimInput)) / dimInput**0.5
        y_dir_base = np.sign(np.random.normal(size=dimInput)) / dimInput**0.5
        # When the input dimension is odd we can't have two orthogonal
        # vectors from {-1,1}^DIM
        if np.abs(np.dot(x_dir_base, y_dir_base)) <= dimInput % 2 + 1e-8:
            break

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
            for index, b in enumerate(recovered_parameters["extracted_bias"]):
                if abs(b) == 0:
                    dead_neuron.append(index)
            dead_neurons.append(dead_neuron)
    print("The dead neurons from the previous layer are", dead_neurons)

    MAX = 35

    start = [10] if init_step > 10 else []
    if args.debug:
        print("Init_step is:", init_step)

    for stepsize in start + list(range(init_step, MAX)):
        if args.debug:
            print("\tTry stepping away", stepsize)
        x_dir = x_dir_base * (exponential_base ** (stepsize - 10))
        y_dir = y_dir_base * (exponential_base ** (stepsize - 10))

        # print("x_dir_base", x_dir_base)
        # print("y_dir_base", y_dir_base)
        # print("x_dir", x_dir)
        # print("y_dir", y_dir)

        # Draw the box as shown in the diagram above, and compute where
        # the critical points are.
        top = do_better_sweep(
            args,
            model,
            offset=multiple_intersection_point + x_dir,
            direction=y_dir,
            low=-1,
            high=1,
        )
        bot = do_better_sweep(
            args,
            model,
            offset=multiple_intersection_point - x_dir,
            direction=y_dir,
            low=-1,
            high=1,
        )
        left = do_better_sweep(
            args,
            model,
            offset=multiple_intersection_point + y_dir,
            direction=x_dir,
            low=-1,
            high=1,
        )
        right = do_better_sweep(
            args,
            model,
            offset=multiple_intersection_point - y_dir,
            direction=x_dir,
            low=-1,
            high=1,
        )
        intersections = top + bot + left + right
        if args.debug:
            print(
                "top", len(top), "bot", len(bot), "left", len(left), "right", len(right)
            )
            print("\tFound", len(intersections), "intersections")

        print("top", len(top), "bot", len(bot), "left", len(left), "right", len(right))
        print("\tFound", len(intersections), "intersections")

        # If we only have two critical points, and we're taking a big step,
        # then something is seriously messed up.
        # This is not an acceptable error. Just abort out and let's try to
        # do the whole thing again.
        if len(intersections) == 2 and stepsize >= 10:
            raise AcceptableFailure()

        if (
            len(intersections) == 0 and stepsize > 20
        ):  # if (len(intersections) == 0 and stepsize > 15):# or (len(intersections) == 3 and stepsize > 5):
            # Probably we're in just a constant flat 0 region
            # At this point we're basically dead in the water.
            # Just fail up and try again.
            if args.debug:
                print("\tIt looks like we're in a flat region, raise failure")
            raise AcceptableFailure()

        # If we somehow went from almost no critical points to more than 4,
        # then we've really messed up.
        # Just fail out and let's hope next time it doesn't happen.
        if len(intersections) > 4 and len(prev_iter_intersections) < 2:
            if args.debug:
                print("\tWe didn't get enough inner points")
            if exponential_base == 1.2:
                # if exponential_base == 0.9:
                if args.debug:
                    print("\tIt didn't work a second time")
                return None, None, 0
            else:
                if args.debug:
                    print("\tTry with smaller step")
                return find_plane_angle(
                    args,
                    model,
                    A,
                    B,
                    known_A,
                    known_B,
                    dimInput,
                    multiple_intersection_point,
                    sign_at_init,
                    init_step,
                    exponential_base=1.2,
                    # exponential_base=0.9,
                )

        # This is the good, expected code-path.
        # We've seen four intersections at least twice before, and now
        # we're seeing more than 4.
        if (len(intersections) > 4 or stepsize > 20) and len(
            prev_iter_intersections
        ) >= 2:
            next_intersections = np.array(prev_iter_intersections[-1])
            intersections = np.array(prev_iter_intersections[-2])

            # Let's first figure out what points are responsible for the prior-layer neurons
            # being zero, and which are from the current-layer neuron being zero
            candidate = []
            for i, a in enumerate(intersections):
                for j, b in enumerate(intersections):
                    if i == j:
                        continue
                    score = np.sum(((a + b) / 2 - multiple_intersection_point) ** 2)
                    a_to_b = b - a
                    a_to_b /= np.sum(a_to_b**2) ** 0.5

                    variance = np.std((next_intersections - a) / a_to_b, axis=1)
                    best_variance = np.min(variance)

                    candidate.append((best_variance, i, j))

            if sorted(candidate)[3][0] < 1e-8:
                # It looks like both lines are linear here
                # We can't distinguish what way is the next best way to go.
                if args.debug:
                    print("\tFailed the box continuation finding procedure. (1)")
                # print("\t",candidate)
                raise AcceptableFailure()

            # Sometimes life is just ugly, and nothing wants to work.
            # Just abort.
            err, index_0, index_1 = min(candidate)
            if err / max(candidate)[0] > 1e-5:
                return None, None, 0

            prior_layer_near_zero = np.zeros(4, dtype=bool)
            prior_layer_near_zero[index_0] = True
            prior_layer_near_zero[index_1] = True

            # Now let's walk through each of these points and check that everything looks sane.
            should_fail = False
            for critical_point, is_prior_layer_zero in zip(
                intersections, prior_layer_near_zero
            ):
                new_A, new_B = A + [known_A], B + [known_B]
                vs = get_hidden_layers(critical_point, new_A, new_B)

                if is_prior_layer_zero:
                    # We expect the prior layer to be zero.
                    if all([np.min(np.abs(x)) > 1e-5 for x in vs]):
                        # If it looks like it's not actually zero, then brutally fail.
                        print("\tAbort 1: failed to find a valid box")
                        should_fail = True
                if args.real_attack == 0:
                    if any([np.min(np.abs(x)) < 1e-10 for x in vs]):
                        # We expect the prior layer to be zero.
                        if not is_prior_layer_zero:
                            # If it looks like it's not actually zero, then brutally fail.
                            print("\tAbort 2: failed to find a valid box")
                            should_fail = True
                if args.real_attack == 1 and args.layerID >= 2:
                    for i in range(len(vs)):
                        x = vs[i]
                        for j in range(len(x)):
                            if np.abs(x[j]) < 1e-5 and j not in dead_neurons[i]:
                                # We expect the prior layer to be zero.
                                if not is_prior_layer_zero:
                                    # If it looks like it's not actually zero, then brutally fail.
                                    print("\tAbort 2: failed to find a valid box")
                                    should_fail = True
                    # if any([np.min(np.abs(x)) < 1e-5 for x in vs]):
                    #     # We expect the prior layer to be zero.
                    #     if not is_prior_layer_zero:
                    #         # If it looks like it's not actually zero, then brutally fail.
                    #         print("\tAbort 2: failed to find a valid box")
                    #         should_fail = True
            if should_fail:
                return None, None, 0

            # Done with error checking, life is good here.
            # Find the direction that corresponds to the next direction we can move in
            # and continue our search from that point.
            for critical_point, is_prior_layer_zero in zip(
                intersections, prior_layer_near_zero
            ):
                # print("\tIs prior layer zero", is_prior_layer_zero)

                sign_at_crit = sign_to_int(
                    get_polytope_at(A, B, known_A, known_B, critical_point)
                )
                if args.debug:
                    print(
                        "\tMove to",
                        sign_at_crit,
                        "versus",
                        sign_at_init,
                        is_prior_layer_zero,
                    )
                if not is_prior_layer_zero:
                    if sign_at_crit != sign_at_init:
                        success = critical_point
                    else:
                        camefrom = critical_point

            # If we didn't get a solution, then abort out.
            # Probably what happened here is that we got more than four points
            # on the box but didn't see exactly four points on the box twice before
            # this means we should decrease the initial step size and try again.
            if success is None:
                print("\tFailed the box continuation finding procedure. (2)")
                raise AcceptableFailure()
                # assert success is not None
            break
        if len(intersections) == 4:
            prev_iter_intersections.append(intersections)
    gc.collect()
    # print("success", success, "camefrom", camefrom, "stepsize", stepsize)
    if args.debug:
        print("success camefrom", camefrom, "stepsize", stepsize)
    return success, camefrom, min(stepsize, MAX - 3)


def sign_to_int(signs):
    """
    Convert a list to an integer.
    [-1, 1, 1, -1], -> 0b0110 -> 6
    """
    return int("".join("0" if x == -1 else "1" for x in signs), 2)


def follow_hyperplane(
    args,
    LAYER,
    start_point,
    A,
    B,
    known_A,
    known_B,
    model,
    dead_neurons,
    dimInput,
    dimOfPrevLayer,
    dimOfLayer,
    special,
    history=[],
    MAX_POINTS=1e3,
    only_need_positive=False,
    target_neuron=None,
):
    """
    This is the ugly algorithm that will let us recover sign for expansive networks.
    Assumes we have extracted up to layer K-1 correctly, and layer K up to sign.

    start_point is a neuron on layer K+1

    known_T is the transformation that computes up to layer K-1, with
    known_A and known_B being the layer K matrix up to sign.

    We're going to come up with a bunch of different inputs,
    each of which has the same critical point held constant at zero.
    """

    def choose_new_direction_from_minimize(previous_axis):
        """
        Given the current point which is at a critical point of the next
        layer neuron, compute which direction we should travel to continue
        with finding more points on this hyperplane.

        Our goal is going to be to pick a direction that lets us explore
        a new part of the space we haven't seen before.
        """

        print("Choose a new direction to travel in")
        if len(history) == 0:
            which_to_change = 0
            new_perp_dir = perp_dir
            new_start_point = start_point
            initial_signs = get_polytope_at(A, B, known_A, known_B, start_point)

            # If we're in the 1 region of the polytope then we try to make it smaller
            # otherwise make it bigger
            fn = min if initial_signs[0] == 1 else max
        else:
            neuron_values = np.array([x[1] for x in history])

            neuron_positive_count = np.sum(neuron_values > 1e-5, axis=0)
            neuron_negative_count = np.sum(neuron_values < -1e-5, axis=0)

            mean_plus_neuron_value = neuron_positive_count / (
                neuron_positive_count + neuron_negative_count + 1
            )
            mean_minus_neuron_value = neuron_negative_count / (
                neuron_positive_count + neuron_negative_count + 1
            )

            # we want to find values that are consistently 0 or 1
            # So map 0 -> 1 and 1 -> 0 and the middle to higher values
            if only_need_positive:
                neuron_consistency = mean_plus_neuron_value
            else:
                neuron_consistency = mean_plus_neuron_value * mean_minus_neuron_value

            # Print out how much progress we've made.
            # This estimate is probably worse than Windows 95's estimated time remaining.
            # At least it's monotonic. Be thankful for that.
            print(
                "Progress", "%.1f" % int(np.mean(neuron_consistency != 0) * 100) + "%"
            )
            print("Counts on each side of each neuron(history)")
            print(neuron_positive_count)
            print(neuron_negative_count)
            print("Neuron consistency: ", neuron_consistency)

            # Choose the smallest value, which is the most consistent
            sorted_neuron_consistency = np.argsort(neuron_consistency)
            which_to_change_index = 0
            while sorted_neuron_consistency[which_to_change_index] in dead_neurons:
                which_to_change_index += 1
            which_to_change = sorted_neuron_consistency[which_to_change_index]
            print("Try to explore the other side of neuron", which_to_change)

            print("Previous axis", previous_axis)

            # We only want to change the specific neuron that we're working on
            # best = (np.inf, None, None)
            # for _, potential_hidden_vector, potential_point in history[-1:]:
            #     value = potential_hidden_vector[which_to_change]
            #     if np.abs(value) < best[0]:
            #         best = (np.abs(value), which_to_change, potential_point)
            # _, which_to_change, new_start_point = best
            # new_perp_dir = perp_dir

            # Original code
            if which_to_change != previous_axis:
                if (
                    previous_axis is not None
                    and neuron_consistency[previous_axis]
                    == neuron_consistency[which_to_change]
                ):
                    # If the previous thing we were working towards has the same value as this one
                    # the don't change our mind and just keep going at that one
                    # (almost always--sometimes we can get stuck, let us get unstuck)
                    which_to_change = previous_axis
                    new_start_point = start_point
                    new_perp_dir = perp_dir
                else:
                    valid_axes = np.where(
                        neuron_consistency == neuron_consistency[which_to_change]
                    )[0]

                    best = (np.inf, None, None)
                    for _, potential_hidden_vector, potential_point in history[-1:]:
                        for potential_axis in valid_axes:
                            value = potential_hidden_vector[potential_axis]
                            if np.abs(value) < best[0]:
                                best = (np.abs(value), potential_axis, potential_point)
                    _, which_to_change, new_start_point = best
                    new_perp_dir = perp_dir

            else:
                new_start_point = start_point
                new_perp_dir = perp_dir

            print("Try to explore the other side of neuron", which_to_change)
            # If we're in the 1 region of the polytope then we try to make it smaller
            # otherwise make it bigger
            fn = (
                min
                if neuron_positive_count[which_to_change]
                > neuron_negative_count[which_to_change]
                else max
            )
            arg_fn = (
                np.argmin
                if neuron_positive_count[which_to_change]
                > neuron_negative_count[which_to_change]
                else np.argmax
            )
            print(
                "Changing",
                which_to_change,
                "to flip sides because mean is",
                mean_plus_neuron_value[which_to_change],
            )

        val = matmul(forward(new_start_point, A, B, with_relu=True), known_A, known_B)[
            which_to_change
        ]

        initial_signs = get_polytope_at(A, B, known_A, known_B, new_start_point)

        # Now we're going to figure out what direction makes this biggest/smallest
        # this doesn't take any queries
        # There's probably an analytical way to do this.
        # But thinking is hard. Just try 1000 random angles.
        # There are no queries involved in this process.

        # Maybe one can make this more efficient??? -Hanna

        choices = []
        print("Figure out what direction makes this biggest/smallest")
        for _ in range(1000):
            random_dir = np.random.normal(size=dimInput)
            # Perpendicular component
            perp_component = (
                np.dot(random_dir, new_perp_dir)
                / (np.dot(new_perp_dir, new_perp_dir))
                * new_perp_dir
            )
            parallel_dir = random_dir - perp_component

            # This is the direction we're going to travel in.
            go_direction = parallel_dir / np.sum(parallel_dir**2) ** 0.5

            try:
                a_bit_further, high = binary_search_towards(
                    A, B, known_A, known_B, new_start_point, initial_signs, go_direction
                )
            except AcceptableFailure:
                # print("Binary search failed.")
                continue
            if a_bit_further is None:
                continue

            # choose a direction that makes the Kth value go down by the most
            val = matmul(
                forward(a_bit_further[np.newaxis, :], A, B, with_relu=True),
                known_A,
                known_B,
            )[0][which_to_change]
            # choices.append([val, new_start_point + high * go_direction])
            choices.append([val, a_bit_further])

        if len(choices) == 0:
            raise AcceptableFailure()
        best_value, multiple_intersection_point = fn(choices, key=lambda x: x[0])

        print("Value", best_value)
        # print("Multiple intersection point", multiple_intersection_point)
        print("Which to change", which_to_change)

        return new_start_point, multiple_intersection_point, which_to_change

    def check_relevant_crits(points_on_plane, neuron_values, neuron_positive_count):
        # Check if critical points are relevant and return only those that are relevant
        relevant_points_on_plane = []
        for i in range(len(points_on_plane)):
            relevant_neurons = neuron_positive_count <= 15
            points_contributing = (neuron_values[i] > 1e-5) & relevant_neurons
            if np.any(points_contributing):
                relevant_points_on_plane.append(points_on_plane[i])
            else:
                mask = neuron_values[i] > 1e-5
                neuron_positive_count[mask] -= 1

        print("we have", len(relevant_points_on_plane), "relevant points")
        return relevant_points_on_plane

    def is_on_prior_layer(query):
        for layer in get_hidden_layers(query, A, B):
            print("Layer", layer)
            for i in range(len(layer)):
                if layer[i] != 0 and np.abs(layer[i]) < 1e-5:
                    print("Layer", i, "has zero")
                    return True

        next_A, next_B = A + [known_A], B + [known_B]
        next_hidden = forward(query, next_A, next_B)
        print("Next hidden", next_hidden)
        for i in range(len(next_hidden)):
            if next_hidden[i] != 0 and np.abs(next_hidden[i]) < 1e-6:
                print("Next hidden has zero")
                return True
        return False

    ###################################################
    ### Actual code to do the sign recovery starts. ###
    ###################################################
    start_box_step = 0
    points_on_plane = []
    current_change_axis = 0
    print("The target weight index is", target_neuron)
    if target_neuron != None:
        target_neuron_count = 0
        init_target_neuron_count = 0
    else:
        target_neuron_count = 1
        init_target_neuron_count = 0

    while True:
        gc.collect()
        print("\n\n")
        print("-----" * 10)

        # Keep track of where we've been, so we can go to new places.
        which_polytope = get_polytope_at(
            A, B, known_A, known_B, start_point, False
        )  # [-1 1 -1]
        hidden_vector = get_hidden_at(A, B, known_A, known_B, LAYER, start_point, False)
        sign_at_init = sign_to_int(which_polytope)  # 0b010 -> 2

        print("Number of collected points", len(points_on_plane))
        print("Number of points in history", len(history))
        neuron_values = np.array([x[1] for x in history])
        neuron_positive_count = np.sum(neuron_values > 1e-5, axis=0)
        neuron_negative_count = np.sum(neuron_values < -1e-5, axis=0)
        print("Positive count (history):", neuron_positive_count)
        print("Negative count (history):", neuron_negative_count)

        if target_neuron != None:
            print("Target neuron: ", target_neuron)
            target_neuron_count = min(
                neuron_positive_count[target_neuron],
                neuron_negative_count[target_neuron],
            )
            if len(points_on_plane) == 0:
                init_target_neuron_count = target_neuron_count
            print(
                "Target neuron count: ", target_neuron_count, init_target_neuron_count
            )

        if len(points_on_plane) > MAX_POINTS:
            print("The number of points on the plane is enough")
            print("Number of collected points on the plane is", len(points_on_plane))

            return points_on_plane, False
            # neuron_values = np.array(
            #     [
            #         get_hidden_at(A, B, known_A, known_B, LAYER, x, False)
            #         for x in points_on_plane
            #     ]
            # )

            # neuron_positive_count = np.sum(neuron_values > 1e-5, axis=0)
            # neuron_negative_count = np.sum(neuron_values < -1e-5, axis=0)
            # print("Neuron positive count (points_on_plane): ", neuron_positive_count)
            # print("Neuron negative count (points_on_plane): ", neuron_negative_count)

            # if not GlobalConfig.set_Carlini:
            #     relevant_points_on_plane = check_relevant_crits(
            #         points_on_plane, neuron_values, neuron_positive_count
            #     )
            #     return relevant_points_on_plane, False
            # else:
            #     return points_on_plane, False

        if (
            (np.all(neuron_positive_count > 0) and np.all(neuron_negative_count > 0))
            or (only_need_positive and np.all(neuron_positive_count > 0))
        ) and (target_neuron_count - init_target_neuron_count > 0):
            print("Have all the points we need (1)")
            print("Neuron positive count: ", neuron_positive_count)
            print("Neuron negative count: ", neuron_negative_count)

            neuron_values = np.array(
                [
                    get_hidden_at(A, B, known_A, known_B, LAYER, x, False)
                    for x in points_on_plane
                ]
            )

            neuron_positive_count = np.sum(neuron_values > 1e-5, axis=0)
            neuron_negative_count = np.sum(neuron_values < -1e-5, axis=0)
            print("Neuron positive count (points_on_plane): ", neuron_positive_count)
            print("Neuron negative count (points_on_plane): ", neuron_negative_count)
            if not GlobalConfig.set_Carlini:
                relevant_points_on_plane = check_relevant_crits(
                    points_on_plane, neuron_values, neuron_positive_count
                )
                return relevant_points_on_plane, True
            else:
                return points_on_plane, True

        # 1. find a way to move along the hyperplane by computing the normal
        # direction using the ratios function. Then find a parallel direction.

        try:
            # perp_dir = get_ratios([start_point], [range(DIM)], eps=1e-4)[0].flatten()
            perp_dir, _ = get_ratios_lstsq(
                args, model, [start_point], A=[], B=[], eps=1e-5
            )
            perp_dir = perp_dir[0].flatten()
            # breakpoint()

            # Notice that we're not using the ratios function here.
            # print("size of known_A", len(known_A))

        except AcceptableFailure:
            print(
                "Failed to compute ratio at start point. Something very bad happened."
            )
            return points_on_plane, False

        # Record these points.
        # print("Adding the points to the set of known good points")
        # points_on_plane.append(start_point)
        # history.append((which_polytope, hidden_vector, np.copy(start_point)))

        # We can't just pick any parallel direction. If we did, then we would
        # not end up covering much of the input space.

        # Instead, we're going to figure out which layer-1 hyperplanes are "visible"
        # from the current point. Then we're going to try and go reach all of them.

        # This is the point at which the first and second layers intersect.
        try:
            start_point, multiple_intersection_point, new_change_axis = (
                choose_new_direction_from_minimize(current_change_axis)
            )
        except AcceptableFailure:
            return points_on_plane, False
        if new_change_axis != current_change_axis:
            try:
                start_point, multiple_intersection_point, current_change_axis = (
                    choose_new_direction_from_minimize(None)
                )
            except AcceptableFailure:
                return points_on_plane, False
        # breakpoint()

        # Check the quality of the point multiple_intersection_point we're going to move to.
        is_good, previous_layer_id = check_multiple_intersection_point(
            multiple_intersection_point, model, args.layerID
        )
        if not is_good:
            print(
                "[multiple_intersection_point, Before]The point is not good, try again"
            )
        else:
            print("[multiple_intersection_point, Before]The point is good")
            print("The previous layer id is ", previous_layer_id + 1)

        neuron_id, layer_id = on_which_hidden_layer(start_point, model)
        print("[start_point]The neuron id is ", neuron_id + 1)
        print("[start_point]The layer id is ", layer_id + 1)

        # Refine the direction we're going to travel in---stay numerically stable.
        towards_multiple_direction = multiple_intersection_point - start_point
        step_distance = np.sum(towards_multiple_direction**2) ** 0.5
        print("Distance we need to step:", step_distance)

        if step_distance > 1e6:
            print("Very large step distance, try again")
            continue

        if step_distance > 1 or True:
            # if step_distance > 1e-3:
            mid_point = (
                1e-4
                * towards_multiple_direction
                / np.sum(towards_multiple_direction**2) ** 0.5
                + start_point
            )

            for range_bound in [1e-3, 1e-2]:
                mid_points = do_better_sweep(
                    args,
                    model,
                    offset=mid_point,
                    direction=perp_dir / np.sum(perp_dir**2) ** 0.5,
                    low=-range_bound,
                    high=range_bound,
                )
                good_mid_points = []
                for point in mid_points:
                    layer_id, neuron_id = on_which_hidden_layer(point, model)
                    print("[new_points]The layer id is ", layer_id + 1)
                    print("[new_points]The neuron id is ", neuron_id + 1)

                    if not is_on_prior_layer(point):
                        print("We find a good point for refinement")
                        good_mid_points.append(point)

                mid_points = good_mid_points

                if len(mid_points) > 0:
                    break

            if len(mid_points) > 0:
                print("We find", len(mid_points), "good points for refinement")
                mid_point = mid_points[
                    np.argmin(np.sum((mid_point - mid_points) ** 2, axis=1))
                ]

                towards_multiple_direction = mid_point - start_point
                towards_multiple_direction = (
                    towards_multiple_direction
                    / np.sum(towards_multiple_direction**2) ** 0.5
                )

                initial_signs = get_polytope_at(A, B, known_A, known_B, start_point)
                try:
                    _, high = binary_search_towards(
                        A,
                        B,
                        known_A,
                        known_B,
                        start_point,
                        initial_signs,
                        towards_multiple_direction,
                    )
                except AcceptableFailure:
                    # print("Binary search failed.")
                    return points_on_plane, False

                multiple_intersection_point = (
                    towards_multiple_direction * high + start_point
                )

        # Check the quality of the point multiple_intersection_point we're going to move to.
        is_good, previous_layer_id = check_multiple_intersection_point(
            multiple_intersection_point, model, args.layerID
        )
        if not is_good:
            print(
                "[multiple_intersection_point, After]The point is not good, try again"
            )
        else:
            print("[multiple_intersection_point, After]The point is good")
            print("The previous layer id is ", previous_layer_id + 1)
        # breakpoint()

        # Find the angle of the next hyperplane
        # First, take random steps away from the intersection point
        # Then run the search algorithm to find some intersections
        # what we find will either be a layer-1 or layer-2 intersection.

        print("Now try to find the continuation direction")
        success = None
        while success is None:
            if start_box_step < 0:
                start_box_step = 0
                print("VERY BAD FAILURE")
                print("Choose a new random point to start from")
                which_point = np.random.randint(0, len(history))
                start_point = history[which_point][2]
                # print("New point is", which_point)
                current_change_axis = np.random.randint(0, dimOfLayer)
                print("New axis to change", current_change_axis)
                break

            if args.debug:
                print("\tStart the box step with size", start_box_step)
            try:
                success, camefrom, stepsize = find_plane_angle(
                    args,
                    model,
                    A,
                    B,
                    known_A,
                    known_B,
                    dimInput,
                    multiple_intersection_point,
                    sign_at_init,
                    start_box_step,
                )
            except AcceptableFailure:
                # Go back to the top and try with a new start point
                print("\tOkay we need to try with a new start point")
                start_box_step = -10

            start_box_step -= 2
            print("start_box_step is", start_box_step)
            # breakpoint()

        if success is None:
            continue
        else:
            print("[Hopefully] Found a continuation direction")
            layer_id, neuron_id = on_which_hidden_layer(success, model)
            print("[success]The layer id is ", layer_id + 1)
            print("[success]The neuron id is ", neuron_id + 1)

        val = matmul(
            forward(multiple_intersection_point, A, B, with_relu=True), known_A, known_B
        )[new_change_axis]
        print("Value at multiple:", val)
        val = matmul(forward(success, A, B, with_relu=True), known_A, known_B)[
            new_change_axis
        ]
        print("Value at success:", val)

        if stepsize < 10:
            new_move_direction = success - multiple_intersection_point

            new_point_close_to_success = success + new_move_direction * 1e-2
            refine = True
            try:
                new_perp_dir, _ = get_ratios_lstsq(
                    args, model, [new_point_close_to_success], A=[], B=[], eps=1e-5
                )
                new_perp_dir = new_perp_dir[0].flatten()
            except AcceptableFailure:
                print(
                    "Failed to compute ratio at start point. Something very bad happened."
                )
                refine = False

            if refine:
                for range_bound in [1e-4, 1e-3, 1e-2]:
                    new_points = do_better_sweep(
                        args,
                        model,
                        offset=new_point_close_to_success,
                        direction=new_perp_dir / np.sum(new_perp_dir**2) ** 0.5,
                        low=-range_bound,
                        high=range_bound,
                    )
                    good_new_points = []
                    for point in new_points:
                        layer_id, neuron_id = on_which_hidden_layer(point, model)
                        print("[new_points]The layer id is ", layer_id + 1)
                        print("[new_points]The neuron id is ", neuron_id + 1)

                        if not is_on_prior_layer(point):
                            good_new_points.append(point)

                    new_points = good_new_points

                    if len(new_points) > 0:
                        break

                if len(new_points) > 0:
                    new_point = new_points[
                        np.argmin(
                            np.sum(
                                (new_point_close_to_success - new_points) ** 2, axis=1
                            )
                        )
                    ]

                    layer_id, neuron_id = on_which_hidden_layer(new_point, model)
                    print("[new_point]The layer id is ", layer_id + 1)
                    print("[new_point]The neuron id is ", neuron_id + 1)

                    new_move_direction = new_point - success

            # We don't want to be right next to the multiple intersection point.
            # So let's binary search to find how far away we can go while remaining in this polytope.
            # Then we'll go half as far as we can maximally go.

            initial_signs = get_polytope_at(A, B, known_A, known_B, success)
            if args.debug:
                print("polytope at success", sign_to_int(initial_signs))

            low = 0
            high = 1
            while high - low > 1e-2:
                mid = (high + low) / 2
                # query_point = multiple_intersection_point + mid * new_move_direction
                query_point = success + mid * new_move_direction
                next_signs = get_polytope_at(A, B, known_A, known_B, query_point)
                if args.debug:
                    print(
                        "polytope at",
                        mid,
                        sign_to_int(next_signs),
                        "%x" % (sign_to_int(next_signs) ^ sign_to_int(initial_signs)),
                    )
                if initial_signs == next_signs:
                    low = mid
                else:
                    high = mid
            print("GO TO", mid)

            # success = multiple_intersection_point + (mid / 2) * new_move_direction
            success = success + (mid / 2) * new_move_direction

            val = matmul(forward(success, A, B, with_relu=True), known_A, known_B)[
                new_change_axis
            ]
            print("Value at moved success:", val)

        from .solve_noise_issue.filter_out_noise_critical_points import (
            is_on_target_layer,
        )

        if is_on_target_layer(args, model, A, B, known_A, known_B, success, dimInput):
            start_point = success
            start_box_step = max(stepsize - 1, 0)

            # # Record these points.
            print("Adding the points to the set of known good points")
            points_on_plane.append(start_point)

            which_polytope = get_polytope_at(
                A, B, known_A, known_B, start_point, False
            )  # [-1 1 -1]
            hidden_vector = get_hidden_at(
                A, B, known_A, known_B, LAYER, start_point, False
            )
            history.append((which_polytope, hidden_vector, np.copy(start_point)))

            if camefrom is not None:
                if is_on_target_layer(
                    args, model, A, B, known_A, known_B, camefrom, dimInput
                ):
                    points_on_plane.append(camefrom)

        else:
            print("Not on the target layer")
            return points_on_plane, False


def get_more_crit_pts(
    args,
    A,
    B,
    known_A0,
    known_B0,
    critical_points,
    LAYER,
    model,
    dead_neurons,
    dimInput,
    dimOfPrevLayer,
    dimOfLayer,
    special,
    already_checked_critical_points=False,
    only_need_positive=False,
    l1_mask=None,
    target_neuron=None,
):
    # Also cut the part that is not necessary for just generating new points
    # Do not need sign recovery here
    """
    Find more critical points for neurons in the target layer, when we have already found at least 2 critical point for each neuron.
    """

    def get_critical_points():
        for point in critical_points:
            # print("Tick")
            if already_checked_critical_points or is_on_following_layer(
                args,
                model,
                A,
                B,
                known_A0,
                known_B0,
                point,
                dimInput,
            ):
                # print("Found layer N point at ", point, already_checked_critical_points)
                yield point

    get_critical_point = get_critical_points()

    print("Start looking for critical point")
    # if dimOfLayer == 8:
    #     MAX_POINTS = 200
    # if dimOfLayer == 16:
    #     MAX_POINTS = 100
    MAX_POINTS = 200
    if args.real_attack == 1:
        MAX_POINTS = 100
    which_point = next(get_critical_point)
    print("Done looking for critical point")

    initial_points = []
    history = []
    pts = []
    which_polytope = get_polytope_at(
        A, B, known_A0, known_B0, which_point, False
    )  # [-1 1 -1]
    hidden_vector = get_hidden_at(A, B, known_A0, known_B0, LAYER, which_point, False)
    history.append((which_polytope, hidden_vector, np.copy(which_point)))

    if already_checked_critical_points:
        for point in get_critical_point:
            initial_points.append(point)
            # pts.append(point)
            which_polytope = get_polytope_at(
                A, B, known_A0, known_B0, point, False
            )  # [-1 1 -1]
            hidden_vector = get_hidden_at(A, B, known_A0, known_B0, LAYER, point, False)
            history.append((which_polytope, hidden_vector, np.copy(point)))

    while True:
        if not already_checked_critical_points:
            history = []
            pts = []

        prev_count = -10
        good = False
        while len(pts) > prev_count + 2:
            print("======" * 10)
            print("RESTART SEARCH", len(pts), prev_count)
            # print(which_point)
            prev_count = len(pts)
            more_points, done = follow_hyperplane(
                args,
                LAYER,
                which_point,
                A,
                B,
                known_A0,
                known_B0,
                model,
                dead_neurons,
                dimInput,
                dimOfPrevLayer,
                dimOfLayer,
                special,
                history=history,
                # MAX_POINTS=MAX_POINTS // 2,
                MAX_POINTS=MAX_POINTS,
                only_need_positive=only_need_positive,
                target_neuron=target_neuron,
            )
            pts.extend(more_points)
            initial_points.extend(more_points)
            print("After adding new points, the length of pts is", len(pts))
            target_neuron = None

            if len(pts) != 0:
                new_A, new_B = A + [known_A0], B + [known_B0]
                neuron_values = forward(pts, new_A, new_B)

                neuron_positive_count = np.sum(neuron_values > 1e-5, axis=0)
                neuron_negative_count = np.sum(neuron_values < -1e-5, axis=0)
                print("Counts (pts)")
                print("neuron_positive_count", neuron_positive_count)
                print("neuron_negative_count", neuron_negative_count)

                if (
                    np.all(neuron_positive_count > 0)
                    and np.all(neuron_negative_count > 0)
                    or (only_need_positive and np.all(neuron_positive_count > 0))
                ):
                    print("Have all the points we need (2)")
                    good = True
                    break

            print("SHOULD BE DONE?", done, only_need_positive)
            if done and only_need_positive:
                print("All weights are recovered")
                good = True
                break

            if len(pts) >= MAX_POINTS:
                print("Have enough; break")
                break

            if len(pts) == 0:
                break

            # if done and only_need_positive:
            #     good = True
            #     break
            # if (
            #     np.all(neuron_positive_count > 0)
            #     and np.all(neuron_negative_count > 0)
            #     or (only_need_positive and np.all(neuron_positive_count > 0))
            # ):
            #     print("Have all the points we need (2)")
            #     good = True
            #     break
        # breakpoint()

        # if len(pts) < MAX_POINTS / 2 and good == False:
        if len(pts) < MAX_POINTS and good == False:
            print("=======" * 10)
            print("Select a new point to start from")
            print("=======" * 10)
            if already_checked_critical_points:
                print("CHOOSE FROM", len(initial_points))  # , initial_points)
                if len(initial_points) >= 2:
                    which_point = initial_points[
                        np.random.randint(0, len(initial_points) - 1)
                    ]
                else:
                    which_point = initial_points[0]
            else:
                which_point = next(get_critical_point)
        else:
            print("Abort")
            break

    critical_points = np.array(pts)  # sorted(list(set(map(tuple,pts))))

    print("Now have critical points", len(critical_points))

    return critical_points
