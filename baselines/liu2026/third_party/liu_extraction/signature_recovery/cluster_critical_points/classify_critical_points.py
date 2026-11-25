import numpy as np
from scipy.linalg import null_space
from ..utils import on_which_hidden_layer, getAllWeightsAndBiases


def identify_good_critical_points(args, model, critical_points):
    # Identify good crtical points
    good_indices = []
    bad_indices = []
    special_sols = []
    ker_spaces = []

    for i, cp in enumerate(critical_points):
        index = cp.index
        ratio = cp.ratio
        point = cp.point
        h_matrix = cp.h_matrix

        print("-------------------------")
        print("Critical point index: ", index)

        layer_id, neuron_id = on_which_hidden_layer(point, model)
        print(
            "[Cheat] This critical point is on layer",
            layer_id + 1,
            "neuron",
            neuron_id + 1,
        )

        if args.layerID == 1:
            print("This is a good critical point with full rank h matrix")
            good_indices.append(index)

            special_sols.append(ratio)
            ker_spaces.append([])
            continue

        column_is_zero = np.mean(np.abs(h_matrix) < 1e-8, axis=0) > 0.5
        if args.debug:
            print("rank of h_matrix is:", np.linalg.matrix_rank(h_matrix))
            print(
                "h_matrix has",
                h_matrix.shape[1] - np.sum(column_is_zero),
                "non-zero columns",
            )

        if np.linalg.matrix_rank(h_matrix) < h_matrix.shape[1] - np.sum(column_is_zero):
            print("This is a bad critical point with small rank h matrix")
            bad_indices.append(index)

            # Compute the kernel space of the h matrix
            h_without_zero = h_matrix[:, ~column_is_zero]
            ker_space_vectors = null_space(h_without_zero)
            insert_position = np.where(column_is_zero)[0]
            zero_row = np.zeros(ker_space_vectors.shape[1])
            for _, position in enumerate(insert_position):
                ker_space_vectors = np.insert(
                    ker_space_vectors,
                    position,
                    zero_row,
                    axis=0,
                )

            special_sols.append(ratio)
            ker_spaces.append(
                ker_space_vectors[
                    :,
                    : h_matrix.shape[1]
                    - np.sum(column_is_zero)
                    - np.linalg.matrix_rank(h_matrix),
                ]
            )
        else:
            print("This is a good critical point with full rank h matrix")
            good_indices.append(index)

            special_sols.append(ratio)
            ker_spaces.append([])

    return good_indices, bad_indices, special_sols, ker_spaces
