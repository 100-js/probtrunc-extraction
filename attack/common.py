import math
import random

import numpy as np


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)


def find_rank_full_sub_indices(matrix, target_size):
    sub_indices = []
    for i in range(matrix.shape[0]):
        test_indices = sub_indices + [i]
        if np.linalg.matrix_rank(matrix[test_indices, :]) == len(test_indices):
            sub_indices.append(i)
        if len(sub_indices) == target_size:
            break

    if len(sub_indices) < target_size:
        raise ValueError("Insufficient matrix rank to construct a full-rank sub-matrix.")

    return sub_indices


def primitive_vector(diff):
    diff = np.asarray(diff)

    if np.all(diff == 0):
        return None

    if diff[np.argmax(diff != 0)] < 0:
        diff = -diff

    gcd_val = abs(int(diff[0]))
    for val in diff[1:]:
        gcd_val = math.gcd(gcd_val, abs(int(val)))
        if gcd_val == 1:
            break

    if gcd_val <= 0:
        return None

    return diff // gcd_val