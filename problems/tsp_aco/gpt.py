import numpy as np
import random
import math
import scipy
import torch
def heuristics_v2(distance_matrix: np.ndarray, distance_exponent: float = 2.274971017621666, neighbor_weight: float = 1.8088439025394076) -> np.ndarray:
    """
    Prioritizes edges based on a combination of distance and the inverse of the number of neighboring nodes.
    """
    n = distance_matrix.shape[0]
    neighbor_count = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            neighbor_count[i, j] = neighbor_count[j, i] = 1
    neighbor_count = np.sum(neighbor_count, axis=0)
    # Prioritize edges connecting nodes with fewer neighbors
    neighbor_factor = 1 / (neighbor_count + 1)
    # Higher priority to shorter edges (inverse square law)
    distance_factor = 1 / (distance_matrix ** distance_exponent)
    # Combine both factors for a comprehensive score
    return distance_factor * (neighbor_factor ** neighbor_weight)
