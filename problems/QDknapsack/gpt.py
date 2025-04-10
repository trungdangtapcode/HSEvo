import numpy as np
import random
import math
import scipy
import torch
def heuristics_v2(num_items: int, capacity: int, items: np.ndarray) -> np.ndarray:
    # Sort items by value-to-weight ratio in descending order
    ratios = items[:, 0] / items[:, 1]
    sorted_indices = np.argsort(ratios)[::-1]
    sorted_items = items[sorted_indices]

    total_value = 0
    total_weight = 0
    res = [0] * num_items

    # Sparsification: Ignore items that contribute minimally to overall utility based on density and value-to-weight ratio
    threshold = np.percentile(sorted_items[:, 0], 20)  # adjust the percentile as needed
    for i, (value, weight) in enumerate(sorted_items):
        if value < threshold:
            break

        if total_weight + weight <= capacity:
            total_weight += weight
            total_value += value
            res[sorted_indices[i]] = 1
        else:
            fraction = (capacity - total_weight) / weight
            total_value += fraction * value
            res[sorted_indices[i]] = fraction
            break

    # Fine-tuning based on remaining capacity, item density, and value-to-weight ratio
    i += 1
    while i < len(sorted_items) and total_weight + sorted_items[i][1] <= capacity:
        if total_weight + sorted_items[i][1] > capacity / 2:  # consider fractional knapsack scenarios
            fraction = (capacity - total_weight) / sorted_items[i][1]
            total_value += fraction * sorted_items[i][0]
            res[sorted_indices[i]] = fraction
            break
        else:
            if sorted_items[i][0] > capacity - total_weight:  # prioritize high-value items
                fraction = (capacity - total_weight) / sorted_items[i][1]
                total_value += fraction * sorted_items[i][0]
                res[sorted_indices[i]] = fraction
                break
            else:
                total_weight += sorted_items[i][1]
                total_value += sorted_items[i][0]
                res[sorted_indices[i]] = 1
                i += 1

    return np.array(res).astype(int)