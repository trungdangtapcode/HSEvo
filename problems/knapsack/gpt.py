import numpy as np
import random
import math
import scipy
import torch
def heuristics_v2(num_items: int, capacity: int, items: np.ndarray) -> np.ndarray:
    # items: ndarray of shape (num_items, 2), with columns [value, weight]
    
    # Compute value-to-weight ratio and store original indices
    indices = np.arange(num_items)
    ratios = items[:, 0] / items[:, 1]
    
    # Sort by ratio descending
    sorted_indices = np.argsort(-ratios)
    
    total_weight = 0
    res = np.zeros(num_items, dtype=int)
    
    for idx in sorted_indices:
        value, weight = items[idx]
        if total_weight + weight <= capacity:
            total_weight += weight
            res[idx] = 1
    
    return res

