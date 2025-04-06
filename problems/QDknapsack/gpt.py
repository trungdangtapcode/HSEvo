import numpy as np
import random
import math
import scipy
import torch
def heuristics_v2(num_items: int, capacity: int, items: np.ndarray) -> np.ndarray:
    """
    Heuristic for the Knapsack Problem that balances density and value-per-weight ratios with capacity constraints.
    
    This function uses a combination of sparsification, prioritization, and greedy search to maximize total value within a weight limit.
    
    :param num_items: Number of items.
    :param capacity: Total knapsack capacity.
    :param items: A NumPy array containing item values and weights (num_items x 2).
    :return: A NumPy array of the same length as the input lists, where each value is 0 or 1, indicating whether to take or not take the i-th item.
    """
    
    # Calculate density for all items
    densities = items[:, 0] / (items[:, 1] ** 2)
    
    # Sort items in descending order of their densities
    ranking = np.argsort(-densities)
    
    res = [False] * num_items
    
    remaining_capacity = capacity
    selected_items = []
    
    for i in range(num_items):
        if not res[ranking[i]]:
            if items[ranking[i], 1] <= remaining_capacity:
                res[ranking[i]] = True
                remaining_capacity -= items[ranking[i], 1]
                selected_items.append(ranking[i])
    
    # Prioritize high-density items with high value-per-weight ratios
    for i in range(num_items):
        if not res[i]:
            if densities[i] > np.mean(densities[selected_items]) and values_per_weight[i] > np.mean(values_per_weight[selected_items]):
                if items[i, 1] <= remaining_capacity:
                    res[i] = True
                    remaining_capacity -= items[i, 1]
    
    return np.array(res)
