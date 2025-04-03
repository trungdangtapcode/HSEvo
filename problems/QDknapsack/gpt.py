import numpy as np
import random
import math
import scipy
import torch
def heuristics_v2(num_items= 5397.958641027184, capacity= 743828.4830435355, threshold= 7.864131555210442, weight_decay= 0.3973366797164795):
    """
    This function combines elements of the two previous heuristics to solve the 0/1 knapsack problem.
    It first calculates value-to-weight ratios and sorts items by their ratio in descending order,
    then uses a simple greedy approach for items that fit within the capacity,
    and finally adds items with high value-to-weight ratios until no further improvements can be made.

    Parameters:
        num_items (int, optional): The number of items. Defaults to 10.
        capacity (int, optional): The total knapsack capacity. Defaults to 100.
        threshold (float, optional): The minimum ratio for an item to be considered in the second stage. Defaults to 0.5.
        weight_decay (float, optional): The decay rate of weights after each iteration. Defaults to 1e-3.

    Returns:
        np.ndarray: A boolean array indicating whether to take or not take the i-th item.
    """
    
    # Calculate value-to-weight ratios
    ratios = items[:, 0] / items[:, 1]
    
    # Sort items by their ratio in descending order
    sorted_items = np.argsort(ratios)[::-1]
    
    # Initialize selected items and total weight
    selected_items = np.zeros(num_items, dtype=bool)
    total_weight = 0
    
    # Use a simple greedy approach for items that fit within the capacity
    for i in sorted_items:
        if total_weight + items[i, 1] <= capacity:
            total_weight += items[i, 1]
            selected_items[i] = True
    
    # Add items with high value-to-weight ratios until no further improvements can be made
    remaining_capacity = capacity - total_weight
    while remaining_capacity > 0:
        max_ratio = -np.inf
        next_item = None
        for i in sorted_items:
            if not selected_items[i] and items[i, 1] <= remaining_capacity:
                ratio = ratios[i]
                if ratio > max_ratio:
                    max_ratio = ratio
                    next_item = i
        
        if next_item is None:
            break
        
        selected_items[next_item] = True
        total_weight += items[next_item, 1]
        remaining_capacity -= items[next_item, 1]
    
    return selected_items
