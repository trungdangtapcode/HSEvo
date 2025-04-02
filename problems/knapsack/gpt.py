import numpy as np

def heuristics_v2(num_items, capacity, items):
    """
    :param num_items: Number of items
    :param capacity: Capacity of the knapsack
    :param items: List of tuples (value, weight) with shape (num_items, 2)
    :return: Total value of the selected items
    """
    total_value = 0
    total_weight = 0
    
    res = []
    for value, weight in items:
        if total_weight + weight <= capacity:
            total_weight += weight
            total_value += value
            res.append(1)
        else: res.append(0)
    
    return np.array(res)