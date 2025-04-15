import numpy as np
import random
import math
import scipy
import torch
def priority_v2(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.

    Args:
        item: Size of item to be added to the bin.
        bins_remain_cap: Array of capacities for each bin.

    Return:
        Array of same size as bins_remain_cap with priority score of each bin.
    """
    ratios = item / bins_remain_cap
    log_ratios = np.log(ratios)
    priorities = -log_ratios
    return priorities
