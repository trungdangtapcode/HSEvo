import sys
import numpy as np
import logging
from os import path, listdir
import gpt
import inspect
from ortools.algorithms.python import knapsack_solver

def get_heuristic_name(module, possible_names: list[str]):
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name

possible_func_names = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]

heuristic_name = get_heuristic_name(gpt, possible_func_names)
heuristics = getattr(gpt, heuristic_name)

def generate_datasets():
    """Generates datasets for training, validation, and testing of the knapsack heuristic."""
    basepath = path.join(path.dirname(__file__), "dataset")
    # basepath = "/kaggle/working/datasets"
    np.random.seed(42)
    
    dataset_sizes = {
        'train': [10000],
        'val': [2000, 5000, 100000],
        'test': [2000, 5000, 100000]
    }
    
    for mood, sizes in dataset_sizes.items():
        for num_items in sizes:
            capacities = np.random.randint(50, 10*num_items)
            values = np.random.randint(1, 100, num_items)
            weights = np.random.randint(1, 50, num_items)
            items = np.stack([values, weights], axis=1)

            instances = np.concatenate([np.array([[num_items, capacities]]), items], axis=0)
            
            dataset_path = path.join(basepath, f"{mood}{num_items}_dataset.npy")
            np.save(dataset_path, instances)
            print(f"[*] Dataset saved: {dataset_path} with {num_items} instances.")


def heuristic2(num_items, capacity, items):
    """
    Greedy heuristic for the 0/1 Knapsack problem based on value-to-weight ratio.
    
    :param num_items: Number of items
    :param capacity: Capacity of the knapsack
    :param items: List of tuples (value, weight) with shape (num_items, 2)
    :return: Total value of the selected items
    """
    # Sort items by value-to-weight ratio in descending order
    items = sorted(items, key=lambda x: x[0] / x[1], reverse=True)
    
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

def read_knapsack_file(filename):
    packed_items = []
    packed_weights = []
    total_weight = 0

    with open(filename, 'r') as f:
        f.readline()
        n = int(f.readline().strip())  # Read number of items
        c = int(f.readline().strip())  # Read capacity
        f.readline()
        
        for line in f:
            p, w = map(int, line.split())
            packed_items.append(p)
            packed_weights.append(w)
            total_weight += w

    return packed_items, [packed_weights], c

def solve(num_items, capacity, items):
    """Solves the knapsack instance using the heuristic function."""
    # print(instance.shape)
    # num_items, capacity = instance[0]
    # items = instance[1:]
    # print(capacity, np.sum(items[:,1]))
    choose = heuristics(num_items, capacity, items)
    s = 0
    c = capacity
    # print(choose)
    for i in range(num_items):
        s += items[i][0]*choose[i]
        c -= items[i][1]*choose[i]
        assert c>=0
    return s

if __name__ == "__main__":
    print("[*] Running ...")
    
    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    # assert mood in ['train', 'val']

    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "KnapsackExample",
    )
    
    basepath = path.join(path.dirname(__file__), "datasets/")
    # basepath = "/kaggle/working/datasets"
    # if not path.isfile(path.join(basepath, "train10000_dataset.npy")):
    #     generate_datasets()
    
    if mood == 'train' or True:
        objs = []
        for dataname in sorted(listdir(basepath))[:50]:
            data_path = path.join(basepath, dataname)
            values, weights, capacities = read_knapsack_file(data_path)
            values = np.expand_dims(values,axis=0)
            weights = np.array(weights)
            items = np.concatenate([values, weights], axis = 0).T
            n = items.shape[0]
            obj = solve(n, capacities, items)
            objs.append(obj)
        print("[*] Average:")
        print(np.mean(objs))
            
    
    else:
        objs = []
        for dataname in sorted(listdir(basepath))[25:]:
            data_path = path.join(basepath, dataname)
            values, weights, capacities = read_knapsack_file(data_path)
            values = np.expand_dims(values,axis=0)
            weights = np.array(weights)
            items = np.concatenate([values, weights], axis = 0).T
            n = items.shape[0]
            obj = solve(n, capacities, items)
            objs.append(obj)
        print(f"[*] Average for {problem_size}: {np.mean(objs)}")
        print(np.mean(objs))
        # for size in [2000, 5000, 100000]:
        #     dataset_path = path.join(basepath, f"{mood}{size}_dataset.npy")
        #     instances = np.load(dataset_path, allow_pickle=True)
        #     logging.info(f"[*] Evaluating {dataset_path}")
        #     objs = solve(instances)
        #     print(f"[*] Average for {problem_size}: {np.mean(objs)}")