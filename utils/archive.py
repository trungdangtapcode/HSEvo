import numpy as np
import matplotlib.pyplot as plt
import random
# plt.style.use('seaborn-v0_8-whitegrid')
from .heap import LimitedHeap
from datetime import datetime
import os

class MAPElitesArchive:
    def __init__(self, feature_dims, bins_per_dim):
        """
        Args:
            feature_dims (int): Number of behavior dimensions (features).
            bins_per_dim (list of int): Number of bins in each dimension.
            fitness_func (callable): Function to evaluate fitness of a solution.
            behavior_func (callable): Function to extract behavioral descriptors.
        """
        assert feature_dims==2
        self.feature_dims = feature_dims
        self.bins_per_dim = [bins_per_dim]*feature_dims

        # Storage: each cell stores a tuple (solution, fitness)
        self.archive = np.empty(self.bins_per_dim, dtype=object)
        self.archive_full = np.empty(self.bins_per_dim, dtype=object)
        limit_size_each_behavior = 100
        self.archive_full.fill(LimitedHeap(limit_size_each_behavior))

        self.trash = []
        self.trash_size = 1000

    def get_index(self, behavior):
        """Convert behavior descriptor into index in archive grid."""
        index = []
        for i, b in enumerate(behavior):
            # Assuming behavior values are normalized in [0, 1]
            bin_idx = int(np.clip(b * self.bins_per_dim[i], 0, self.bins_per_dim[i] - 1))
            index.append(bin_idx)
            
        return np.array(index)

    def get_elite(self, behavior):
        "get elite's fitness for a behavior"
        idx = self.get_index(behavior)
        entry = self.archive[tuple(idx)]
        if entry is None:
            return float("inf")
        else:
            return entry[1]
        
    def get_list_solution(self, behavior, distance = 2):
        """Input is behavior, output is list of solutions
        Behavior that have distance near to input behavior will be added to list then sort
        """
        raise NotImplementedError("Not implemented yet")
        idx = self.get_index(behavior)
        entry = self.archive[tuple(idx)]
        if entry is None:
            return []
        else:
            elite = entry[0]
            elite_fitness = entry[1]
            elite_behavior = entry[2]
            # get all solutions in archive
            all_solutions = [entry for entry in self.archive.flatten() if entry is not None]
            # filter by distance
            filtered_solutions = [sol for sol in all_solutions if np.linalg.norm(np.array(sol[2]) - np.array(elite_behavior)) < distance]
            # sort by fitness
            filtered_solutions.sort(key=lambda x: x[1])
            return filtered_solutions

    def add(self, solution, fitness, behavior: tuple):
        """Evaluate and add a solution to the archive if it's better."""
        idx = self.get_index(behavior)
        
        current_entry = self.archive[tuple(idx)]
        if current_entry is None or fitness < current_entry[1]:
            if current_entry is not None: 
                self.add_trash(current_entry)
            self.archive[tuple(idx)] = (solution, fitness, behavior)
            return True
        else:
            self.add_trash((solution, fitness, behavior))

        return False
    
    def sample_elite(self):
        """Sample an elite solution from the archive."""
        # Flatten the archive and filter out None entries
        non_empty_cells = [entry for entry in self.archive.flatten() if entry is not None]
        
        if len(non_empty_cells) == 0:
            return None
        
        non_empty_cells = np.array(non_empty_cells, dtype=object)
        
        # Randomly select one of the non-empty cells
        selected_entry = random.choice(non_empty_cells)
        
        return selected_entry[0], selected_entry[1], selected_entry[2]

    def sample_non_elite(self):
        """Sample a non-elite solution from the archive."""
        # Flatten the archive and filter out None entries
        if (len(self.trash) == 0):
            return None
        trash = self.trash

        return random.choice(trash)        

    def add_trash(self, entry):
        """Add a solution to the trash."""
        if len(self.trash) >= self.trash_size:
            pos = np.random.randint(0, self.trash_size//2)
            self.trash.pop(pos)
        self.trash.append(entry)

    def get_elites(self):
        """Return all non-empty elite solutions."""
        return [entry for entry in self.archive.flatten() if entry is not None]

    def size(self):
        """Return number of filled cells in archive."""
        return sum(1 for cell in self.archive.flatten() if cell is not None)
        
    def print_elites(self):
        """print elites (dim=2)"""
        print(self.archive)
    
    def save_img(self):
        """save img of archive (dim=2)"""
        def get_fitness(x, y):
            entry = self.archive[x, y]
            if entry is None:
                return float("inf")
            else:
                return entry[1]
        x = np.arange(self.bins_per_dim[0]) 
        y = np.arange(self.bins_per_dim[1])
        X, Y = np.meshgrid(x, y)
        Z = np.clip(-np.vectorize(get_fitness)(X, Y),-10,10)
        plt.contourf(X, Y, Z)
        # plt.imshow(self.archive)
        plt.colorbar()

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        savepath = os.path.join(os.path.abspath(__file__),'..','..', "outputs", "plots", f"plot_{current_time}.png")
        plt.savefig(savepath)
        plt.close()  # Optional: closes the current figure to free memory
        
        # plt.show()
    

    def __getitem__(self, idx):
        """Access a specific cell of the archive."""
        return self.archive[idx]



archive = MAPElitesArchive(2,10)

# archive.add("concac",2,(.1, 1.3))
# archive.add("concac",-10,(1.1, 1.3))
# # print(archive.get_elites())
# # archive.print_elites()
# # archive.save_img()
# archive.add("concac",3,(.1, 1.3))

# print(archive.sample_non_elite())