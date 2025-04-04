import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('seaborn-v0_8-whitegrid')

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

    def get_index(self, behavior):
        """Convert behavior descriptor into index in archive grid."""
        index = []
        for i, b in enumerate(behavior):
            # Assuming behavior values are normalized in [0, 1]
            bin_idx = int(np.clip(b * self.bins_per_dim[i], 0, self.bins_per_dim[i] - 1))
            index.append(bin_idx)
            
        return np.array(index)

    def add(self, solution, fitness, behavior):
        """Evaluate and add a solution to the archive if it's better."""
        idx = self.get_index(behavior)
        
        current_entry = self.archive[tuple(idx)]
        if current_entry is None or fitness > current_entry[1]:
            self.archive[tuple(idx)] = (solution, fitness, behavior)

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
                return 0
            else:
                return entry[1]
        x = np.arange(self.bins_per_dim[0]) 
        y = np.arange(self.bins_per_dim[1])
        X, Y = np.meshgrid(x, y)
        plt.contourf(X, Y, np.vectorize(get_fitness)(X, Y))
        # plt.imshow(self.archive)
        plt.colorbar()
        plt.show()
    

    def __getitem__(self, idx):
        """Access a specific cell of the archive."""
        return self.archive[idx]



# archive = MAPElitesArchive(2,10)

# # archive.add("concac",2,(.1, 1.3))
# # archive.add("concac",10,(1.1, 1.3))
# # print(archive.get_elites())
# # archive.print_elites()
# archive.save_img()