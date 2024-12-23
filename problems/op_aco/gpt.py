import numpy as np

def heuristics_v2(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> np.ndarray:
    n = len(prize)
    heuristics = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Reward based on prize, distance and remaining budget.
            reward = prize[j] / distance[i, j]
            
            # Punish long edges disproportionately
            if distance[i,j] > maxlen *0.6:
                reward *= 0.1

            # Boost edges to nodes with high prize-to-distance ratio among neighbors.
            neighbor_rewards = prize[np.arange(n) != i] / distance[i, np.arange(n) != i]
            neighbor_rewards[np.isinf(neighbor_rewards)] = 0  #Handle division by zero
            neighbor_rewards[np.isnan(neighbor_rewards)] = 0
            boost = np.max(neighbor_rewards)
            reward *= (1 + 0.5 * boost)

            #Consider return to depot more favorably.
            if j == 0:
                reward *= 2.0
                
            heuristics[i, j] = reward
    return heuristics
