import numpy as np

def select_next_node_v2(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    """
    Select the next node using a heuristic inspired by the principles of alternating current.  
    We aim for a balanced path, avoiding excessively long jumps while prioritizing proximity to the destination.
    """

    # Initialize scores.  Think of these as potentials in an electrical field.
    scores = {}

    for node in unvisited_nodes:
        # Direct distance to next node.  A strong attractive force.
        direct_distance = distance_matrix[current_node][node]

        # Distance to destination.  A guiding force, stronger the closer we are to the end.
        destination_distance = distance_matrix[node][destination_node]

        # A repulsive force, proportional to the average distance to other unvisited nodes.
        # Avoids getting trapped in clusters.
        repulsion = np.mean([distance_matrix[node][n] for n in unvisited_nodes if n != node]) if len(unvisited_nodes) > 1 else 0

        #  The heuristic combines attractive and repulsive forces. The weights are chosen empirically
        # to reflect the relative importance of each factor.  Imagine tuning the capacitance and
        # inductance in an AC circuit to achieve optimal resonance.

        score = 0.6 * (1 / direct_distance) + 0.3 * (1/ destination_distance) - 0.1 * repulsion

        scores[node] = score

    # Select the node with the highest "potential".
    next_node = max(scores, key=scores.get)
    return next_node
