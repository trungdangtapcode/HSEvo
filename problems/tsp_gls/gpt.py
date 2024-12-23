def update_edge_distance_v2(edge_distance, local_opt_tour, edge_n_used):
    n = len(edge_distance)
    result = edge_distance.copy()
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in local_opt_tour or (j, i) in local_opt_tour:
                result[i][j] -= 1
                result[j][i] -= 1
            result[i][j] += edge_n_used[i][j] * 0.1
            result[j][i] += edge_n_used[j][i] * 0.1
    return result
