# src/textrank.py
import numpy as np

def run_pagerank(graph: np.ndarray, d: float = 0.85, max_iter: int = 50, tol: float = 1e-4) -> np.ndarray:
    """
    Manual implementation of PageRank.
    Phase 1, Pg 7, Lines 105-120.
    """
    N = graph.shape[0]
    
    # Initialize scores to 1.0 (or 1/N)
    scores = np.ones(N)
    
    # Pre-calculate total output weights for each node (sum of columns)
    # Note: In undirected text graphs, adjacency is usually symmetric.
    # W_ji is weight from j to i.
    
    for iteration in range(max_iter):
        prev_scores = np.copy(scores)
        
        for i in range(N):
            sum_neighbors = 0
            
            # Find neighbors pointing to i (in undirected, just row/col i)
            # We iterate over all j to find incoming edges to i
            for j in range(N):
                if graph[j, i] > 0: # If there is an edge from j to i
                    weight_ji = graph[j, i]
                    total_weight_j = np.sum(graph[j])
                    
                    if total_weight_j > 0:
                         sum_neighbors += (weight_ji / total_weight_j) * prev_scores[j]
            
            # PageRank Formula: (1-d) + d * sum(...)
            scores[i] = (1 - d) + (d * sum_neighbors)
            
        # Check convergence
        diff = np.sum(np.abs(scores - prev_scores))
        if diff < tol:
            break
            
    return scores