import numpy as np

def run_pagerank(graph: np.ndarray, d: float = 0.85, max_iter: int = 50, tol: float = 1e-4) -> np.ndarray:
    """
    Manual implementation of PageRank.
    Phase 1, Pg 7, Lines 105-120.
    """
    N = graph.shape[0]
    
    scores = np.ones(N)
    
    for iteration in range(max_iter):
        prev_scores = np.copy(scores)
        
        for i in range(N):
            sum_neighbors = 0
            
            for j in range(N):
                if graph[j, i] > 0:
                    weight_ji = graph[j, i]
                    total_weight_j = np.sum(graph[j])
                    
                    if total_weight_j > 0:
                         sum_neighbors += (weight_ji / total_weight_j) * prev_scores[j]
            
            scores[i] = (1 - d) + (d * sum_neighbors)
            
        # Check convergence
        diff = np.sum(np.abs(scores - prev_scores))
        if diff < tol:
            break
            
    return scores