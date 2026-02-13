import numpy as np

def calculate_cosine_similarity_matrix(tfidf_matrix: np.ndarray) -> np.ndarray:
    """
    Calculates the Cosine Similarity between all pairs of sentences.
    Phase 1, Pg 6, Lines 75-86.
    Returns: N x N similarity matrix.
    """
    # Compute dot product
    dot_product = np.dot(tfidf_matrix, tfidf_matrix.T)
    
    # Compute magnitudes (norms)
    norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    
    # Avoid division by zero
    norms[norms == 0] = 1e-10
    
    # Calculate cosine similarity
    similarity_matrix = dot_product / (norms @ norms.T)
    
    # Fill diagonal with 0 (a sentence doesn't vote for itself in standard PageRank)
    np.fill_diagonal(similarity_matrix, 0)
    
    return similarity_matrix

def build_graph(similarity_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """
    Filters the similarity matrix based on a threshold to create a graph adjacency matrix.
    Phase 1, Pg 7, Lines 96-101.
    """
    graph = np.copy(similarity_matrix)
    graph[graph < threshold] = 0
    return graph