import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict

class HybridMerger:
    def __init__(self, alpha: float, beta: float):
        """
        Initializes the Hybrid Merger.
        Alpha: Weight for TextRank Score.
        Beta: Weight for Semantic Similarity.
        """
        self.alpha = alpha
        self.beta = beta
        print(">> [HybridMerger] Loading Embedding Model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def merge_scores(self, 
                     original_sentences: List[str], 
                     textrank_scores: np.ndarray, 
                     llm_summary: str) -> List[Dict]:
        
        N = len(original_sentences)
        original_embeddings = self.model.encode(original_sentences, convert_to_tensor=True)
        llm_embedding = self.model.encode(llm_summary, convert_to_tensor=True)
        
        semantic_scores = util.cos_sim(original_embeddings, llm_embedding).cpu().numpy().flatten()
        
        t_min, t_max = np.min(textrank_scores), np.max(textrank_scores)
        if t_max > t_min:
            norm_textrank = (textrank_scores - t_min) / (t_max - t_min)
        else:
            norm_textrank = np.zeros_like(textrank_scores)

        merged_results = []
        for i in range(N):
            final_score = (self.alpha * norm_textrank[i]) + (self.beta * semantic_scores[i])
            
            merged_results.append({
                "index": i,
                "text": original_sentences[i],
                "textrank_score": float(norm_textrank[i]),
                "semantic_score": float(semantic_scores[i]),
                "final_score": float(final_score)
            })
            
        return merged_results

    def get_top_n(self, merged_results: List[Dict], n: int = 3) -> str:
        """
        Sorts by Final Score and reconstructs the summary.
        Preserves original order (Phase 1, Pg 9, Line 149).
        """
        # Sort by score descending
        sorted_by_score = sorted(merged_results, key=lambda x: x['final_score'], reverse=True)
        
        # Pick top N
        top_candidates = sorted_by_score[:n]
        
        # Sort by index ascending (to maintain flow)
        top_candidates_ordered = sorted(top_candidates, key=lambda x: x['index'])
        
        # Join
        summary = " ".join([item['text'] for item in top_candidates_ordered])
        return summary