import numpy as np
from abc import ABC, abstractmethod
from typing import List
from src.vectorization import ManualTFIDF
from src.graph_utils import calculate_cosine_similarity_matrix, build_graph
from src.textrank import run_pagerank

class SummarizationStrategy(ABC):
    """
    کلاس پایه انتزاعی برای تمام استراتژی‌های خلاصه سازی کلاسیک.
    همه روش‌ها باید خروجی امتیاز (Score) برای تمام جملات برگردانند
    تا در مرحله Hybrid Merge قابل استفاده باشند.
    """
    @abstractmethod
    def calculate_scores(self, sentences: List[str], config: dict) -> np.ndarray:
        pass

class TextRankStrategy(SummarizationStrategy):
    """
    پیاده سازی روش گراف مبنا (TextRank).
    """
    def calculate_scores(self, sentences: List[str], config: dict) -> np.ndarray:
        # 1. Vectorization
        vectorizer = ManualTFIDF()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # 2. Graph Construction
        sim_matrix = calculate_cosine_similarity_matrix(tfidf_matrix)
        graph = build_graph(sim_matrix, config['textrank']['similarity_threshold'])
        
        # 3. PageRank
        scores = run_pagerank(
            graph, 
            d=config['textrank']['damping_factor'],
            max_iter=config['textrank']['max_iterations'],
            tol=config['textrank']['convergence_threshold']
        )
        return scores

class FrequencyStrategy(SummarizationStrategy):
    """
    پیاده سازی روش Frequency-based.
    امتیاز جمله = میانگین فراوانی کلمات تشکیل دهنده آن.
    """
    def calculate_scores(self, sentences: List[str], config: dict) -> np.ndarray:
        vectorizer = ManualTFIDF()
        _ = vectorizer.fit_transform(sentences)
        
        word_freq = {}
        tokenized_sentences = [vectorizer._tokenize(s) for s in sentences]
        
        all_tokens = [w for s in tokenized_sentences for w in s]
        total_tokens = len(all_tokens)
        for w in all_tokens:
            word_freq[w] = word_freq.get(w, 0) + 1
            
        scores = []
        for tokens in tokenized_sentences:
            if not tokens:
                scores.append(0.0)
                continue
            sent_score = sum(word_freq.get(w, 0) for w in tokens) / len(tokens)
            scores.append(sent_score)
            
        return np.array(scores)

def get_strategy(method_name: str) -> SummarizationStrategy:
    strategies = {
        'textrank': TextRankStrategy(),
        'frequency': FrequencyStrategy(),
    }
    if method_name not in strategies:
        raise ValueError(f"Unknown classic method: {method_name}. Available: {list(strategies.keys())}")
    return strategies[method_name]