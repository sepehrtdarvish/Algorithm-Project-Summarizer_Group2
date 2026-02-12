# src/classic_strategies.py
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
        # استفاده از متد fit_transform برای محاسبه TF و واژگان
        _ = vectorizer.fit_transform(sentences)
        
        # محاسبه فراوانی کلی هر کلمه در کل داکیومنت (از روی vocab و شمارش ساده)
        # برای سادگی از tf_matrix داخلی vectorizer استفاده نمی‌کنیم و مستقیم می‌شماریم
        word_freq = {}
        tokenized_sentences = [vectorizer._tokenize(s) for s in sentences]
        
        # ساخت جدول فراوانی
        all_tokens = [w for s in tokenized_sentences for w in s]
        total_tokens = len(all_tokens)
        for w in all_tokens:
            word_freq[w] = word_freq.get(w, 0) + 1
            
        scores = []
        for tokens in tokenized_sentences:
            if not tokens:
                scores.append(0.0)
                continue
            # امتیاز = مجموع احتمال وقوع کلمات / طول جمله
            sent_score = sum(word_freq.get(w, 0) for w in tokens) / len(tokens)
            scores.append(sent_score)
            
        return np.array(scores)

class SentenceRankingStrategy(SummarizationStrategy):
    """
    پیاده سازی روش Sentence Ranking (Heuristic).
    ترکیبی از: موقعیت جمله (اول متن مهم‌تر است) + طول جمله.
    """
    def calculate_scores(self, sentences: List[str], config: dict) -> np.ndarray:
        n = len(sentences)
        scores = np.zeros(n)
        
        for i, sent in enumerate(sentences):
            # 1. Position Score: جملات اول امتیاز ۱ می‌گیرند و به تدریج کم می‌شود
            pos_score = 1.0 / (i + 1)
            
            # 2. Length Score: جملات خیلی کوتاه جریمه می‌شوند (نرمال شده)
            # فرض ساده: طول بیشتر (تا حدی) بهتر است
            len_score = min(len(sent.split()), 20) / 20.0
            
            # ترکیب وزن‌دار (قابل تنظیم)
            scores[i] = (0.7 * pos_score) + (0.3 * len_score)
            
        return scores

class GreedyStrategy(SummarizationStrategy):
    """
    پیاده سازی نسخه امتیازی Greedy (Centroid-based).
    جملاتی که بیشترین شباهت را به 'بردار میانگین کل متن' دارند، امتیاز بالاتر می‌گیرند.
    این استراتژی معادل انتخاب جملاتی است که نماینده کل متن هستند.
    """
    def calculate_scores(self, sentences: List[str], config: dict) -> np.ndarray:
        vectorizer = ManualTFIDF()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # محاسبه بردار میانگین (Centroid) کل متن
        doc_centroid = np.mean(tfidf_matrix, axis=0)
        
        # محاسبه شباهت کسینوسی هر جمله با Centroid
        scores = []
        norm_centroid = np.linalg.norm(doc_centroid)
        if norm_centroid == 0:
            return np.zeros(len(sentences))
            
        for vec in tfidf_matrix:
            norm_vec = np.linalg.norm(vec)
            if norm_vec == 0:
                scores.append(0.0)
            else:
                sim = np.dot(vec, doc_centroid) / (norm_vec * norm_centroid)
                scores.append(sim)
                
        return np.array(scores)

# Factory برای ساخت راحت کلاس‌ها
def get_strategy(method_name: str) -> SummarizationStrategy:
    strategies = {
        'textrank': TextRankStrategy(),
        'frequency': FrequencyStrategy(),
        'ranking': SentenceRankingStrategy(),
        'greedy': GreedyStrategy()
    }
    if method_name not in strategies:
        raise ValueError(f"Unknown classic method: {method_name}. Available: {list(strategies.keys())}")
    return strategies[method_name]