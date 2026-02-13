import numpy as np
from typing import List, Tuple, Dict
import math

class ManualTFIDF:
    """
    Implements TF-IDF vectorization based on the logic provided in Phase 1 Document.
    """
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf_vector: np.ndarray = None

    def _tokenize(self, sentence: str) -> List[str]:
        """Simple word tokenization (lowercase, remove punctuation logic)."""
        import string
        translator = str.maketrans('', '', string.punctuation)
        return sentence.translate(translator).lower().split()

    def fit_transform(self, sentences: List[str]) -> np.ndarray:
        """
        Calculates TF, IDF, and returns the TF-IDF matrix.
        Output Shape: (N_Sentences, N_Vocab)
        """
        tokenized_sentences = [self._tokenize(s) for s in sentences]
        
        # 1. Build Vocabulary
        unique_words = set(word for s in tokenized_sentences for word in s)
        self.vocab = {word: i for i, word in enumerate(sorted(unique_words))}
        n_sentences = len(sentences)
        n_vocab = len(self.vocab)

        # 2. Calculate TF (Term Frequency)
        tf_matrix = np.zeros((n_sentences, n_vocab))
        for i, sent in enumerate(tokenized_sentences):
            sent_len = len(sent)
            if sent_len == 0: continue
            for word in sent:
                if word in self.vocab:
                    tf_matrix[i, self.vocab[word]] += 1
            # Normalize TF by sentence length (Phase 1, Pg 6, Line 67)
            tf_matrix[i] = tf_matrix[i] / sent_len

        # 3. Calculate IDF (Inverse Document Frequency)
        # Phase 1, Pg 6, Line 74: Log(Total / doc_count)
        doc_counts = np.zeros(n_vocab)
        for sent in tokenized_sentences:
            seen_words = set(sent)
            for word in seen_words:
                if word in self.vocab:
                    doc_counts[self.vocab[word]] += 1
        
        # Avoid division by zero
        doc_counts = np.maximum(doc_counts, 1)
        self.idf_vector = np.log(n_sentences / doc_counts)

        # 4. Calculate TF-IDF
        # Broadcasting: TF * IDF
        tfidf_matrix = tf_matrix * self.idf_vector
        
        return tfidf_matrix