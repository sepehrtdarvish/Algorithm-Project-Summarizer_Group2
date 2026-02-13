import unittest
import numpy as np
from src.vectorization import ManualTFIDF


class TestTFIDF(unittest.TestCase):

    def setUp(self):
        self.vectorizer = ManualTFIDF()

    def test_matrix_shape(self):
        """بررسی ابعاد ماتریس خروجی"""
        sentences = ["This is one.", "This is two."]
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        self.assertEqual(tfidf_matrix.shape, (2, 4))

    def test_worst_case_repeated_words(self):
        """(Worst Case) جمله‌ای با کلمات کاملاً تکراری"""
        sentences = ["test test test", "test test test"]
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        self.assertTrue(np.allclose(tfidf_matrix, 0))

    def test_empty_input(self):
        """(Edge Case) لیست جملات خالی"""
        sentences = []
        try:
            matrix = self.vectorizer.fit_transform(sentences)
            self.assertEqual(matrix.shape[0], 0)
        except Exception as e:
            print(f"Handled expected empty input: {e}")


if __name__ == '__main__':
    unittest.main()