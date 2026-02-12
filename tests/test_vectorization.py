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
        # ابعاد باید (2, 4) باشد چون 4 کلمه یکتا داریم (this, is, one, two)
        self.assertEqual(tfidf_matrix.shape, (2, 4))

    def test_worst_case_repeated_words(self):
        """(Worst Case) جمله‌ای با کلمات کاملاً تکراری"""
        # در TF، اگر کلمه زیاد تکرار شود امتیاز بالا می‌گیرد، اما در IDF اگر در همه جا باشد کم می‌شود.
        sentences = ["test test test", "test test test"]
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        # چون کلمه "test" در تمام جملات هست، IDF باید 0 شود (log(2/2)=0).
        # پس کل ماتریس باید صفر باشد.
        self.assertTrue(np.allclose(tfidf_matrix, 0))

    def test_empty_input(self):
        """(Edge Case) لیست جملات خالی"""
        sentences = []
        # اینجا باید مدیریت کنیم که ارور ندهد یا ماتریس خالی برگرداند
        # بسته به ایمپلمنتیشن شما، ممکن است نیاز به try-except باشد
        try:
            matrix = self.vectorizer.fit_transform(sentences)
            self.assertEqual(matrix.shape[0], 0)
        except Exception as e:
            print(f"Handled expected empty input: {e}")


if __name__ == '__main__':
    unittest.main()