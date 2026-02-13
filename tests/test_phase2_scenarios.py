import unittest
import numpy as np
from src.preprocessing import split_into_sentences, filter_sentences
from src.vectorization import ManualTFIDF
from src.textrank import run_pagerank


class TestPhase2Scenarios(unittest.TestCase):

    def run_pipeline(self, text):
        # 1. Preprocessing
        # پاکسازی فضاهای اضافی متن ورودی
        clean_text = text.strip()
        sentences = split_into_sentences(clean_text)
        filtered = filter_sentences(sentences, min_length=10)

        if len(filtered) < 2:
            return filtered, None

        # 2. Vectorization
        tfidf = ManualTFIDF()
        matrix = tfidf.fit_transform(filtered)

        # 3. Graph Building
        sim_matrix = np.dot(matrix, matrix.T)

        np.fill_diagonal(sim_matrix, 0)

        # 4. PageRank
        scores = run_pagerank(sim_matrix, d=0.85)
        return filtered, scores

    def test_01_simple_topic_dominance(self):
        """سناریو ۱: جملات مرتبط باید امتیاز متفاوت بگیرند."""
        text = """
        Python is a programming language used in AI.
        Python is also great for data science projects.
        Bananas are yellow fruits that grow on trees.
        """
        sentences, scores = self.run_pipeline(text)

        variance = np.var(scores)
        self.assertGreater(variance, 0.0, "Scores should not be identical (all 1.0).")

    def test_02_simple_coherence(self):
        """سناریو ۲: چک کردن تعداد خروجی."""
        text = "This is sentence number one exactly. This is sentence number two exactly. This is sentence number three exactly."
        sentences, scores = self.run_pipeline(text)
        self.assertEqual(len(scores), 3)

    # ==========================================
    # 2. تست‌های ورودی متوسط و سخت (واقعی)
    # ==========================================
    def test_03_medium_history_text(self):
        """سناریو ۳: متن تاریخی واقعی."""
        text = """
        The Industrial Revolution was a period of major industrialization.
        It began in Great Britain and quickly spread throughout the world.
        This period saw the mechanization of agriculture and textile manufacturing.
        Steam power became the main source of energy for industry.
        """
        sentences, scores = self.run_pipeline(text)

        self.assertEqual(len(scores), 4)
        self.assertGreater(np.std(scores), 0)

    def test_04_hard_technical_terms(self):
        """سناریو ۴: متن فنی NLP."""
        text = """
        Natural language processing is a subfield of linguistics and artificial intelligence.
        NLP focuses on how to program computers to analyze natural language data.
        The goal is a computer capable of understanding the contents of documents.
        This technology allows for extracting insights from large texts.
        """
        sentences, scores = self.run_pipeline(text)

        self.assertFalse(np.any(np.isnan(scores)))

        self.assertIsNotNone(scores)

    # ==========================================
    # 3. تست‌های Edge Cases
    # ==========================================
    def test_05_edge_empty_input(self):
        """سناریو ۵: ورودی خالی."""
        text = ""
        sentences, scores = self.run_pipeline(text)
        self.assertEqual(len(sentences), 0)

    def test_06_edge_short_sentences(self):
        """سناریو ۶: جملات خیلی کوتاه باید حذف شوند."""
        text = "Hi. No. Go. This is a valid long sentence for testing."
        sentences, scores = self.run_pipeline(text)
        # انتظار: ۳ تای اول حذف شوند و ۱ مورد بماند
        self.assertEqual(len(sentences), 1)

    # ==========================================
    # 4. تست‌های Worst-Case و پایداری
    # ==========================================
    def test_08_worst_repetitive_text(self):
        """سناریو ۸: جملات کاملاً تکراری."""
        text = """
        This is a repeated sentence for testing algorithms.
        This is a repeated sentence for testing algorithms.
        This is a repeated sentence for testing algorithms.
        """
        sentences, scores = self.run_pipeline(text)

        variance = np.var(scores)
        self.assertAlmostEqual(variance, 0.0, places=5)

    def test_10_stability_check(self):
        """سناریو ۱۰: پایداری الگوریتم (اجرای روی متن واقعی)."""
        text = """
        Stability testing verifies that the system works consistently.
        Running the algorithm twice should yield the same results.
        This ensures the code is deterministic.
        """
        _, scores1 = self.run_pipeline(text)
        _, scores2 = self.run_pipeline(text)

        np.testing.assert_array_almost_equal(scores1, scores2)

    def test_07_edge_single_sentence(self):
        """سناریو ۷: متن فقط شامل یک جمله است (گراف تشکیل نمی‌شود)."""
        text = "Only one sentence exists in this entire text."
        sentences, scores = self.run_pipeline(text)

        self.assertIsNone(scores)

    def test_09_worst_no_punctuation(self):
        """سناریو ۹: متن طولانی بدون نقطه (چالش Tokenizer)."""
        text = "This is a very long text without any dots so it should be treated as one single sentence by the splitter even if it is long"
        sentences, scores = self.run_pipeline(text)

        self.assertEqual(len(sentences), 1)
        self.assertIsNone(scores)


if __name__ == '__main__':
    unittest.main()