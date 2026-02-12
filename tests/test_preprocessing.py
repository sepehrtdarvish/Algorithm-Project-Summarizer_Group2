import unittest
from src.preprocessing import split_into_sentences, filter_sentences


class TestPreprocessing(unittest.TestCase):

    def test_split_sentences_basic(self):
        """بررسی جداسازی ساده جملات"""
        text = "Hello world. This is a test."
        result = split_into_sentences(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "Hello world.")

    def test_edge_case_empty(self):
        """(Edge Case) ورودی خالی نباید خطا دهد"""
        text = ""
        result = split_into_sentences(text)
        self.assertEqual(result, [])

    def test_abbreviations(self):
        """(Hard Input) تشخیص مخفف‌ها مثل Mr. نباید جمله را بشکند"""
        text = "Mr. Smith went to Washington."
        result = split_into_sentences(text)
        self.assertEqual(len(result), 1)  # نباید بعد از Mr. بشکند

    def test_filter_short_sentences(self):
        """حذف جملات خیلی کوتاه"""
        sentences = ["Hi", "This is a valid sentence.", "No"]
        # فرض کنیم مینیمم طول ۱۰ است
        result = filter_sentences(sentences, min_length=10)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "This is a valid sentence.")


if __name__ == '__main__':
    unittest.main()