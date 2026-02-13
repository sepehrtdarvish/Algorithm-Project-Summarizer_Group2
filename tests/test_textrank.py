import unittest
import numpy as np
from src.textrank import run_pagerank


class TestPageRank(unittest.TestCase):

    def test_convergence_theoretical(self):
        """(Theoretical Comparison) تست همگرایی روی یک گراف ساده"""
        graph = np.array([[0, 1], [1, 0]])

        scores = run_pagerank(graph, d=0.85)

        self.assertAlmostEqual(scores[0], scores[1], places=4)

    def test_disconnected_graph(self):
        """(Edge Case) گراف بدون یال (هیچ شباهتی بین جملات نیست)"""
        graph = np.zeros((3, 3))
        scores = run_pagerank(graph, d=0.85)

        expected_score = 1 - 0.85
        for score in scores:
            self.assertAlmostEqual(score, expected_score, places=4)


if __name__ == '__main__':
    unittest.main()