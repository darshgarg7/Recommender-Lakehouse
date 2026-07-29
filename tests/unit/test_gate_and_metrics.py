import unittest

from marketplace_recommender.evaluation.ranking_metrics import ndcg_at_k, recall_at_k
from marketplace_recommender.retrieval.two_tower import cold_start_gate


class GateAndMetricTests(unittest.TestCase):
    def test_gate_uses_content_for_zero_history_and_grows_monotonically(self):
        values = [cold_start_gate(count) for count in (0, 1, 10, 100)]
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values, sorted(values))
        self.assertLess(values[-1], 1.0)
        self.assertEqual(cold_start_gate(1, has_content=False), 1.0)

    def test_ranking_metrics(self):
        ranked = ["a", "b", "c"]
        self.assertEqual(recall_at_k(ranked, {"b"}, 1), 0.0)
        self.assertEqual(recall_at_k(ranked, {"b"}, 2), 1.0)
        self.assertGreater(ndcg_at_k(ranked, {"b"}, 2), 0.0)
        self.assertLess(ndcg_at_k(ranked, {"b"}, 2), 1.0)


if __name__ == "__main__":
    unittest.main()
