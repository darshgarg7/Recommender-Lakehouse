import unittest

from marketplace_recommender.ranking.reranker import rerank


class RegretBoundedRerankerTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            {
                "parent_asin": "anchor",
                "ranking_score": 1.0,
                "novelty": 0.0,
                "cold_start_bucket": "warm",
                "brand_or_store": "A",
            },
            {
                "parent_asin": "tail",
                "ranking_score": 0.95,
                "novelty": 1.0,
                "cold_start_bucket": "zero-history",
                "brand_or_store": "B",
            },
            {
                "parent_asin": "floor",
                "ranking_score": 0.0,
                "novelty": 0.0,
                "cold_start_bucket": "warm",
                "brand_or_store": "C",
            },
        ]
        self.vectors = {row["parent_asin"]: [1.0, 0.0] for row in self.candidates}

    def _rerank(self, budget: float) -> list[dict]:
        return rerank(
            self.candidates,
            self.vectors,
            limit=3,
            novelty_weight=5.0,
            long_tail_weight=5.0,
            redundancy_weight=0.0,
            max_per_brand=3,
            max_score_regret=budget,
        )

    def test_marketplace_bonus_cannot_cross_the_regret_budget(self):
        rows = self._rerank(0.049)
        self.assertEqual(rows[0]["parent_asin"], "anchor")
        self.assertTrue(all(row["score_regret"] <= 0.049 + 1e-12 for row in rows))

    def test_tail_candidate_becomes_admissible_at_the_declared_boundary(self):
        rows = self._rerank(0.05)
        self.assertEqual(rows[0]["parent_asin"], "tail")
        self.assertAlmostEqual(rows[0]["score_regret"], 0.05)

    def test_invalid_regret_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            self._rerank(1.01)


if __name__ == "__main__":
    unittest.main()
