import unittest

from marketplace_recommender.evaluation.experiment import promotion_decision


def report(ndcg: float, zero_history_recall: float) -> dict:
    return {
        "ranking": {"ndcg_at_10": ndcg},
        "cohorts": {"zero-history": {"recall_at_10": zero_history_recall}},
    }


class PromotionDecisionTests(unittest.TestCase):
    def test_candidate_promotes_only_when_both_guardrails_pass(self):
        decision = promotion_decision(
            {
                "popularity": report(0.40, 0.10),
                "content_similarity": report(0.50, 0.30),
                "full_reranked": report(0.495, 0.40),
            }
        )
        self.assertTrue(decision["promoted"])
        self.assertEqual(decision["serving_champion"], "full_reranked")
        self.assertEqual(decision["best_baseline"], "content_similarity")

    def test_failed_candidate_falls_back_to_strongest_baseline(self):
        decision = promotion_decision(
            {
                "popularity": report(0.60, 0.00),
                "content_similarity": report(0.30, 0.20),
                "full_reranked": report(0.40, 0.25),
            }
        )
        self.assertFalse(decision["promoted"])
        self.assertEqual(decision["serving_champion"], "popularity")
        self.assertFalse(decision["relevance_guardrail_passed"])


if __name__ == "__main__":
    unittest.main()
