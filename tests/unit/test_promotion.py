import unittest

from marketplace_recommender.governance.promotion import PromotionPolicy, promotion_decision


def report(
    ndcg: float,
    zero_history_recall: float,
    sparse_recall: float,
    candidate_coverage: float = 1.0,
) -> dict:
    return {
        "ranking": {"ndcg_at_10": ndcg},
        "retrieval": {"candidate_coverage": candidate_coverage},
        "cohorts": {
            "zero-history": {"recall_at_10": zero_history_recall, "example_count": 20},
            "sparse": {"recall_at_10": sparse_recall, "example_count": 30},
        },
        "example_count": 50,
    }


class PromotionDecisionTests(unittest.TestCase):
    def test_candidate_promotes_only_when_both_guardrails_pass(self):
        decision = promotion_decision(
            {
                "popularity": report(0.40, 0.10, 0.20),
                "content_similarity": report(0.50, 0.30, 0.35),
                "full_reranked": report(0.495, 0.40, 0.36),
            }
        )
        self.assertTrue(decision["promoted"])
        self.assertEqual(decision["serving_champion"], "full_reranked")
        self.assertEqual(decision["best_baseline"], "content_similarity")

    def test_failed_candidate_falls_back_to_strongest_baseline(self):
        decision = promotion_decision(
            {
                "popularity": report(0.60, 0.00, 0.30),
                "content_similarity": report(0.30, 0.20, 0.35),
                "full_reranked": report(0.40, 0.25, 0.40),
            }
        )
        self.assertFalse(decision["promoted"])
        self.assertEqual(decision["serving_champion"], "popularity")
        self.assertFalse(decision["relevance_guardrail_passed"])

    def test_missing_required_cohort_fails_closed(self):
        incomplete = report(0.50, 0.30, 0.35)
        del incomplete["cohorts"]["sparse"]
        decision = promotion_decision(
            {
                "content_similarity": report(0.50, 0.30, 0.35),
                "full_reranked": incomplete,
            }
        )
        self.assertFalse(decision["promoted"])
        sparse = next(
            row for row in decision["guardrails"] if row["name"] == "cohort_sparse_noninferiority"
        )
        self.assertEqual(sparse["failure_reason"], "missing_metric_or_below_threshold")

    def test_policy_identifier_changes_with_the_contract(self):
        default_id = PromotionPolicy().contract()["policy_id"]
        strict_id = PromotionPolicy(max_relative_relevance_regression=0.0).contract()["policy_id"]
        self.assertNotEqual(default_id, strict_id)


if __name__ == "__main__":
    unittest.main()
