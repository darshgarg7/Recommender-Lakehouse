from __future__ import annotations

import unittest

from marketplace_recommender.evaluation.experiment import (
    evaluate_rankings,
    group_future_positives,
)


class EvaluationExperimentTests(unittest.TestCase):
    def test_evaluation_uses_candidates_for_retrieval_and_ranked_for_relevance(self) -> None:
        result = evaluate_rankings(
            [
                {
                    "ranked": ["hit", "other"],
                    "candidates": ["other", "hit", "third"],
                    "target": "hit",
                    "cohort": "zero_history",
                },
                {
                    "ranked": ["miss"],
                    "target": "target",
                    "cohort": "warm",
                },
            ],
            {"hit", "other", "third", "miss", "target"},
        )

        self.assertEqual(result["example_count"], 2)
        self.assertEqual(result["ranking"]["recall_at_10"], 0.5)
        self.assertEqual(result["retrieval"]["recall_at_100"], 0.5)
        self.assertEqual(result["retrieval"]["candidate_coverage"], 0.8)
        self.assertEqual(set(result["cohorts"]), {"zero_history", "warm"})

    def test_future_positive_grouping_respects_closed_window_and_policy(self) -> None:
        rows = [
            {
                "user_id": "u",
                "parent_asin": item,
                "review_timestamp": timestamp,
                "verified_purchase": verified,
                "rating": rating,
            }
            for item, timestamp, verified, rating in (
                ("start", 10, True, 4),
                ("end", 20, True, 5),
                ("unverified", 15, False, 5),
                ("low", 15, True, 2),
                ("late", 21, True, 5),
            )
        ]

        self.assertEqual(group_future_positives(rows, 10, 20), {"u": {"start", "end"}})


if __name__ == "__main__":
    unittest.main()
