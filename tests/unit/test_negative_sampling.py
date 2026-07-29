import unittest

from marketplace_recommender.retrieval.negative_sampling import sample_negatives


class NegativeSamplingTests(unittest.TestCase):
    def test_future_positive_is_never_sampled(self):
        interactions = [
            {
                "user_id": "u",
                "parent_asin": "old",
                "review_timestamp": 10,
                "verified_purchase": True,
                "rating": 5,
            },
            {
                "user_id": "u",
                "parent_asin": "future",
                "review_timestamp": 30,
                "verified_purchase": True,
                "rating": 4,
            },
        ]
        sampled = sample_negatives(
            user_id="u",
            label_timestamp=20,
            horizon_end=40,
            catalog={"old", "future", "safe"},
            interactions=interactions,
            count=10,
            seed=7,
        )
        self.assertEqual(sampled, ["safe"])


if __name__ == "__main__":
    unittest.main()
