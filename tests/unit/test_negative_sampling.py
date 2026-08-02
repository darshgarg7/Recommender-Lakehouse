import unittest

from marketplace_recommender.retrieval.negative_sampling import (
    assert_no_future_positive_negatives,
    sample_negatives,
)


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

    def test_popularity_sampling_is_unique_deterministic_and_domain_bounded(self):
        interactions = [
            {
                "user_id": "other",
                "parent_asin": item,
                "review_timestamp": timestamp,
                "verified_purchase": True,
                "rating": 5,
            }
            for timestamp, item in enumerate(("popular", "popular", "popular", "tail"), start=1)
        ]
        parameters = {
            "user_id": "u",
            "label_timestamp": 10,
            "horizon_end": 20,
            "catalog": {"popular", "tail", "outside"},
            "interactions": interactions,
            "count": 3,
            "seed": 9,
            "strategy": "popularity",
            "same_domain_items": {"popular", "tail"},
        }

        first = sample_negatives(**parameters)
        second = sample_negatives(**parameters)

        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertEqual(set(first), {"popular", "tail"})

    def test_future_positive_assertion_fails_on_overlap(self):
        assert_no_future_positive_negatives(["safe"], {"future"})
        with self.assertRaisesRegex(AssertionError, "future positives"):
            assert_no_future_positive_negatives(["safe", "future"], {"future"})


if __name__ == "__main__":
    unittest.main()
