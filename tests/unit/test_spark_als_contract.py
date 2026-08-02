import unittest

from marketplace_recommender.retrieval.spark_als import (
    SparkAlsBenchmarkConfig,
    benchmark_fingerprint,
    ndcg_for_single_relevant_rank,
)


class SparkAlsBenchmarkContractTests(unittest.TestCase):
    def test_default_benchmark_is_a_temporal_two_stage_retriever(self):
        config = SparkAlsBenchmarkConfig()
        config.validate()
        self.assertGreaterEqual(config.rank, 32)
        self.assertGreater(config.candidate_k, config.recommendation_k)
        self.assertEqual(config.rrf_als_weights[0], 0.0)
        self.assertEqual(config.rrf_als_weights[-1], 1.0)

    def test_invalid_evaluation_boundaries_fail_closed(self):
        with self.assertRaises(ValueError):
            SparkAlsBenchmarkConfig(validation_fraction=0.6, test_fraction=0.4).validate()
        with self.assertRaises(ValueError):
            SparkAlsBenchmarkConfig(candidate_k=5, recommendation_k=10).validate()

    def test_benchmark_identity_is_order_independent_and_content_bound(self):
        first = benchmark_fingerprint({"rank": 64, "cutoff": 10})
        second = benchmark_fingerprint({"cutoff": 10, "rank": 64})
        changed = benchmark_fingerprint({"cutoff": 11, "rank": 64})
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertEqual(len(first), 64)

    def test_single_target_ndcg_uses_rank_discount(self):
        self.assertEqual(ndcg_for_single_relevant_rank(1), 1.0)
        self.assertGreater(ndcg_for_single_relevant_rank(2), ndcg_for_single_relevant_rank(10))
        self.assertEqual(ndcg_for_single_relevant_rank(11), 0.0)
        self.assertEqual(ndcg_for_single_relevant_rank(None), 0.0)


if __name__ == "__main__":
    unittest.main()
