from __future__ import annotations

import unittest

from marketplace_recommender.retrieval.sasrec_torch import (
    SasRecConfig,
    build_next_item_examples,
    ranking_metrics_from_rank,
)


class SasRecContractTests(unittest.TestCase):
    def test_examples_are_causal_and_skip_seen_targets(self) -> None:
        examples = build_next_item_examples([[1, 2, 3, 2, 4]], max_sequence_length=3)
        self.assertEqual(examples, [([1, 2], 3), ([2, 3, 2], 4)])
        self.assertTrue(all(target not in history for history, target in examples))

    def test_rank_metrics_apply_the_declared_cutoffs(self) -> None:
        self.assertEqual(ranking_metrics_from_rank(1)["ndcg_at_10"], 1.0)
        self.assertEqual(ranking_metrics_from_rank(11)["recall_at_10"], 0.0)
        self.assertEqual(ranking_metrics_from_rank(99)["candidate_recall_at_100"], 1.0)
        self.assertEqual(ranking_metrics_from_rank(None)["mrr_at_10"], 0.0)

    def test_invalid_transformer_width_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            SasRecConfig(hidden_size=63, attention_heads=4).validate()


if __name__ == "__main__":
    unittest.main()
