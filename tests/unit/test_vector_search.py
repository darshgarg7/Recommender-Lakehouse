from __future__ import annotations

import math
import unittest

from marketplace_recommender.retrieval.vector_search import (
    VectorSearchBenchmarkConfig,
    _index_is_current_and_idle,
    mips_item_extension,
    mips_query_extension,
    parse_vector_search_rows,
    summarize_load,
)


class VectorSearchTests(unittest.TestCase):
    def test_mips_transform_preserves_inner_product_order_under_l2(self) -> None:
        query = [0.8, 0.2]
        items = [[0.9, 0.1], [0.4, 0.7], [0.1, 0.2]]
        maximum = max(sum(value * value for value in item) for item in items)
        transformed_items = [mips_item_extension(item, maximum) for item in items]
        inner_order = sorted(
            range(len(items)),
            key=lambda index: -sum(left * right for left, right in zip(query, items[index])),
        )
        for scale in (0.5, 1.0, 2.0):
            transformed_query = mips_query_extension(query, scale)
            l2_order = sorted(
                range(len(items)),
                key=lambda index: sum(
                    (left - right) ** 2
                    for left, right in zip(transformed_query, transformed_items[index], strict=True)
                ),
            )
            self.assertEqual(inner_order, l2_order)
        self.assertTrue(all(len(item) == 3 for item in transformed_items))
        with self.assertRaises(ValueError):
            mips_query_extension(query, 0.0)

    def test_response_parser_and_load_summary_are_contractual(self) -> None:
        response = {
            "manifest": {"columns": [{"name": "parent_asin"}, {"name": "score"}]},
            "result": {"data_array": [["A", 0.1], ["B", 0.2]]},
        }
        self.assertEqual(parse_vector_search_rows(response)[0]["parent_asin"], "A")
        load = summarize_load([1.0, 2.0, 3.0, 4.0], wall_seconds=0.5)
        self.assertEqual(load["completed_requests"], 4)
        self.assertEqual(load["throughput_qps"], 8.0)
        self.assertTrue(math.isclose(float(load["latency_p50_ms"]), 2.5))

    def test_benchmark_config_rejects_invalid_recall(self) -> None:
        with self.assertRaises(ValueError):
            VectorSearchBenchmarkConfig(minimum_recall_at_10=1.1).validate()
        with self.assertRaises(ValueError):
            VectorSearchBenchmarkConfig(k=10, ann_candidate_pool_size=9).validate()

    def test_index_must_be_idle_and_current_before_queries_are_certified(self) -> None:
        current = {
            "status": {
                "ready": True,
                "detailed_state": "ONLINE_NO_PENDING_UPDATE",
                "triggered_update_status": {"last_processed_commit_version": 7},
            }
        }
        self.assertTrue(_index_is_current_and_idle(current, 7))
        self.assertFalse(_index_is_current_and_idle(current, 8))
        current["status"]["detailed_state"] = "ONLINE_TRIGGERED_UPDATE"
        self.assertFalse(_index_is_current_and_idle(current, 7))


if __name__ == "__main__":
    unittest.main()
