from __future__ import annotations

import unittest

from marketplace_recommender.retrieval.temporal_cutoffs import exact_temporal_cutoffs


class ExactTemporalCutoffTests(unittest.TestCase):
    def test_invalid_quantiles_fail_before_spark_is_required(self) -> None:
        with self.assertRaises(ValueError):
            exact_temporal_cutoffs(None, "timestamp", ())
        with self.assertRaises(ValueError):
            exact_temporal_cutoffs(None, "timestamp", (-0.1, 0.9))
        with self.assertRaises(ValueError):
            exact_temporal_cutoffs(None, "timestamp", (0.8, 1.1))


if __name__ == "__main__":
    unittest.main()
