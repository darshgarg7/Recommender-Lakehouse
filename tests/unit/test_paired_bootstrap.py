from __future__ import annotations

import unittest

from marketplace_recommender.evaluation.paired_bootstrap import (
    paired_bootstrap_mean_difference,
)


class PairedBootstrapTests(unittest.TestCase):
    def test_positive_paired_effect_has_reproducible_interval(self) -> None:
        baseline = [0.0, 0.2, 0.0, 0.4, 0.1] * 20
        candidate = [0.1, 0.3, 0.1, 0.5, 0.2] * 20
        first = paired_bootstrap_mean_difference(baseline, candidate, samples=500, seed=7)
        second = paired_bootstrap_mean_difference(baseline, candidate, samples=500, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first.point_estimate, 0.1)
        self.assertGreater(first.lower, 0.0)
        self.assertGreater(first.probability_of_improvement, 0.99)

    def test_pairing_and_input_contracts_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "same matched examples"):
            paired_bootstrap_mean_difference([0.0], [0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "at least one"):
            paired_bootstrap_mean_difference([], [])
        with self.assertRaisesRegex(ValueError, "at least 100"):
            paired_bootstrap_mean_difference([0.0], [1.0], samples=99)


if __name__ == "__main__":
    unittest.main()
