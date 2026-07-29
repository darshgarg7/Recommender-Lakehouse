import unittest

from marketplace_recommender.retrieval.capabilities import resolve_capabilities


class CapabilityResolutionTests(unittest.TestCase):
    def test_representation_progresses_with_available_evidence(self):
        cold = resolve_capabilities(content_available=True, behavioral_events=0)
        sparse = resolve_capabilities(content_available=True, behavioral_events=4)
        warm = resolve_capabilities(content_available=True, behavioral_events=40)
        self.assertEqual(cold.representation_strategy, "content_cold_start")
        self.assertEqual(sparse.representation_strategy, "distilled_sparse_hybrid")
        self.assertEqual(warm.representation_strategy, "warm_hybrid")
        self.assertNotIn("collaborative", cold.evidence_capabilities)
        self.assertIn("collaborative", sparse.evidence_capabilities)

    def test_catalog_graph_is_declared_only_when_observed(self):
        profile = resolve_capabilities(
            content_available=False,
            behavioral_events=0,
            observed_retrieval_channels=("bought_together",),
        )
        self.assertEqual(profile.representation_strategy, "graph_seeded_fallback")
        self.assertEqual(profile.evidence_capabilities, ("catalog_graph",))

    def test_negative_behavior_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_capabilities(content_available=True, behavioral_events=-1)


if __name__ == "__main__":
    unittest.main()
