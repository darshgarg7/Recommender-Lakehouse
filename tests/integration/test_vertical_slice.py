import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from marketplace_recommender.cli import run_demo


CONFIG = """\
tier: local
seed: 17
output_dir: {output}
domains: [Electronics]
interaction_count: 120
sequence_max_length: 50
candidate_limit: 100
recommendation_limit: 10
validation_fraction: 0.20
test_fraction: 0.20
negative_count: 8
rerank:
  novelty_weight: 0.08
  long_tail_weight: 0.12
  redundancy_weight: 0.10
  max_per_brand: 5
"""


class VerticalSliceTests(unittest.TestCase):
    def test_demo_is_replay_safe_and_materializes_serving_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.yml"
            config.write_text(CONFIG.format(output=root / "output"), encoding="utf-8")
            with redirect_stdout(StringIO()):
                first = run_demo(config)
                second = run_demo(config)
            self.assertEqual(first["fingerprints"], second["fingerprints"])
            self.assertGreater(second["bronze"]["replayed_files"], 0)
            output = root / "output" / "serving" / "gold_batch_recommendations.jsonl"
            self.assertTrue(output.exists())
            self.assertTrue((root / "output/ml/local-hybrid-v1.json").exists())
            self.assertGreater(second["verified_claims"]["batch_recommendation_rows"], 0)


if __name__ == "__main__":
    unittest.main()
