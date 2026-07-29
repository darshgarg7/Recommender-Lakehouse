import unittest
from pathlib import Path

from marketplace_recommender.pipelines.databricks import is_sha256


ROOT = Path(__file__).resolve().parents[2]


class DatabricksBundleContractTests(unittest.TestCase):
    def test_bundle_is_serverless_and_has_checksum_bootstrap(self):
        bundle = (ROOT / "databricks.yml").read_text(encoding="utf-8")
        self.assertNotIn("existing_cluster_id", bundle)
        self.assertIn("task_key: bootstrap", bundle)
        self.assertIn("environment_key: default", bundle)
        self.assertIn("reviews_checksum", bundle)
        self.assertIn("metadata_checksum", bundle)

    def test_checksum_contract_requires_hexadecimal_sha256(self):
        self.assertTrue(is_sha256("a" * 64))
        self.assertTrue(is_sha256("AB" * 32))
        self.assertFalse(is_sha256("g" * 64))
        self.assertFalse(is_sha256("a" * 63))


if __name__ == "__main__":
    unittest.main()
