import tempfile
import unittest
from pathlib import Path

from marketplace_recommender.pipelines.silver import build_silver
from marketplace_recommender.storage import read_jsonl, write_jsonl_atomic


class RealAmazonShapeTests(unittest.TestCase):
    def test_parent_asin_and_nested_metadata_shapes_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bronze = root / "bronze"
            common = {
                "source_file": "fixture.jsonl",
                "source_domain": "Magazine_Subscriptions",
                "source_checksum": "abc",
                "source_row_number": 1,
                "schema_version": 1,
                "rescued_data": {},
                "ingested_at": "2026-01-01T00:00:00+00:00",
            }
            write_jsonl_atomic(
                bronze / "bronze_product_metadata.jsonl",
                [
                    {
                        **common,
                        "raw_payload": {
                            "parent_asin": "PARENT",
                            "title": "Magazine",
                            "store": "Publisher",
                            "categories": ["Magazine Subscriptions"],
                            "details": '{"Language":"English"}',
                            "images": {"hi_res": ["https://example.invalid/image.jpg"]},
                        },
                    }
                ],
            )
            write_jsonl_atomic(
                bronze / "bronze_reviews.jsonl",
                [
                    {
                        **common,
                        "raw_payload": {
                            "user_id": "USER",
                            "asin": "CHILD",
                            "parent_asin": "PARENT",
                            "rating": 5.0,
                            "verified_purchase": True,
                            "timestamp": 1_600_000_000_000,
                        },
                    }
                ],
            )
            build_silver(bronze, root / "silver")
            interaction = next(read_jsonl(root / "silver/silver_interactions.jsonl"))
            product = next(read_jsonl(root / "silver/silver_products.jsonl"))
            self.assertEqual(interaction["parent_asin"], "PARENT")
            self.assertEqual(product["structured_attributes"], {"Language": "English"})
            self.assertEqual(product["image_references"], ["https://example.invalid/image.jpg"])


if __name__ == "__main__":
    unittest.main()
