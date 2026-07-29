import json
import tempfile
import unittest
from pathlib import Path

from marketplace_recommender.ingestion.manifest import ManifestStore
from marketplace_recommender.pipelines.bronze import ingest_manifest_objects
from marketplace_recommender.storage import file_checksum, read_jsonl


class BronzeRecoveryTests(unittest.TestCase):
    def test_corrupt_row_is_quarantined_without_losing_valid_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "reviews.jsonl"
            source.write_text(
                json.dumps({"ok": 1}) + "\n{bad json\n" + json.dumps({"ok": 2}) + "\n"
            )
            manifest = ManifestStore(root / "manifest.jsonl")
            manifest.upsert(
                {
                    "source_url": f"file://{source}",
                    "source_domain": "Electronics",
                    "source_kind": "reviews",
                    "object_path": str(source),
                    "compressed_bytes": source.stat().st_size,
                    "checksum": file_checksum(source),
                    "download_started_at": "2025-01-01T00:00:00+00:00",
                    "download_completed_at": "2025-01-01T00:00:01+00:00",
                    "download_status": "complete",
                    "retry_count": 0,
                    "ingestion_status": "pending",
                }
            )
            result = ingest_manifest_objects(manifest, root / "bronze")
            self.assertEqual(result["reviews"], 2)
            self.assertEqual(result["quarantined"], 1)
            self.assertEqual(len(list(read_jsonl(root / "bronze/bronze_reviews.jsonl"))), 2)


if __name__ == "__main__":
    unittest.main()
