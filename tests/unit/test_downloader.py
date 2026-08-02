from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from marketplace_recommender.ingestion.downloader import BoundedDownloader
from marketplace_recommender.ingestion.manifest import ManifestStore
from marketplace_recommender.storage import file_checksum


def source_object(path: Path, **overrides: str) -> dict[str, str]:
    return {
        "source_url": path.as_uri(),
        "source_domain": "Appliances",
        "source_kind": "reviews",
        "filename": "reviews.jsonl",
        **overrides,
    }


class BoundedDownloaderTests(unittest.TestCase):
    def test_local_download_is_validated_and_manifested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            source.write_text(
                "\n".join(json.dumps({"id": value}) for value in (1, 2)) + "\n",
                encoding="utf-8",
            )
            manifest = ManifestStore(root / "manifest.jsonl")

            records = BoundedDownloader(manifest, workers=1, retries=0).download_all(
                [source_object(source)], root / "landing"
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["validated_rows"], 2)
            self.assertEqual(records[0]["download_status"], "complete")
            self.assertEqual(records[0]["checksum"], file_checksum(root / "landing/reviews.jsonl"))
            self.assertEqual(manifest.all(), records)

    def test_completed_object_replays_without_reading_changed_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            source.write_text('{"id": 1}\n', encoding="utf-8")
            manifest = ManifestStore(root / "manifest.jsonl")
            downloader = BoundedDownloader(manifest, workers=1, retries=0)
            obj = source_object(source)
            first = downloader.download_all([obj], root / "landing")[0]

            source.write_text("not-json\n", encoding="utf-8")
            replay = downloader.download_all([obj], root / "landing")[0]

            self.assertEqual(replay, first)
            self.assertEqual(len(manifest.all()), 1)
            self.assertEqual((root / "landing/reviews.jsonl").read_text(), '{"id": 1}\n')

    def test_checksum_mismatch_never_promotes_partial_object(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            source.write_text('{"id": 1}\n', encoding="utf-8")
            manifest = ManifestStore(root / "manifest.jsonl")

            with self.assertRaisesRegex(RuntimeError, "failed to download"):
                BoundedDownloader(manifest, workers=1, retries=0).download_all(
                    [source_object(source, expected_checksum="0" * 64)], root / "landing"
                )

            self.assertFalse((root / "landing/reviews.jsonl").exists())
            self.assertFalse((root / "landing/reviews.jsonl.part").exists())
            self.assertEqual(manifest.all(), [])

    def test_malformed_json_never_promotes_partial_object(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            source.write_text('{"id": 1}\ninvalid\n', encoding="utf-8")
            manifest = ManifestStore(root / "manifest.jsonl")

            with self.assertRaisesRegex(RuntimeError, "failed to download"):
                BoundedDownloader(manifest, workers=1, retries=0).download_all(
                    [source_object(source)], root / "landing"
                )

            self.assertFalse((root / "landing/reviews.jsonl").exists())
            self.assertFalse((root / "landing/reviews.jsonl.part").exists())
            self.assertEqual(manifest.all(), [])


if __name__ == "__main__":
    unittest.main()
