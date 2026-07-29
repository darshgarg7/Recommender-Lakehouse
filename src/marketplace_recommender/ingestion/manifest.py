from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from marketplace_recommender.storage import read_jsonl, write_jsonl_atomic


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ManifestStore:
    """Small local analogue of the Bronze Delta manifest, keyed by source URL and checksum."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def all(self) -> list[dict[str, Any]]:
        return list(read_jsonl(self.path)) if self.path.exists() else []

    def upsert(self, record: dict[str, Any]) -> dict[str, Any]:
        required = {
            "source_url",
            "source_domain",
            "source_kind",
            "object_path",
            "compressed_bytes",
            "checksum",
            "download_status",
            "retry_count",
            "ingestion_status",
        }
        missing = required - record.keys()
        if missing:
            raise ValueError(f"manifest record missing: {sorted(missing)}")
        key = (record["source_url"], record["checksum"])
        with self._lock:
            rows = self.all()
            matches = [
                index
                for index, row in enumerate(rows)
                if (row["source_url"], row["checksum"]) == key
            ]
            if matches:
                rows[matches[-1]] = record
            else:
                rows.append(record)
            write_jsonl_atomic(
                self.path, sorted(rows, key=lambda row: (row["source_url"], row["checksum"]))
            )
        return record

    def mark_ingested(self, source_url: str, checksum: str, status: str = "committed") -> None:
        rows = self.all()
        match = next(
            (
                row
                for row in rows
                if row["source_url"] == source_url and row["checksum"] == checksum
            ),
            None,
        )
        if match is None:
            raise KeyError(f"manifest object not found: {source_url} {checksum}")
        match["ingestion_status"] = status
        match["ingestion_completed_at"] = utc_now()
        self.upsert(match)

    def committed_checksums(self) -> set[str]:
        return {row["checksum"] for row in self.all() if row.get("ingestion_status") == "committed"}
