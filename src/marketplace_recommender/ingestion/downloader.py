from __future__ import annotations

import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from marketplace_recommender.ingestion.manifest import ManifestStore, utc_now
from marketplace_recommender.ingestion.validation import validate_jsonl
from marketplace_recommender.storage import file_checksum


class BoundedDownloader:
    def __init__(self, manifest: ManifestStore, workers: int = 4, retries: int = 3) -> None:
        if workers < 1 or retries < 0:
            raise ValueError("workers must be positive and retries non-negative")
        self.manifest = manifest
        self.workers = workers
        self.retries = retries

    def download_all(
        self,
        objects: Iterable[dict[str, str]],
        landing_dir: str | Path,
    ) -> list[dict[str, object]]:
        landing = Path(landing_dir)
        landing.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self._download_one, obj, landing) for obj in objects]
            return [future.result() for future in as_completed(futures)]

    def _download_one(self, obj: dict[str, str], landing: Path) -> dict[str, object]:
        url = obj["source_url"]
        destination = landing / obj.get("filename", Path(url).name)
        partial = destination.with_suffix(destination.suffix + ".part")
        prior = next(
            (
                row
                for row in self.manifest.all()
                if row["source_url"] == url
                and row.get("download_status") == "complete"
                and Path(str(row["object_path"])).exists()
            ),
            None,
        )
        if prior is not None and file_checksum(prior["object_path"]) == prior["checksum"]:
            return prior
        started = utc_now()
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                if url.startswith("file://"):
                    shutil.copyfile(url[7:], partial)
                else:
                    request = urllib.request.Request(url)
                    if partial.exists() and partial.stat().st_size:
                        request.add_header("Range", f"bytes={partial.stat().st_size}-")
                    mode = "ab" if "Range" in request.headers else "wb"
                    with (
                        urllib.request.urlopen(request, timeout=120) as response,
                        partial.open(mode) as out,
                    ):
                        shutil.copyfileobj(response, out)
                checksum = file_checksum(partial)
                expected_checksum = obj.get("expected_checksum")
                if expected_checksum and checksum != expected_checksum:
                    raise ValueError(
                        f"checksum mismatch for {url}: expected {expected_checksum}, got {checksum}"
                    )
                row_count, failures = validate_jsonl(
                    partial,
                    compressed=destination.suffix == ".gz",
                )
                if failures:
                    raise ValueError(f"JSON validation failed: {failures[:3]}")
                partial.replace(destination)
                record: dict[str, object] = {
                    **obj,
                    "object_path": str(destination),
                    "compressed_bytes": destination.stat().st_size,
                    "checksum": checksum,
                    "download_started_at": started,
                    "download_completed_at": utc_now(),
                    "download_status": "complete",
                    "retry_count": attempt,
                    "ingestion_status": "pending",
                    "validated_rows": row_count,
                }
                return self.manifest.upsert(record)
            except Exception as exc:  # each source is isolated from unrelated downloads
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 8))
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"failed to download {url} after {self.retries + 1} attempts"
        ) from last_error
