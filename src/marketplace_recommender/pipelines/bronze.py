from __future__ import annotations

import json
from pathlib import Path
from marketplace_recommender.ingestion.manifest import ManifestStore, utc_now
from marketplace_recommender.storage import read_jsonl, write_jsonl_atomic


def ingest_manifest_objects(
    manifest: ManifestStore,
    bronze_dir: str | Path,
) -> dict[str, int]:
    """Ingest pending JSONL objects without allowing one malformed row to stop a domain."""
    destination = Path(bronze_dir)
    destination.mkdir(parents=True, exist_ok=True)
    table_paths = {
        "reviews": destination / "bronze_reviews.jsonl",
        "metadata": destination / "bronze_product_metadata.jsonl",
    }
    existing = {
        kind: list(read_jsonl(path)) if path.exists() else [] for kind, path in table_paths.items()
    }
    seen = {
        kind: {(row["source_checksum"], row["source_row_number"]) for row in rows}
        for kind, rows in existing.items()
    }
    quarantine_path = destination / "bronze_quarantined_records.jsonl"
    quarantined = list(read_jsonl(quarantine_path)) if quarantine_path.exists() else []
    counts = {"reviews": 0, "metadata": 0, "quarantined": 0, "replayed_files": 0}

    for item in manifest.all():
        if item["download_status"] != "complete":
            continue
        kind = str(item["source_kind"])
        if kind not in table_paths:
            continue
        if item.get("ingestion_status") == "committed":
            item["replay_attempts"] = int(item.get("replay_attempts", 0)) + 1
            item["last_replay_at"] = utc_now()
            manifest.upsert(item)
            counts["replayed_files"] += 1
            continue
        source = Path(str(item["object_path"]))
        with source.open(encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, start=1):
                identity = (item["checksum"], row_number)
                if identity in seen[kind]:
                    continue
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise ValueError("top-level value must be an object")
                    existing[kind].append(
                        {
                            "source_file": str(source),
                            "source_domain": item["source_domain"],
                            "source_checksum": item["checksum"],
                            "source_row_number": row_number,
                            "raw_payload": payload,
                            "schema_version": 1,
                            "rescued_data": {},
                            "ingested_at": item["download_completed_at"],
                        }
                    )
                    seen[kind].add(identity)
                    counts[kind] += 1
                except (json.JSONDecodeError, ValueError) as exc:
                    quarantined.append(
                        {
                            "source_file": str(source),
                            "source_checksum": item["checksum"],
                            "source_row_number": row_number,
                            "reason": str(exc),
                            "raw_payload": line.rstrip("\n")[:10_000],
                            "quarantined_at": item["download_completed_at"],
                        }
                    )
                    counts["quarantined"] += 1
        manifest.mark_ingested(str(item["source_url"]), str(item["checksum"]))

    for kind, rows in existing.items():
        rows.sort(key=lambda row: (row["source_checksum"], row["source_row_number"]))
        write_jsonl_atomic(table_paths[kind], rows)
    write_jsonl_atomic(quarantine_path, quarantined)
    write_jsonl_atomic(destination / "bronze_ingestion_manifest.jsonl", manifest.all())
    return counts
