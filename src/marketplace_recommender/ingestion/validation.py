from __future__ import annotations

import gzip
import json
from pathlib import Path


def validate_jsonl(path: str | Path, *, compressed: bool | None = None) -> tuple[int, list[str]]:
    source = Path(path)
    use_gzip = source.suffix == ".gz" if compressed is None else compressed
    opener = gzip.open if use_gzip else open
    failures: list[str] = []
    count = 0
    with opener(source, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError as exc:
                failures.append(f"line {line_number}: {exc.msg}")
    return count, failures
