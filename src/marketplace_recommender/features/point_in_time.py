from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from typing import Any, Iterable


def assert_point_in_time(rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        label_time = int(row["label_timestamp"])
        feature_time = int(row.get("feature_timestamp") or 0)
        if feature_time >= label_time and feature_time != 0:
            raise AssertionError(
                f"feature timestamp {feature_time} is not before label timestamp {label_time}"
            )
        for event_time in row.get("historical_event_times", []):
            if int(event_time) >= label_time:
                raise AssertionError("future interaction found inside user history")


def latest_before(
    feature_rows: Iterable[dict[str, Any]],
    labels: Iterable[dict[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    """Dependency-free as-of join with a strict less-than temporal boundary."""
    indexed: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        indexed[row[key]].append(row)
    for rows in indexed.values():
        rows.sort(key=lambda row: row["feature_timestamp"])
    output: list[dict[str, Any]] = []
    for label in labels:
        candidates = indexed.get(label[key], [])
        times = [row["feature_timestamp"] for row in candidates]
        position = bisect_left(times, label["label_timestamp"]) - 1
        output.append({**label, **(candidates[position] if position >= 0 else {})})
    return output
