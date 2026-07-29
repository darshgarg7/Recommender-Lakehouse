from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

DATASET_START_MS = int(datetime(1996, 5, 1, tzinfo=timezone.utc).timestamp() * 1000)
DATASET_END_MS = int(datetime(2023, 10, 1, tzinfo=timezone.utc).timestamp() * 1000)


def stable_interaction_id(user_id: str, asin: str, timestamp_ms: int, rating: float) -> str:
    payload = f"{user_id}\x1f{asin}\x1f{timestamp_ms}\x1f{rating:.1f}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def normalize_timestamp(value: Any) -> int:
    if isinstance(value, (int, float)):
        timestamp = int(value)
        return timestamp * 1000 if timestamp < 10_000_000_000 else timestamp
    if not isinstance(value, str):
        raise ValueError("timestamp must be epoch milliseconds or an ISO-8601 string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def validate_rating(value: Any) -> float:
    rating = float(value)
    if not 1.0 <= rating <= 5.0:
        raise ValueError(f"rating outside [1, 5]: {rating}")
    return rating


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value in {0, 1}:
        return bool(value)
    raise ValueError(f"invalid boolean: {value!r}")


def cold_start_bucket(interaction_count: int) -> str:
    if interaction_count <= 0:
        return "zero-history"
    if interaction_count <= 10:
        return "sparse"
    if interaction_count <= 100:
        return "developing"
    return "warm"
