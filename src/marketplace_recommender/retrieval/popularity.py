from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def popularity_scores(interactions: Iterable[dict[str, Any]], cutoff: int) -> dict[str, float]:
    counts = Counter(
        row["parent_asin"]
        for row in interactions
        if row["review_timestamp"] < cutoff and row["verified_purchase"] and row["rating"] >= 4
    )
    maximum = max(counts.values(), default=1)
    return {item: count / maximum for item, count in counts.items()}


def recommend_popular(
    scores: dict[str, float], seen: set[str], limit: int
) -> list[tuple[str, float]]:
    return [
        pair
        for pair in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        if pair[0] not in seen
    ][:limit]
