from __future__ import annotations

import random
from collections import Counter
from typing import Any, Iterable


def sample_negatives(
    *,
    user_id: str,
    label_timestamp: int,
    horizon_end: int,
    catalog: Iterable[str],
    interactions: Iterable[dict[str, Any]],
    count: int,
    seed: int,
    strategy: str = "random",
    same_domain_items: set[str] | None = None,
) -> list[str]:
    rows = list(interactions)
    known_before = {
        row["parent_asin"]
        for row in rows
        if row["user_id"] == user_id
        and row["review_timestamp"] < label_timestamp
        and row["verified_purchase"]
        and row["rating"] >= 4
    }
    future_positives = {
        row["parent_asin"]
        for row in rows
        if row["user_id"] == user_id
        and label_timestamp <= row["review_timestamp"] <= horizon_end
        and row["verified_purchase"]
        and row["rating"] >= 4
    }
    eligible = sorted(set(catalog) - known_before - future_positives)
    if same_domain_items is not None:
        eligible = [item for item in eligible if item in same_domain_items]
    rng = random.Random(f"{seed}:{user_id}:{label_timestamp}:{strategy}")
    if strategy == "popularity":
        frequencies = Counter(
            row["parent_asin"] for row in rows if row["review_timestamp"] < label_timestamp
        )
        pool = [item for item in eligible for _ in range(max(1, frequencies[item]))]
        sampled: list[str] = []
        while pool and len(sampled) < count:
            item = rng.choice(pool)
            sampled.append(item)
            pool = [candidate for candidate in pool if candidate != item]
        return sampled
    rng.shuffle(eligible)
    return eligible[:count]


def assert_no_future_positive_negatives(
    negatives: Iterable[str],
    future_positive_items: set[str],
) -> None:
    overlap = set(negatives) & future_positive_items
    if overlap:
        raise AssertionError(f"future positives sampled as negatives: {sorted(overlap)}")
