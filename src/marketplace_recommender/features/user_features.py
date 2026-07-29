from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable

DAY_MS = 86_400_000


def user_features_asof(
    interactions: Iterable[dict[str, Any]],
    products: Iterable[dict[str, Any]],
    observation_times: Iterable[tuple[str, int]],
) -> list[dict[str, Any]]:
    product_map = {row["parent_asin"]: row for row in products}
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interactions:
        by_user[row["user_id"]].append(row)
    output: list[dict[str, Any]] = []
    for user_id, timestamp in sorted(set(observation_times)):
        history = [row for row in by_user[user_id] if row["review_timestamp"] < timestamp]
        positives = [row for row in history if row["verified_purchase"] and row["rating"] >= 4]
        domains = Counter(row["domain"] for row in positives)
        categories: Counter[str] = Counter()
        brands: Counter[str] = Counter()
        for row in positives:
            product = product_map.get(row["parent_asin"], {})
            categories.update(product.get("category_path", []))
            brand = product.get("brand_or_store")
            if brand:
                brands[brand] += 1
        category_total = sum(categories.values())
        entropy = (
            -sum(
                (count / category_total) * math.log(count / category_total)
                for count in categories.values()
            )
            if category_total
            else 0.0
        )
        ratings = Counter(str(int(row["rating"])) for row in history)
        output.append(
            {
                "user_id": user_id,
                "feature_timestamp": max((row["review_timestamp"] for row in history), default=0),
                "observation_timestamp": timestamp,
                "positive_interactions_30d": sum(
                    row["review_timestamp"] >= timestamp - 30 * DAY_MS for row in positives
                ),
                "positive_interactions_365d": sum(
                    row["review_timestamp"] >= timestamp - 365 * DAY_MS for row in positives
                ),
                "preferred_domains": [value for value, _ in domains.most_common(3)],
                "preferred_categories": [value for value, _ in categories.most_common(5)],
                "preferred_brands": [value for value, _ in brands.most_common(5)],
                "rating_distribution": dict(sorted(ratings.items())),
                "activity_recency": timestamp
                - max((row["review_timestamp"] for row in history), default=timestamp),
                "category_entropy": entropy,
                "long_tail_affinity": 0.0,
            }
        )
    return output
