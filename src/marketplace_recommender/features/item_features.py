from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from typing import Any, Iterable

from marketplace_recommender.schemas import cold_start_bucket

DAY_MS = 86_400_000


def item_statistics_asof(
    interactions: Iterable[dict[str, Any]],
    observation_points: Iterable[tuple[str, int]],
) -> list[dict[str, Any]]:
    events = sorted(interactions, key=lambda row: (row["review_timestamp"], row["interaction_id"]))
    points = sorted(set(observation_points), key=lambda pair: (pair[1], pair[0]))
    positive_times: dict[str, list[int]] = defaultdict(list)
    positive_counts: Counter[str] = Counter()
    verified_counts: Counter[str] = Counter()
    rating_sums: Counter[str] = Counter()
    dissatisfaction_counts: Counter[str] = Counter()
    last_event: dict[str, int] = {}
    output: list[dict[str, Any]] = []
    event_index = 0
    maximum_count = 0
    for product, timestamp in points:
        while event_index < len(events) and events[event_index]["review_timestamp"] < timestamp:
            row = events[event_index]
            item = row["parent_asin"]
            last_event[item] = row["review_timestamp"]
            if row["verified_purchase"]:
                verified_counts[item] += 1
                rating_sums[item] += row["rating"]
                if row["rating"] <= 2:
                    dissatisfaction_counts[item] += 1
                if row["rating"] >= 4:
                    positive_counts[item] += 1
                    positive_times[item].append(row["review_timestamp"])
                    maximum_count = max(maximum_count, positive_counts[item])
            event_index += 1
        times = positive_times[product]
        recent_7 = len(times) - bisect_left(times, timestamp - 7 * DAY_MS)
        recent_30 = len(times) - bisect_left(times, timestamp - 30 * DAY_MS)
        rated_count = verified_counts[product]
        count = positive_counts[product]
        output.append(
            {
                "parent_asin": product,
                "feature_timestamp": last_event.get(product, 0),
                "observation_timestamp": timestamp,
                "positive_interactions_7d": recent_7,
                "positive_interactions_30d": recent_30,
                "positive_interactions_lifetime": count,
                "historical_average_rating": rating_sums[product] / rated_count
                if rated_count
                else None,
                "historical_dissatisfaction_rate": (
                    dissatisfaction_counts[product] / rated_count if rated_count else 0.0
                ),
                "interaction_velocity": recent_30 / 30.0,
                "popularity_percentile": count / max(maximum_count, 1),
                "cold_start_bucket": cold_start_bucket(count),
            }
        )
    return output


def item_content_features(products: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for product in products:
        details = product.get("structured_attributes") or {}
        detail_text = " ".join(f"{key} {value}" for key, value in sorted(details.items()))
        content = " ".join(
            [
                product.get("title", ""),
                product.get("brand_or_store", ""),
                " ".join(product.get("category_path", [])),
                " ".join(product.get("description", [])),
                " ".join(product.get("feature_bullets", [])),
                detail_text,
            ]
        ).strip()
        output.append(
            {
                "parent_asin": product["parent_asin"],
                "domain": product["domain"],
                "brand_or_store": product.get("brand_or_store", ""),
                "category_path": product.get("category_path", []),
                "content_text": content,
                "has_content": bool(content),
                "has_image": bool(product.get("image_references")),
                # crawl_price is intentionally not copied into historical features.
            }
        )
    return output
