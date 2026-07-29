from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from marketplace_recommender.storage import write_jsonl_atomic


PRODUCT_FAMILIES = [
    ("audio", "wireless noise cancelling headphones", "SoundArc"),
    ("audio", "portable bluetooth speaker", "SoundArc"),
    ("computers", "mechanical gaming keyboard", "KeyForge"),
    ("computers", "ergonomic wireless mouse", "KeyForge"),
    ("photo", "compact mirrorless camera", "LumaShot"),
    ("photo", "travel camera tripod", "LumaShot"),
    ("smart-home", "wifi smart light bulbs", "NestBeam"),
    ("smart-home", "indoor security camera", "NestBeam"),
]


def generate_local_source(
    directory: str | Path, interaction_count: int, seed: int
) -> dict[str, Path]:
    """Create a deterministic Amazon-shaped shard with preference and long-tail signal."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    products: list[dict[str, Any]] = []
    variants: list[str] = []
    for family_index, (category, phrase, brand) in enumerate(PRODUCT_FAMILIES):
        for version in range(3):
            parent = f"P{family_index:02d}{version:02d}"
            asin = f"A{family_index:02d}{version:02d}0"
            variants.append(asin)
            products.append(
                {
                    "parent_asin": parent,
                    "asin": asin,
                    "title": f"{brand} {phrase} model {version + 1}",
                    "store": brand,
                    "main_category": "Electronics",
                    "categories": ["Electronics", category],
                    "description": [f"Reliable {phrase} for everyday use"],
                    "features": ["durable", "energy efficient", f"series {version + 1}"],
                    "details": {"family": category, "generation": version + 1},
                    "price": round(24.0 + family_index * 11.5 + version * 7.0, 2),
                    "images": [{"hi_res": f"https://example.invalid/{parent}.jpg"}],
                    "bought_together": [f"P{family_index ^ 1:02d}00"],
                }
            )

    users = [f"U{index:03d}" for index in range(40)]
    start = datetime(2021, 1, 1, tzinfo=timezone.utc)
    reviews: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    while len(reviews) < interaction_count and attempts < interaction_count * 20:
        attempts += 1
        user_index = rng.randrange(len(users))
        preferred_pair = user_index % 4
        selector = rng.random()
        if selector < 0.25:
            # A global head item creates a realistic exposure imbalance for reranking tests.
            family_index = 0
        elif selector < 0.82:
            family_index = preferred_pair * 2 + rng.randrange(2)
        else:
            family_index = rng.randrange(len(PRODUCT_FAMILIES))
        if selector < 0.25:
            version = 0
        else:
            # Reserve third-generation products for the late interval to exercise item cold start.
            version_weights = (
                [0.75, 0.25, 0.0]
                if len(reviews) < int(interaction_count * 0.60)
                else [0.55, 0.25, 0.20]
            )
            version = rng.choices(range(3), weights=version_weights, k=1)[0]
        asin = f"A{family_index:02d}{version:02d}0"
        key = (users[user_index], asin)
        if key in seen:
            continue
        seen.add(key)
        affinity = family_index // 2 == preferred_pair or selector < 0.25
        rating = rng.choices(
            [1, 2, 3, 4, 5],
            weights=[1, 2, 3, 8, 12] if affinity else [3, 4, 4, 5, 3],
            k=1,
        )[0]
        event_time = start + timedelta(days=len(reviews) * 3 + rng.randrange(2))
        reviews.append(
            {
                "user_id": users[user_index],
                "asin": asin,
                "rating": rating,
                "verified_purchase": rng.random() < 0.94,
                "timestamp": int(event_time.timestamp() * 1000),
                "title": "Works well" if rating >= 4 else "Mixed experience",
                "text": "A deterministic synthetic review used only for local validation.",
                "helpful_vote": rng.randrange(4),
            }
        )
    reviews.sort(key=lambda row: (row["timestamp"], row["user_id"], row["asin"]))
    metadata_path = target / "meta_Electronics.jsonl"
    reviews_path = target / "reviews_Electronics.jsonl"
    write_jsonl_atomic(metadata_path, products)
    write_jsonl_atomic(reviews_path, reviews)
    return {"metadata": metadata_path, "reviews": reviews_path}
