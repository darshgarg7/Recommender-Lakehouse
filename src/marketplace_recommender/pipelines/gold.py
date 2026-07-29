from __future__ import annotations

from pathlib import Path
from typing import Any

from marketplace_recommender.evaluation.temporal_split import TemporalCutoffs
from marketplace_recommender.features.item_features import (
    item_content_features,
    item_statistics_asof,
)
from marketplace_recommender.features.point_in_time import assert_point_in_time
from marketplace_recommender.features.sequences import build_sequences
from marketplace_recommender.features.user_features import user_features_asof
from marketplace_recommender.storage import read_jsonl, write_jsonl_atomic


def build_gold(
    silver_dir: str | Path,
    gold_dir: str | Path,
    cutoffs: TemporalCutoffs,
    sequence_max_length: int = 100,
) -> dict[str, int]:
    source = Path(silver_dir)
    target = Path(gold_dir)
    target.mkdir(parents=True, exist_ok=True)
    interactions = list(read_jsonl(source / "silver_interactions.jsonl"))
    products = list(read_jsonl(source / "silver_products.jsonl"))
    labels = []
    for row in interactions:
        if not row["verified_purchase"] or row["rating"] == 3:
            continue
        labels.append(
            {
                "interaction_id": row["interaction_id"],
                "user_id": row["user_id"],
                "parent_asin": row["parent_asin"],
                "label_timestamp": row["review_timestamp"],
                "label": 1 if row["rating"] >= 4 else 0,
                "rating": row["rating"],
                "split": cutoffs.split_for(row["review_timestamp"]),
            }
        )
    observations = [(row["user_id"], row["label_timestamp"]) for row in labels]
    observation_points = [(row["parent_asin"], row["label_timestamp"]) for row in labels]
    sequences = build_sequences(interactions, observations, sequence_max_length)
    sequence_index = {(row["user_id"], row["observation_timestamp"]): row for row in sequences}
    item_stats = item_statistics_asof(interactions, observation_points)
    item_index = {(row["parent_asin"], row["observation_timestamp"]): row for row in item_stats}
    user_features = user_features_asof(interactions, products, observations)
    user_index = {(row["user_id"], row["observation_timestamp"]): row for row in user_features}
    examples: list[dict[str, Any]] = []
    for label in labels:
        key = (label["user_id"], label["label_timestamp"])
        item = item_index.get((label["parent_asin"], label["label_timestamp"]), {})
        sequence = sequence_index[key]
        user = user_index[key]
        examples.append(
            {
                **label,
                "feature_timestamp": max(
                    int(item.get("feature_timestamp") or 0),
                    int(user.get("feature_timestamp") or 0),
                ),
                "historical_parent_asins": sequence["historical_parent_asins"],
                "historical_event_times": sequence["historical_event_times"],
                "item_positive_interactions": item.get("positive_interactions_lifetime", 0),
                "cold_start_bucket": item.get("cold_start_bucket", "zero-history"),
                "preferred_domains": user.get("preferred_domains", []),
                "preferred_categories": user.get("preferred_categories", []),
            }
        )
    assert_point_in_time(examples)
    content = item_content_features(products)
    tables = {
        "gold_user_sequences_asof": sequences,
        "gold_user_features_asof": user_features,
        "gold_item_statistics_asof": item_stats,
        "gold_item_content_features": content,
        "gold_training_labels": labels,
        "gold_training_examples": examples,
    }
    for name, rows in tables.items():
        write_jsonl_atomic(target / f"{name}.jsonl", rows)
    return {name: len(rows) for name, rows in tables.items()}
