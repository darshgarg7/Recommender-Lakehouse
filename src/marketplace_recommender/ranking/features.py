from __future__ import annotations

from typing import Any

from marketplace_recommender.retrieval.two_tower import HybridTwoTower, cold_start_gate
from marketplace_recommender.retrieval.vectors import dot

FEATURE_NAMES = [
    "hybrid_ann_score",
    "cointeraction_score",
    "graph_score",
    "trend_score",
    "user_item_similarity",
    "domain_affinity",
    "category_affinity",
    "brand_affinity",
    "popularity",
    "cold_start_gate",
    "novelty",
]


def candidate_features(
    candidate: dict[str, Any],
    history: list[str],
    tower: HybridTwoTower,
    product: dict[str, Any],
    user_preferences: dict[str, list[str]],
    popularity: dict[str, float],
) -> dict[str, float]:
    item = candidate["parent_asin"]
    user_vector = tower.user_embedding(history)
    item_vector = tower.item_embedding(item)
    scores = candidate.get("retrieval_scores", {})
    pop = popularity.get(item, 0.0)
    categories = set(product.get("category_path", []))
    preferred_categories = set(user_preferences.get("preferred_categories", []))
    return {
        "hybrid_ann_score": scores.get("hybrid_ann", 0.0),
        "cointeraction_score": scores.get("cointeraction", 0.0),
        "graph_score": scores.get("bought_together", 0.0),
        "trend_score": scores.get("trend", 0.0),
        "user_item_similarity": dot(user_vector, item_vector),
        "domain_affinity": float(
            product.get("domain") in user_preferences.get("preferred_domains", [])
        ),
        "category_affinity": float(bool(categories & preferred_categories)),
        "brand_affinity": float(
            product.get("brand_or_store") in user_preferences.get("preferred_brands", [])
        ),
        "popularity": pop,
        "cold_start_gate": cold_start_gate(
            tower.interaction_counts.get(item, 0), tower.has_content.get(item, False)
        ),
        "novelty": 1.0 - pop,
    }
