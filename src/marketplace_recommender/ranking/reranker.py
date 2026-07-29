from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from marketplace_recommender.retrieval.vectors import dot


def rerank(
    candidates: Iterable[dict[str, Any]],
    item_vectors: dict[str, list[float]],
    *,
    limit: int,
    novelty_weight: float,
    long_tail_weight: float,
    redundancy_weight: float,
    max_per_brand: int,
    max_score_regret: float,
) -> list[dict[str, Any]]:
    """Rerank inside a bounded, normalized learned-score regret budget.

    The marketplace objectives may reorder only candidates whose normalized learned score is no
    more than ``max_score_regret`` below the best currently eligible item. This converts long-tail
    exposure from an unconstrained bonus into an auditable decision contract.
    """
    if not 0.0 <= max_score_regret <= 1.0:
        raise ValueError("max_score_regret must be within [0, 1]")
    remaining = [dict(row) for row in candidates]
    scores = [float(row["ranking_score"]) for row in remaining]
    low, high = min(scores, default=0.0), max(scores, default=0.0)
    span = high - low
    for candidate in remaining:
        candidate["normalized_relevance_score"] = (
            (float(candidate["ranking_score"]) - low) / span if span > 1e-12 else 1.0
        )
    selected: list[dict[str, Any]] = []
    brands: Counter[str] = Counter()
    while remaining and len(selected) < limit:
        eligible = [
            candidate
            for candidate in remaining
            if brands[candidate.get("brand_or_store") or "__unknown__"] < max_per_brand
        ]
        if not eligible:
            break
        relevance_anchor = max(
            float(candidate["normalized_relevance_score"]) for candidate in eligible
        )
        admissible = [
            candidate
            for candidate in eligible
            if relevance_anchor - float(candidate["normalized_relevance_score"])
            <= max_score_regret + 1e-12
        ]
        best: tuple[float, str, dict[str, Any]] | None = None
        for candidate in admissible:
            item = candidate["parent_asin"]
            redundancy = max(
                (
                    dot(item_vectors.get(item, []), item_vectors.get(old["parent_asin"], []))
                    for old in selected
                ),
                default=0.0,
            )
            tail_value = (
                1.0 if candidate.get("cold_start_bucket") in {"zero-history", "sparse"} else 0.0
            )
            final = (
                candidate["ranking_score"]
                + novelty_weight * candidate.get("novelty", 0.0)
                + long_tail_weight * tail_value
                - redundancy_weight * redundancy
            )
            choice = (final, item, candidate)
            if best is None or choice[:2] > best[:2]:
                best = choice
        if best is None:
            break
        final, _, chosen = best
        chosen["final_score"] = final
        chosen["relevance_anchor_score"] = relevance_anchor
        chosen["score_regret"] = relevance_anchor - float(chosen["normalized_relevance_score"])
        chosen["max_score_regret"] = max_score_regret
        chosen["decision_reason"] = (
            "relevance_anchor"
            if chosen["score_regret"] <= 1e-12
            else "bounded_marketplace_objective"
        )
        selected.append(chosen)
        brands[chosen.get("brand_or_store") or "__unknown__"] += 1
        remaining.remove(chosen)
    for rank, candidate in enumerate(selected, start=1):
        candidate["rank"] = rank
    return selected
