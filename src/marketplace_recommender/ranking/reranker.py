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
) -> list[dict[str, Any]]:
    remaining = [dict(row) for row in candidates]
    selected: list[dict[str, Any]] = []
    brands: Counter[str] = Counter()
    while remaining and len(selected) < limit:
        best: tuple[float, str, dict[str, Any]] | None = None
        for candidate in remaining:
            brand = candidate.get("brand_or_store") or "__unknown__"
            if brands[brand] >= max_per_brand:
                continue
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
        selected.append(chosen)
        brands[chosen.get("brand_or_store") or "__unknown__"] += 1
        remaining.remove(chosen)
    for rank, candidate in enumerate(selected, start=1):
        candidate["rank"] = rank
    return selected
