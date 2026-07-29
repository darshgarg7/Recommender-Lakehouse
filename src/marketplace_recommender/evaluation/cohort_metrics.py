from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def marketplace_metrics(
    recommendation_rows: Iterable[dict[str, Any]], catalog: set[str]
) -> dict[str, float]:
    rows = list(recommendation_rows)
    exposed = {row["parent_asin"] for row in rows}
    item_counts = Counter(row["parent_asin"] for row in rows)
    brand_counts = Counter(row.get("brand_or_store", "") for row in rows)
    total = len(rows)
    return {
        "catalog_coverage": len(exposed) / len(catalog) if catalog else 0.0,
        "long_tail_exposure": (
            sum(row.get("cold_start_bucket") in {"zero-history", "sparse"} for row in rows) / total
            if total
            else 0.0
        ),
        "item_concentration": max(item_counts.values(), default=0) / total if total else 0.0,
        "brand_concentration": max(brand_counts.values(), default=0) / total if total else 0.0,
    }


def metrics_by_cohort(examples: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    from marketplace_recommender.evaluation.ranking_metrics import aggregate_ranking_metrics

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in examples:
        groups.setdefault(row["cohort"], []).append(row)
    return {
        cohort: {**aggregate_ranking_metrics(rows), "example_count": len(rows)}
        for cohort, rows in sorted(groups.items())
    }
