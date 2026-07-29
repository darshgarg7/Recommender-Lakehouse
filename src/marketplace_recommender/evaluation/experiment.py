from __future__ import annotations

from collections import defaultdict
from typing import Any

from marketplace_recommender.evaluation.cohort_metrics import metrics_by_cohort
from marketplace_recommender.evaluation.ranking_metrics import aggregate_ranking_metrics
from marketplace_recommender.evaluation.retrieval_metrics import retrieval_metrics


def evaluate_rankings(rankings: list[dict[str, Any]], catalog: set[str]) -> dict[str, Any]:
    per_example = [
        {
            "ranked": row["ranked"],
            "candidates": row.get("candidates", row["ranked"]),
            "relevant": {row["target"]},
            "cohort": row["cohort"],
        }
        for row in rankings
    ]
    return {
        "ranking": aggregate_ranking_metrics(per_example),
        "retrieval": retrieval_metrics(
            [{"ranked": row["candidates"], "relevant": row["relevant"]} for row in per_example],
            catalog,
        ),
        "cohorts": metrics_by_cohort(per_example),
        "example_count": len(per_example),
    }


def group_future_positives(
    interactions: list[dict[str, Any]], start: int, end: int
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in interactions:
        if (
            start <= row["review_timestamp"] <= end
            and row["verified_purchase"]
            and row["rating"] >= 4
        ):
            result[row["user_id"]].add(row["parent_asin"])
    return result
