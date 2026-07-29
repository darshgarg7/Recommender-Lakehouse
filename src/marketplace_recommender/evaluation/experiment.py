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


def promotion_decision(
    reports: dict[str, dict[str, Any]],
    *,
    candidate: str = "full_reranked",
    max_relative_relevance_regression: float = 0.02,
) -> dict[str, Any]:
    """Select a serving champion using relevance and zero-history guardrails."""
    if candidate not in reports:
        raise ValueError(f"missing candidate report: {candidate}")
    baselines = [name for name in reports if name != candidate]
    if not baselines:
        raise ValueError("promotion requires at least one baseline")
    best_baseline = max(
        baselines,
        key=lambda name: reports[name]["ranking"]["ndcg_at_10"],
    )
    baseline_ndcg = float(reports[best_baseline]["ranking"]["ndcg_at_10"])
    candidate_ndcg = float(reports[candidate]["ranking"]["ndcg_at_10"])
    minimum_ndcg = baseline_ndcg * (1.0 - max_relative_relevance_regression)
    baseline_cold = float(
        reports[best_baseline].get("cohorts", {}).get("zero-history", {}).get("recall_at_10", 0.0)
    )
    candidate_cold = float(
        reports[candidate].get("cohorts", {}).get("zero-history", {}).get("recall_at_10", 0.0)
    )
    relevance_passed = candidate_ndcg >= minimum_ndcg
    cold_start_passed = candidate_cold >= baseline_cold
    promoted = relevance_passed and cold_start_passed
    return {
        "candidate": candidate,
        "best_baseline": best_baseline,
        "serving_champion": candidate if promoted else best_baseline,
        "promoted": promoted,
        "relevance_guardrail_passed": relevance_passed,
        "cold_start_guardrail_passed": cold_start_passed,
        "max_relative_relevance_regression": max_relative_relevance_regression,
        "baseline_ndcg_at_10": baseline_ndcg,
        "candidate_ndcg_at_10": candidate_ndcg,
        "baseline_zero_history_recall_at_10": baseline_cold,
        "candidate_zero_history_recall_at_10": candidate_cold,
    }
