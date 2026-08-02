from __future__ import annotations

from typing import Any

from marketplace_recommender.evaluation.paired_bootstrap import paired_bootstrap_mean_difference
from marketplace_recommender.retrieval.spark_als_config import SparkAlsBenchmarkConfig


def _ranking_metrics(
    candidates: Any,
    targets: Any,
    item_popularity: Any,
    catalog_size: int,
    config: SparkAlsBenchmarkConfig,
) -> dict[str, float | int]:
    from pyspark.sql import functions as F

    per_user = _per_user_ranking_metrics(candidates, targets)
    aggregate = per_user.agg(
        F.count("*").alias("evaluation_users"),
        F.avg("recall_at_10").alias("recall_at_10"),
        F.avg("ndcg_at_10").alias("ndcg_at_10"),
        F.avg("mrr_at_10").alias("mrr_at_10"),
        F.avg("candidate_recall_at_100").alias("candidate_recall_at_100"),
    ).first()
    top_ten = candidates.where(F.col("rank") <= config.recommendation_k)
    distinct_recommended = top_ten.select("item_index").distinct().count()
    tail_threshold = item_popularity.approxQuantile("popularity", [0.5], 0.001)[0]
    tail_exposure = (
        top_ten.join(item_popularity.select("item_index", "popularity"), "item_index", "inner")
        .agg(F.avg((F.col("popularity") <= F.lit(tail_threshold)).cast("double")).alias("value"))
        .first()
        .value
    )
    return {
        "evaluation_users": int(aggregate.evaluation_users),
        "recall_at_10": float(aggregate.recall_at_10 or 0.0),
        "ndcg_at_10": float(aggregate.ndcg_at_10 or 0.0),
        "mrr_at_10": float(aggregate.mrr_at_10 or 0.0),
        "candidate_recall_at_100": float(aggregate.candidate_recall_at_100 or 0.0),
        "catalog_coverage_at_10": distinct_recommended / catalog_size if catalog_size else 0.0,
        "long_tail_exposure_at_10": float(tail_exposure or 0.0),
    }


def _per_user_ranking_metrics(candidates: Any, targets: Any) -> Any:
    """Return one metric row per matched evaluation user for paired inference."""
    from pyspark.sql import functions as F

    target_rows = targets.select("user_index", "target_item_index").alias("targets")
    candidate_rows = candidates.select("user_index", "item_index", "rank").alias("candidates")
    hit_ranks = target_rows.join(
        candidate_rows,
        (F.col("targets.user_index") == F.col("candidates.user_index"))
        & (F.col("targets.target_item_index") == F.col("candidates.item_index")),
        "left",
    ).select(F.col("targets.user_index").alias("user_index"), F.col("candidates.rank"))
    return hit_ranks.select(
        "user_index",
        F.when(F.col("rank") <= 10, 1.0).otherwise(0.0).alias("recall_at_10"),
        F.when(F.col("rank") <= 10, 1.0 / F.log2(F.col("rank") + 1))
        .otherwise(0.0)
        .alias("ndcg_at_10"),
        F.when(F.col("rank") <= 10, 1.0 / F.col("rank")).otherwise(0.0).alias("mrr_at_10"),
        F.when(F.col("rank") <= 100, 1.0).otherwise(0.0).alias("candidate_recall_at_100"),
    )


def _paired_uncertainty(
    baseline_candidates: Any,
    candidate_candidates: Any,
    targets: Any,
    config: SparkAlsBenchmarkConfig,
) -> tuple[dict[str, dict[str, float | int | str]], list[dict[str, float | int]]]:
    """Collect bounded user metrics and compute deterministic paired intervals."""
    from pyspark.sql import functions as F

    baseline = _per_user_ranking_metrics(baseline_candidates, targets).select(
        "user_index",
        *[
            F.col(metric).alias(f"baseline_{metric}")
            for metric in ("ndcg_at_10", "recall_at_10", "candidate_recall_at_100")
        ],
    )
    candidate = _per_user_ranking_metrics(candidate_candidates, targets).select(
        "user_index",
        *[
            F.col(metric).alias(f"candidate_{metric}")
            for metric in ("ndcg_at_10", "recall_at_10", "candidate_recall_at_100")
        ],
    )
    rows = baseline.join(candidate, "user_index", "inner").orderBy("user_index").collect()
    if len(rows) != targets.count():
        raise RuntimeError("paired uncertainty did not retain every test user")
    results: dict[str, dict[str, float | int | str]] = {}
    evidence_rows: list[dict[str, float | int]] = []
    for offset, metric in enumerate(("ndcg_at_10", "recall_at_10", "candidate_recall_at_100")):
        estimate = paired_bootstrap_mean_difference(
            (row[f"baseline_{metric}"] for row in rows),
            (row[f"candidate_{metric}"] for row in rows),
            samples=config.bootstrap_samples,
            confidence_level=config.confidence_level,
            seed=config.seed + offset,
        )
        results[metric] = estimate.as_dict()
    for row in rows:
        evidence_rows.append({key: row[key] for key in row.asDict()})
    return results, evidence_rows
