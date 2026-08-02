from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from marketplace_recommender.governance.code_identity import source_tree_sha256
from marketplace_recommender.retrieval.spark_als_config import (
    SparkAlsBenchmarkConfig,
    benchmark_fingerprint,
)
from marketplace_recommender.retrieval.spark_als_data import (
    _fit_and_retrieve,
    _materialize,
    _merge_by_key,
    _rrf_candidates,
)
from marketplace_recommender.retrieval.spark_als_metrics import (
    _paired_uncertainty,
    _ranking_metrics,
)
from marketplace_recommender.retrieval.temporal_cutoffs import exact_temporal_cutoffs

ALS_IMPLEMENTATION_SOURCES = tuple(
    Path(__file__).with_name(name)
    for name in (
        "spark_als.py",
        "spark_als_benchmark.py",
        "spark_als_config.py",
        "spark_als_data.py",
        "spark_als_metrics.py",
        "temporal_cutoffs.py",
    )
)


def train_spark_als_benchmark(
    spark: Any,
    catalog: str,
    schema_prefix: str,
    model_artifact_path: str,
    job_run_id: str,
    job_id: str,
    config: SparkAlsBenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Train, tune, refit, evaluate, and certify a distributed implicit ALS model."""
    from pyspark.sql import functions as F

    config = config or SparkAlsBenchmarkConfig()
    config.validate()
    prefix = f"{schema_prefix}_" if schema_prefix else ""
    silver_table = f"{catalog}.{prefix}silver.silver_interactions"
    features_schema = f"{catalog}.{prefix}features"
    serving_schema = f"{catalog}.{prefix}serving"
    monitoring_schema = f"{catalog}.{prefix}monitoring"
    scratch_schema = f"{catalog}.{prefix}scratch"
    for schema in (features_schema, serving_schema, monitoring_schema, scratch_schema):
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")

    started = time.perf_counter()
    interactions = spark.table(silver_table)
    positive = interactions.where(F.col("verified_purchase") & (F.col("rating") >= 4)).select(
        "user_id",
        "parent_asin",
        "review_timestamp",
        "rating",
        "helpful_votes",
    )
    positive_events = positive.count()
    train_quantile = 1.0 - config.validation_fraction - config.test_fraction
    validation_quantile = 1.0 - config.test_fraction
    train_cutoff, validation_cutoff = exact_temporal_cutoffs(
        positive,
        "review_timestamp",
        (train_quantile, validation_quantile),
    )
    maximum_timestamp = int(positive.agg(F.max("review_timestamp")).first()[0])
    delta_version = int(spark.sql(f"DESCRIBE HISTORY {silver_table} LIMIT 1").first().version)
    implementation_sha256 = source_tree_sha256(ALS_IMPLEMENTATION_SOURCES)
    bootstrap_path = Path(__file__).parents[1] / "evaluation" / "paired_bootstrap.py"
    bootstrap_implementation_sha256 = hashlib.sha256(bootstrap_path.read_bytes()).hexdigest()
    definition = {
        "contract_version": "spark-als-temporal-benchmark/v6",
        "implementation_sha256": implementation_sha256,
        "bootstrap_implementation_sha256": bootstrap_implementation_sha256,
        "source_table": silver_table,
        "source_delta_version": delta_version,
        "positive_definition": "verified_purchase = true AND rating >= 4",
        "train_cutoff": train_cutoff,
        "validation_cutoff": validation_cutoff,
        "maximum_timestamp": maximum_timestamp,
        "config": asdict(config),
    }
    benchmark_id = benchmark_fingerprint(definition)

    validation_started = time.perf_counter()
    validation = _fit_and_retrieve(
        spark,
        positive,
        train_cutoff,
        train_cutoff,
        validation_cutoff,
        config,
        scratch_schema,
        "validation",
    )
    validation_rows: list[dict[str, Any]] = []
    for weight in config.rrf_als_weights:
        candidate_frame = _rrf_candidates(
            validation["als"], validation["popularity"], weight, config
        )
        candidate_table = f"{scratch_schema}.als_validation_rrf_{int(weight * 100):03d}"
        validation["scratch_tables"].append(candidate_table)
        candidate_frame = _materialize(spark, candidate_frame, candidate_table)
        metrics = _ranking_metrics(
            candidate_frame,
            validation["targets"],
            validation["item_popularity"],
            validation["train_items"],
            config,
        )
        validation_rows.append({"rrf_als_weight": weight, **metrics})
    selected = max(
        validation_rows,
        key=lambda row: (
            row["ndcg_at_10"],
            row["recall_at_10"],
            row["candidate_recall_at_100"],
            -abs(row["rrf_als_weight"] - 0.5),
        ),
    )
    selected_weight = float(selected["rrf_als_weight"])
    validation_champion = (
        "popularity"
        if selected_weight == 0.0
        else "spark_implicit_als"
        if selected_weight == 1.0
        else "temporal_hybrid_rrf"
    )
    validation_seconds = time.perf_counter() - validation_started

    test_started = time.perf_counter()
    final = _fit_and_retrieve(
        spark,
        positive,
        validation_cutoff,
        validation_cutoff,
        None,
        config,
        scratch_schema,
        "test",
    )
    hybrid_table = f"{scratch_schema}.als_test_hybrid"
    final["scratch_tables"].append(hybrid_table)
    hybrid = _materialize(
        spark,
        _rrf_candidates(final["als"], final["popularity"], selected_weight, config),
        hybrid_table,
    )
    candidate_frames = {
        "popularity": final["popularity"],
        "spark_implicit_als": final["als"],
        "temporal_hybrid_rrf": hybrid,
    }
    test_metrics = {
        name: _ranking_metrics(
            candidates,
            final["targets"],
            final["item_popularity"],
            final["train_items"],
            config,
        )
        for name, candidates in candidate_frames.items()
    }
    paired_uncertainty, paired_user_rows = _paired_uncertainty(
        final["popularity"], hybrid, final["targets"], config
    )
    # Validation chooses the candidate and test is touched once as a release gate.
    # Test never tunes a hyperparameter or substitutes another learned candidate.
    primary_interval = paired_uncertainty["ndcg_at_10"]
    release_qualified = validation_champion == "popularity" or (
        validation_champion == "temporal_hybrid_rrf" and float(primary_interval["lower"]) > 0.0
    )
    serving_champion = validation_champion if release_qualified else "popularity"
    test_seconds = time.perf_counter() - test_started

    model_root = model_artifact_path.rstrip("/")
    final["model"].write().overwrite().save(f"{model_root}/spark_implicit_als")
    user_factors = (
        final["model"]
        .userFactors.withColumnRenamed("id", "user_index")
        .join(final["users"], "user_index", "inner")
        .withColumn("benchmark_id", F.lit(benchmark_id))
    )
    item_factors = (
        final["model"]
        .itemFactors.withColumnRenamed("id", "item_index")
        .join(final["items"], "item_index", "inner")
        .withColumn("benchmark_id", F.lit(benchmark_id))
    )
    user_factors.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(f"{features_schema}.als_user_factors")
    item_factors.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(f"{features_schema}.als_item_factors")

    champion_recommendations = (
        candidate_frames[serving_champion]
        .where(F.col("rank") <= config.recommendation_k)
        .join(final["items"], "item_index", "inner")
        .join(
            final["targets"].select(
                "user_index",
                "user_id",
                F.col("parent_asin").alias("target_parent_asin"),
                F.col("review_timestamp").alias("target_timestamp"),
            ),
            "user_index",
            "inner",
        )
        .withColumn("benchmark_id", F.lit(benchmark_id))
        .withColumn("serving_champion", F.lit(serving_champion))
        .withColumn("is_heldout_target", F.col("parent_asin") == F.col("target_parent_asin"))
        .select(
            "benchmark_id",
            "user_id",
            "parent_asin",
            "rank",
            "model_score",
            "serving_champion",
            "target_parent_asin",
            "target_timestamp",
            "is_heldout_target",
        )
    )
    champion_recommendations.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(f"{serving_schema}.als_offline_recommendations")

    total_seconds = time.perf_counter() - started
    common = {
        "benchmark_id": benchmark_id,
        "job_id": str(job_id),
        "job_run_id": str(job_run_id),
        "source_table": silver_table,
        "source_delta_version": delta_version,
        "positive_events": positive_events,
        "train_cutoff": train_cutoff,
        "validation_cutoff": validation_cutoff,
        "maximum_timestamp": maximum_timestamp,
        "als_rank": config.rank,
        "als_max_iter": config.max_iter,
        "als_reg_param": config.reg_param,
        "als_alpha": config.alpha,
        "created_at_epoch_ms": int(time.time() * 1000),
    }
    metric_rows: list[dict[str, Any]] = []
    for row in validation_rows:
        model_name = f"temporal_hybrid_rrf_alpha_{row['rrf_als_weight']:.2f}"
        metric_rows.append(
            {
                **common,
                "row_id": benchmark_fingerprint(
                    {"benchmark_id": benchmark_id, "split": "validation", "model": model_name}
                ),
                "split": "validation",
                "model_name": model_name,
                "rrf_als_weight": float(row["rrf_als_weight"]),
                "train_events": validation["train_events"],
                "train_pairs": validation["train_pairs_count"],
                "train_users": validation["train_users"],
                "train_items": validation["train_items"],
                "evaluation_users": row["evaluation_users"],
                "candidate_recall_at_100": row["candidate_recall_at_100"],
                "recall_at_10": row["recall_at_10"],
                "ndcg_at_10": row["ndcg_at_10"],
                "mrr_at_10": row["mrr_at_10"],
                "catalog_coverage_at_10": row["catalog_coverage_at_10"],
                "long_tail_exposure_at_10": row["long_tail_exposure_at_10"],
                "is_selected": row["rrf_als_weight"] == selected_weight,
                "is_champion": False,
            }
        )
    for name, metrics in test_metrics.items():
        metric_rows.append(
            {
                **common,
                "row_id": benchmark_fingerprint(
                    {"benchmark_id": benchmark_id, "split": "test", "model": name}
                ),
                "split": "test",
                "model_name": name,
                "rrf_als_weight": selected_weight if name == "temporal_hybrid_rrf" else None,
                "train_events": final["train_events"],
                "train_pairs": final["train_pairs_count"],
                "train_users": final["train_users"],
                "train_items": final["train_items"],
                "evaluation_users": metrics["evaluation_users"],
                "candidate_recall_at_100": metrics["candidate_recall_at_100"],
                "recall_at_10": metrics["recall_at_10"],
                "ndcg_at_10": metrics["ndcg_at_10"],
                "mrr_at_10": metrics["mrr_at_10"],
                "catalog_coverage_at_10": metrics["catalog_coverage_at_10"],
                "long_tail_exposure_at_10": metrics["long_tail_exposure_at_10"],
                "is_selected": name == "temporal_hybrid_rrf",
                "is_champion": name == serving_champion,
            }
        )
    metrics_frame = spark.createDataFrame(metric_rows)
    _merge_by_key(
        spark, metrics_frame, f"{monitoring_schema}.recommender_benchmark_metrics", "row_id"
    )
    paired_user_frame = (
        spark.createDataFrame(paired_user_rows)
        .withColumn("benchmark_id", F.lit(benchmark_id))
        .withColumn("baseline_model", F.lit("popularity"))
        .withColumn("candidate_model", F.lit("temporal_hybrid_rrf"))
    )
    paired_user_frame.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(f"{monitoring_schema}.recommender_paired_user_metrics")

    uncertainty_rows = [
        {
            "row_id": benchmark_fingerprint(
                {
                    "benchmark_id": benchmark_id,
                    "comparison": "hybrid_vs_popularity",
                    "metric": metric,
                }
            ),
            "benchmark_id": benchmark_id,
            "baseline_model": "popularity",
            "candidate_model": "temporal_hybrid_rrf",
            "metric": metric,
            **estimate,
            "created_at_epoch_ms": int(time.time() * 1000),
        }
        for metric, estimate in paired_uncertainty.items()
    ]
    _merge_by_key(
        spark,
        spark.createDataFrame(uncertainty_rows),
        f"{monitoring_schema}.recommender_benchmark_uncertainty",
        "row_id",
    )

    unseen_leakage_rows = champion_recommendations.join(
        final["train_pairs"].select("user_id", "parent_asin").distinct(),
        ["user_id", "parent_asin"],
        "inner",
    ).count()
    recommendation_rows = champion_recommendations.count()
    duplicate_rank_rows = (
        recommendation_rows - champion_recommendations.select("user_id", "rank").distinct().count()
    )
    checks = {
        "temporal_cutoffs_are_ordered": train_cutoff < validation_cutoff <= maximum_timestamp,
        "distributed_training_is_nonempty": final["train_events"] > 0,
        "collaborative_core_has_multiple_users": final["train_users"] > 1,
        "collaborative_core_has_multiple_items": final["train_items"] > 1,
        "validation_targets_are_nonempty": validation["targets"].count() > 0,
        "test_targets_are_nonempty": final["targets"].count() > 0,
        "user_factors_cover_training_users": user_factors.count() == final["train_users"],
        "item_factors_cover_training_items": item_factors.count() == final["train_items"],
        "recommendations_exclude_training_history": unseen_leakage_rows == 0,
        "recommendation_ranks_are_unique": duplicate_rank_rows == 0,
        "all_comparators_were_evaluated": set(test_metrics)
        == {"popularity", "spark_implicit_als", "temporal_hybrid_rrf"},
        "candidate_was_selected_on_validation": validation_champion
        in {"popularity", "spark_implicit_als", "temporal_hybrid_rrf"},
        "paired_bootstrap_covers_every_test_user": len(paired_user_rows)
        == final["targets"].count(),
        "test_did_not_tune_the_candidate": validation_champion
        == (
            "popularity"
            if selected_weight == 0.0
            else "spark_implicit_als"
            if selected_weight == 1.0
            else "temporal_hybrid_rrf"
        ),
        "release_gate_retains_baseline_on_uncertain_ndcg": release_qualified
        or serving_champion == "popularity",
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    summary = {
        **definition,
        "benchmark_id": benchmark_id,
        "job_id": str(job_id),
        "job_run_id": str(job_run_id),
        "positive_events": positive_events,
        "validation": {
            "train_events": validation["train_events"],
            "train_pairs": validation["train_pairs_count"],
            "train_users": validation["train_users"],
            "train_items": validation["train_items"],
            "evaluation_users": validation["targets"].count(),
            "rrf_search": validation_rows,
            "selected_als_weight": selected_weight,
            "runtime_seconds": validation_seconds,
        },
        "test": {
            "train_events": final["train_events"],
            "train_pairs": final["train_pairs_count"],
            "train_users": final["train_users"],
            "train_items": final["train_items"],
            "evaluation_users": final["targets"].count(),
            "metrics": test_metrics,
            "validation_selected_candidate": validation_champion,
            "release_qualified": release_qualified,
            "serving_champion": serving_champion,
            "release_rule": "lower bound of paired 95% NDCG@10 delta must exceed zero",
            "recommendation_rows": recommendation_rows,
            "paired_uncertainty_vs_popularity": paired_uncertainty,
            "runtime_seconds": test_seconds,
        },
        "artifacts": {
            "model_path": f"{model_root}/spark_implicit_als",
            "user_factors_table": f"{features_schema}.als_user_factors",
            "item_factors_table": f"{features_schema}.als_item_factors",
            "recommendations_table": f"{serving_schema}.als_offline_recommendations",
            "metrics_table": f"{monitoring_schema}.recommender_benchmark_metrics",
            "paired_user_metrics_table": f"{monitoring_schema}.recommender_paired_user_metrics",
            "uncertainty_table": f"{monitoring_schema}.recommender_benchmark_uncertainty",
        },
        "checks": checks,
        "failed_checks": failed_checks,
        "total_runtime_seconds": total_seconds,
    }
    summary_sha256 = benchmark_fingerprint(summary)
    from pyspark.sql.types import (
        ArrayType,
        BooleanType,
        LongType,
        StringType,
        StructField,
        StructType,
    )

    certificate_schema = StructType(
        [
            StructField("benchmark_id", StringType(), nullable=False),
            StructField("contract_version", StringType(), nullable=False),
            StructField("job_id", StringType(), nullable=False),
            StructField("job_run_id", StringType(), nullable=False),
            StructField("passed", BooleanType(), nullable=False),
            StructField("check_count", LongType(), nullable=False),
            StructField("failed_check_count", LongType(), nullable=False),
            StructField(
                "failed_checks", ArrayType(StringType(), containsNull=False), nullable=False
            ),
            StructField("summary_sha256", StringType(), nullable=False),
            StructField("summary_json", StringType(), nullable=False),
            StructField("created_at_epoch_ms", LongType(), nullable=False),
        ]
    )
    certificate = spark.createDataFrame(
        [
            {
                "benchmark_id": benchmark_id,
                "contract_version": "spark-als-temporal-benchmark/v6",
                "job_id": str(job_id),
                "job_run_id": str(job_run_id),
                "passed": not failed_checks,
                "check_count": len(checks),
                "failed_check_count": len(failed_checks),
                "failed_checks": failed_checks,
                "summary_sha256": summary_sha256,
                "summary_json": json.dumps(summary, sort_keys=True, separators=(",", ":")),
                "created_at_epoch_ms": int(time.time() * 1000),
            }
        ],
        schema=certificate_schema,
    )
    _merge_by_key(
        spark,
        certificate,
        f"{monitoring_schema}.recommender_benchmark_certifications",
        "benchmark_id",
    )
    print(json.dumps({**summary, "summary_sha256": summary_sha256}, indent=2, sort_keys=True))
    if failed_checks:
        raise RuntimeError(
            "recommender benchmark certification failed: " + ", ".join(failed_checks)
        )
    for table in validation["scratch_tables"] + final["scratch_tables"]:
        spark.sql(f"DROP TABLE IF EXISTS {table}")
    return summary
