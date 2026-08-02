from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from marketplace_recommender.evaluation.paired_bootstrap import paired_bootstrap_mean_difference
from marketplace_recommender.governance.code_identity import source_tree_sha256
from marketplace_recommender.retrieval.sasrec_lineage import _log_mlflow_model
from marketplace_recommender.retrieval.sasrec_model import (
    SasRecConfig,
    _build_model,
    _seed_everything,
    build_next_item_examples,
)
from marketplace_recommender.retrieval.sasrec_training import (
    _evaluate,
    _indexed_sequences,
    _popularity_evaluate,
    _temporal_targets,
    _train,
)
from marketplace_recommender.retrieval.temporal_cutoffs import exact_temporal_cutoffs

SASREC_IMPLEMENTATION_SOURCES = tuple(
    Path(__file__).with_name(name)
    for name in (
        "sasrec_benchmark.py",
        "sasrec_lineage.py",
        "sasrec_model.py",
        "sasrec_torch.py",
        "sasrec_training.py",
        "temporal_cutoffs.py",
    )
)


def train_sasrec_benchmark(
    spark: Any,
    catalog: str,
    schema_prefix: str,
    model_artifact_path: str,
    job_run_id: str,
    job_id: str,
    config: SasRecConfig | None = None,
) -> dict[str, Any]:
    """Train and certify a causal self-attention sequential recommender."""
    import mlflow
    import torch
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    config = config or SasRecConfig()
    config.validate()
    _seed_everything(torch, config.seed)
    prefix = f"{schema_prefix}_" if schema_prefix else ""
    silver_table = f"{catalog}.{prefix}silver.silver_interactions"
    features_schema = f"{catalog}.{prefix}features"
    serving_schema = f"{catalog}.{prefix}serving"
    monitoring_schema = f"{catalog}.{prefix}monitoring"
    for schema in (features_schema, serving_schema, monitoring_schema):
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    positive = (
        spark.table(silver_table)
        .where(F.col("verified_purchase") & (F.col("rating") >= 4))
        .select("user_id", "parent_asin", "review_timestamp")
    )
    train_cutoff, validation_cutoff = exact_temporal_cutoffs(
        positive,
        "review_timestamp",
        (0.80, 0.90),
    )
    source_delta_version = int(
        spark.sql(f"DESCRIBE HISTORY {silver_table} LIMIT 1").first().version
    )
    item_support = (
        positive.where(F.col("review_timestamp") < F.lit(train_cutoff))
        .groupBy("parent_asin")
        .agg(F.countDistinct("user_id").alias("user_count"))
        .where(F.col("user_count") >= config.minimum_item_users)
    )
    vocabulary = item_support.select("parent_asin").withColumn(
        "item_index",
        F.row_number().over(Window.orderBy("parent_asin")).cast("int"),
    )
    vocabulary.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(f"{features_schema}.sasrec_item_vocabulary")
    vocabulary = spark.table(f"{features_schema}.sasrec_item_vocabulary")
    item_count = vocabulary.count()
    if item_count < 100:
        raise RuntimeError("sequential vocabulary is unexpectedly small")

    training_sequences = _indexed_sequences(
        positive, vocabulary, train_cutoff, config.maximum_training_users
    )
    training_examples = build_next_item_examples(
        [history for _, history in training_sequences], config.max_sequence_length
    )
    random.Random(config.seed).shuffle(training_examples)
    training_examples = training_examples[: config.maximum_training_examples]
    validation = _temporal_targets(
        positive,
        vocabulary,
        train_cutoff,
        validation_cutoff,
        config.evaluation_user_limit,
    )
    if not validation[0]:
        raise RuntimeError("sequential validation cohort is empty")

    model = _build_model(torch, item_count, config)
    model, best_epoch, epoch_rows = _train(
        torch,
        model,
        training_examples,
        item_count,
        config,
        config.epochs,
        validation,
    )
    validation_sasrec, validation_sasrec_rows, _ = _evaluate(
        torch, model, *validation, config=config
    )
    training_counts: dict[int, int] = {}
    for _, history in training_sequences:
        for item in history:
            training_counts[item] = training_counts.get(item, 0) + 1
    validation_popularity, validation_popularity_rows = _popularity_evaluate(
        validation[1], validation[2], training_counts, config
    )

    final_sequences = _indexed_sequences(
        positive, vocabulary, validation_cutoff, config.maximum_training_users
    )
    final_examples = build_next_item_examples(
        [history for _, history in final_sequences], config.max_sequence_length
    )
    random.Random(config.seed).shuffle(final_examples)
    final_examples = final_examples[: config.maximum_training_examples]
    _seed_everything(torch, config.seed)
    final_model = _build_model(torch, item_count, config)
    final_model, _, _ = _train(torch, final_model, final_examples, item_count, config, best_epoch)
    test = _temporal_targets(
        positive,
        vocabulary,
        validation_cutoff,
        None,
        config.evaluation_user_limit,
    )
    if not test[0]:
        raise RuntimeError("sequential test cohort is empty")
    test_sasrec, test_sasrec_rows, test_recommendations = _evaluate(
        torch, final_model, *test, config=config
    )
    final_counts: dict[int, int] = {}
    for _, history in final_sequences:
        for item in history:
            final_counts[item] = final_counts.get(item, 0) + 1
    test_popularity, test_popularity_rows = _popularity_evaluate(
        test[1], test[2], final_counts, config
    )
    uncertainty = paired_bootstrap_mean_difference(
        (row["ndcg_at_10"] for row in test_popularity_rows),
        (row["ndcg_at_10"] for row in test_sasrec_rows),
        samples=config.bootstrap_samples,
        seed=config.seed,
    ).as_dict()
    implementation_sha256 = source_tree_sha256(SASREC_IMPLEMENTATION_SOURCES)
    definition = {
        "contract_version": "sasrec-temporal-benchmark/v3",
        "implementation_sha256": implementation_sha256,
        "source_table": silver_table,
        "source_delta_version": source_delta_version,
        "train_cutoff": train_cutoff,
        "validation_cutoff": validation_cutoff,
        "config": asdict(config),
        "best_epoch_selected_on_validation": best_epoch,
    }
    benchmark_id = hashlib.sha256(
        json.dumps(definition, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    model_path = f"{model_artifact_path.rstrip('/')}/sasrec_encoder.pt"
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "config": asdict(config),
            "item_count": item_count,
            "benchmark_id": benchmark_id,
        },
        model_path,
    )
    vocabulary_rows = vocabulary.orderBy("item_index").collect()
    embedding_values = final_model.item_embedding.weight.detach().numpy()
    embedding_rows = [
        {
            "benchmark_id": benchmark_id,
            "parent_asin": row.parent_asin,
            "item_index": int(row.item_index),
            "features": [float(value) for value in embedding_values[int(row.item_index)]],
        }
        for row in vocabulary_rows
    ]
    spark.createDataFrame(embedding_rows).write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(f"{features_schema}.sasrec_item_embeddings")
    item_lookup = {int(row.item_index): row.parent_asin for row in vocabulary_rows}
    recommendation_rows: list[dict[str, Any]] = []
    for user_id, target, recommendations in zip(
        test[0], test[2], test_recommendations, strict=True
    ):
        for rank, (item_index, score) in enumerate(
            recommendations[: config.recommendation_k], start=1
        ):
            recommendation_rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "user_id": user_id,
                    "parent_asin": item_lookup[item_index],
                    "rank": rank,
                    "model_score": score,
                    "target_parent_asin": item_lookup[target],
                    "is_heldout_target": item_index == target,
                }
            )
    spark.createDataFrame(recommendation_rows).write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(f"{serving_schema}.sasrec_offline_recommendations")
    checks = {
        "global_temporal_cutoffs_are_ordered": train_cutoff < validation_cutoff,
        "training_examples_are_nonempty": bool(training_examples and final_examples),
        "validation_and_test_are_nonempty": bool(validation[0] and test[0]),
        "epoch_selection_uses_validation_only": 1 <= best_epoch <= config.epochs,
        "test_users_have_two_or_more_history_events": all(len(history) >= 2 for history in test[1]),
        "test_targets_are_novel_to_history": all(
            target not in history for history, target in zip(test[1], test[2], strict=True)
        ),
        "item_embeddings_cover_vocabulary": len(embedding_rows) == item_count,
        "recommendations_have_fixed_width": len(recommendation_rows)
        == len(test[0]) * config.recommendation_k,
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    summary: dict[str, Any] = {
        **definition,
        "benchmark_id": benchmark_id,
        "job_id": str(job_id),
        "job_run_id": str(job_run_id),
        "item_count": item_count,
        "training_users": len(training_sequences),
        "training_examples": len(training_examples),
        "final_training_users": len(final_sequences),
        "final_training_examples": len(final_examples),
        "validation_users": len(validation[0]),
        "test_users": len(test[0]),
        "validation": {
            "sasrec": validation_sasrec,
            "popularity": validation_popularity,
            "epoch_search": epoch_rows,
        },
        "test": {
            "sasrec": test_sasrec,
            "popularity": test_popularity,
            "paired_ndcg_uncertainty": uncertainty,
        },
        "artifacts": {
            "model_path": model_path,
            "vocabulary_table": f"{features_schema}.sasrec_item_vocabulary",
            "item_embeddings_table": f"{features_schema}.sasrec_item_embeddings",
            "recommendations_table": f"{serving_schema}.sasrec_offline_recommendations",
        },
        "checks": checks,
        "failed_checks": failed_checks,
    }
    summary_path = f"{model_artifact_path.rstrip('/')}/sasrec_benchmark_summary.json"
    Path(summary_path).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["artifacts"]["summary_path"] = summary_path
    lineage = _log_mlflow_model(
        mlflow,
        torch,
        final_model,
        positive,
        silver_table,
        source_delta_version,
        config,
        summary,
    )
    summary["mlflow_lineage"] = lineage
    summary["deployment_decision"] = {
        "status": "candidate" if lineage["alias"] == "candidate" else "rejected",
        "basis": "validation SASRec NDCG@10 must exceed popularity",
        "test_was_used_for_promotion": False,
    }
    Path(summary_path).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lineage_row = {
        "benchmark_id": benchmark_id,
        "job_id": str(job_id),
        "job_run_id": str(job_run_id),
        "source_table": silver_table,
        "source_delta_version": source_delta_version,
        "mlflow_run_id": lineage["run_id"],
        "registered_model_name": lineage["registered_model_name"],
        "registered_model_version": lineage["registered_model_version"],
        "model_alias": lineage["alias"] or "",
        "deployment_status": summary["deployment_decision"]["status"],
        "promotion_basis": lineage["promotion_basis"],
        "summary_json": json.dumps(summary, sort_keys=True, separators=(",", ":")),
        "created_at_epoch_ms": int(time.time() * 1_000),
    }
    spark.createDataFrame([lineage_row]).write.format("delta").mode("append").saveAsTable(
        f"{monitoring_schema}.model_deployment_lineage"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failed_checks:
        raise RuntimeError("SASRec certification failed: " + ", ".join(failed_checks))
    return summary
