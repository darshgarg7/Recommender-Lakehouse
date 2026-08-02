from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from marketplace_recommender.evaluation.paired_bootstrap import (
    paired_bootstrap_mean_difference,
)


@dataclass(frozen=True)
class SparkAlsBenchmarkConfig:
    """Configuration for the distributed temporal ALS benchmark."""

    rank: int = 64
    max_iter: int = 12
    reg_param: float = 0.08
    alpha: float = 20.0
    seed: int = 20250308
    validation_fraction: float = 0.10
    test_fraction: float = 0.10
    min_user_items: int = 2
    min_item_users: int = 2
    recommendation_k: int = 10
    candidate_k: int = 200
    evaluation_user_limit: int = 10_000
    rrf_constant: int = 60
    rrf_als_weights: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    bootstrap_samples: int = 10_000
    confidence_level: float = 0.95

    def validate(self) -> None:
        if self.rank <= 0 or self.max_iter <= 0:
            raise ValueError("rank and max_iter must be positive")
        if self.reg_param <= 0 or self.alpha <= 0:
            raise ValueError("reg_param and alpha must be positive")
        if self.validation_fraction <= 0 or self.test_fraction <= 0:
            raise ValueError("validation and test fractions must be positive")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation and test fractions must sum to less than one")
        if self.min_user_items < 2 or self.min_item_users < 2:
            raise ValueError("the collaborative core requires at least two users and items")
        if self.recommendation_k <= 0 or self.candidate_k < self.recommendation_k:
            raise ValueError("candidate_k must be at least recommendation_k")
        if self.evaluation_user_limit <= 0 or self.rrf_constant <= 0:
            raise ValueError("evaluation_user_limit and rrf_constant must be positive")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if not self.rrf_als_weights or any(
            not 0.0 <= value <= 1.0 for value in self.rrf_als_weights
        ):
            raise ValueError("RRF ALS weights must be in [0, 1]")


def benchmark_fingerprint(payload: dict[str, Any]) -> str:
    """Content-address a benchmark definition without run-time metadata."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def ndcg_for_single_relevant_rank(rank: int | None, k: int = 10) -> float:
    if rank is None or rank <= 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _merge_by_key(spark: Any, frame: Any, table: str, key: str) -> None:
    if not spark.catalog.tableExists(table):
        frame.write.format("delta").mode("append").saveAsTable(table)
        return
    from delta.tables import DeltaTable

    (
        DeltaTable.forName(spark, table)
        .alias("target")
        .merge(frame.alias("source"), f"target.{key} = source.{key}")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )


def _materialize(spark: Any, frame: Any, table: str) -> Any:
    """Materialize reusable state without Spark cache, which serverless forbids."""
    frame.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        table
    )
    return spark.table(table)


def _prepare_training(positive: Any, cutoff: int, config: SparkAlsBenchmarkConfig) -> Any:
    """Build a two-sided collaborative core and aggregate implicit confidence."""
    from pyspark.sql import functions as F

    raw = positive.where(F.col("review_timestamp") < F.lit(cutoff))
    eligible_users = (
        raw.groupBy("user_id")
        .agg(F.countDistinct("parent_asin").alias("item_count"))
        .where(F.col("item_count") >= config.min_user_items)
        .select("user_id")
    )
    user_core = raw.join(eligible_users, "user_id", "inner")
    eligible_items = (
        user_core.groupBy("parent_asin")
        .agg(F.countDistinct("user_id").alias("user_count"))
        .where(F.col("user_count") >= config.min_item_users)
        .select("parent_asin")
    )
    item_core = user_core.join(eligible_items, "parent_asin", "inner")
    # Removing low-support items can reduce a user to one edge. Close the core once
    # more so every learned user factor has at least two distinct observations.
    closed_users = (
        item_core.groupBy("user_id")
        .agg(F.countDistinct("parent_asin").alias("item_count"))
        .where(F.col("item_count") >= config.min_user_items)
        .select("user_id")
    )
    return (
        item_core.join(closed_users, "user_id", "inner")
        .groupBy("user_id", "parent_asin")
        .agg(
            F.count("*").alias("event_count"),
            F.sum(
                F.lit(1.0)
                + F.lit(0.25) * (F.col("rating") - F.lit(4.0))
                + F.lit(0.05) * F.log1p(F.coalesce(F.col("helpful_votes"), F.lit(0)))
            ).alias("confidence"),
        )
    )


def _index_training(train_pairs: Any) -> tuple[Any, Any, Any]:
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    # Spark Connect serializes fitted StringIndexer labels back to the client and
    # caps the model at 256 MiB. Stable SQL ranks keep high-cardinality mappings
    # distributed and make reruns byte-for-byte deterministic.
    users = (
        train_pairs.select("user_id")
        .distinct()
        .withColumn(
            "user_index",
            (F.row_number().over(Window.orderBy("user_id")) - F.lit(1)).cast("int"),
        )
    )
    items = (
        train_pairs.select("parent_asin")
        .distinct()
        .withColumn(
            "item_index",
            (F.row_number().over(Window.orderBy("parent_asin")) - F.lit(1)).cast("int"),
        )
    )
    indexed = train_pairs.join(users, "user_id", "inner").join(items, "parent_asin", "inner")
    return indexed, users, items


def _evaluation_targets(
    positive: Any,
    train_pairs: Any,
    users: Any,
    items: Any,
    start: int,
    end: int | None,
    config: SparkAlsBenchmarkConfig,
) -> Any:
    """Select one chronologically first, warm, previously unseen target per user."""
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    candidates = positive.where(F.col("review_timestamp") >= F.lit(start))
    if end is not None:
        candidates = candidates.where(F.col("review_timestamp") < F.lit(end))
    known_novel = (
        candidates.join(users, "user_id", "inner")
        .join(items, "parent_asin", "inner")
        .join(
            train_pairs.select("user_id", "parent_asin").distinct(),
            ["user_id", "parent_asin"],
            "left_anti",
        )
    )
    earliest = (
        known_novel.withColumn(
            "_target_order",
            F.row_number().over(
                Window.partitionBy("user_id").orderBy("review_timestamp", "parent_asin")
            ),
        )
        .where(F.col("_target_order") == 1)
        .drop("_target_order")
    )
    if earliest.limit(config.evaluation_user_limit + 1).count() > config.evaluation_user_limit:
        earliest = earliest.orderBy(F.xxhash64("user_id")).limit(config.evaluation_user_limit)
    return earliest.withColumnRenamed("item_index", "target_item_index")


def _fit_als(indexed: Any, config: SparkAlsBenchmarkConfig) -> Any:
    from pyspark.ml.recommendation import ALS

    return ALS(
        userCol="user_index",
        itemCol="item_index",
        ratingCol="confidence",
        implicitPrefs=True,
        alpha=config.alpha,
        rank=config.rank,
        maxIter=config.max_iter,
        regParam=config.reg_param,
        nonnegative=True,
        coldStartStrategy="drop",
        seed=config.seed,
        checkpointInterval=2,
        numUserBlocks=10,
        numItemBlocks=10,
    ).fit(indexed.select("user_index", "item_index", "confidence"))


def _als_candidates(
    spark: Any, model: Any, targets: Any, seen: Any, config: SparkAlsBenchmarkConfig
) -> Any:
    import numpy as np
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType
    from pyspark.sql.window import Window

    # Unity Catalog serverless currently rejects MLlib's recommendForUserSubset
    # higher-order-function plan. Score the same learned factors exactly in bounded
    # NumPy blocks; Spark still performs distributed ALS training and all joins,
    # filtering, materialization, metrics, and certification.
    retrieval_width = config.candidate_k + 100
    evaluation_factors = (
        model.userFactors.alias("factors")
        .join(
            targets.select("user_index").distinct().alias("targets"),
            F.col("factors.id") == F.col("targets.user_index"),
            "inner",
        )
        .select(F.col("factors.id").alias("user_index"), F.col("factors.features"))
        .orderBy("user_index")
        .collect()
    )
    item_factor_rows = model.itemFactors.orderBy("id").collect()
    if not evaluation_factors or not item_factor_rows:
        raise RuntimeError("ALS did not produce factors for the temporal evaluation cohort")
    item_ids = np.asarray([row.id for row in item_factor_rows], dtype=np.int32)
    item_matrix = np.asarray([row.features for row in item_factor_rows], dtype=np.float32)
    candidate_rows: list[tuple[int, int, float]] = []
    block_size = 128
    for offset in range(0, len(evaluation_factors), block_size):
        block = evaluation_factors[offset : offset + block_size]
        user_matrix = np.asarray([row.features for row in block], dtype=np.float32)
        scores = user_matrix @ item_matrix.T
        width = min(retrieval_width, scores.shape[1])
        top_columns = np.argpartition(scores, -width, axis=1)[:, -width:]
        top_scores = np.take_along_axis(scores, top_columns, axis=1)
        top_items = item_ids[top_columns]
        order = np.lexsort((top_items, -top_scores), axis=1)
        top_columns = np.take_along_axis(top_columns, order, axis=1)
        for row_offset, user in enumerate(block):
            for column in top_columns[row_offset]:
                candidate_rows.append(
                    (int(user.user_index), int(item_ids[column]), float(scores[row_offset, column]))
                )
    schema = StructType(
        [
            StructField("user_index", IntegerType(), nullable=False),
            StructField("item_index", IntegerType(), nullable=False),
            StructField("model_score", DoubleType(), nullable=False),
        ]
    )
    raw = spark.createDataFrame(candidate_rows, schema=schema).join(
        seen, ["user_index", "item_index"], "left_anti"
    )
    return (
        raw.withColumn(
            "rank",
            F.row_number().over(
                Window.partitionBy("user_index").orderBy(F.desc("model_score"), F.asc("item_index"))
            ),
        )
        .where(F.col("rank") <= config.candidate_k)
        .select("user_index", "item_index", "rank", "model_score")
    )


def _popularity_candidates(
    indexed: Any, targets: Any, seen: Any, config: SparkAlsBenchmarkConfig
) -> tuple[Any, Any]:
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    item_popularity = indexed.groupBy("item_index").agg(
        F.sum("event_count").alias("popularity"),
        F.sum("confidence").alias("popularity_score"),
    )
    popular_pool = (
        item_popularity.orderBy(F.desc("popularity_score"), F.asc("item_index"))
        .limit(config.candidate_k + 200)
        .select("item_index", F.col("popularity_score").alias("model_score"))
    )
    raw = (
        targets.select("user_index")
        .distinct()
        .crossJoin(F.broadcast(popular_pool))
        .join(seen, ["user_index", "item_index"], "left_anti")
    )
    candidates = (
        raw.withColumn(
            "rank",
            F.row_number().over(
                Window.partitionBy("user_index").orderBy(F.desc("model_score"), F.asc("item_index"))
            ),
        )
        .where(F.col("rank") <= config.candidate_k)
        .select("user_index", "item_index", "rank", "model_score")
    )
    return candidates, item_popularity


def _rrf_candidates(
    als: Any, popularity: Any, weight: float, config: SparkAlsBenchmarkConfig
) -> Any:
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    joined = als.select("user_index", "item_index", F.col("rank").alias("als_rank")).join(
        popularity.select("user_index", "item_index", F.col("rank").alias("popularity_rank")),
        ["user_index", "item_index"],
        "full",
    )
    score = F.when(
        F.col("als_rank").isNotNull(),
        F.lit(weight) / (F.lit(config.rrf_constant) + F.col("als_rank")),
    ).otherwise(F.lit(0.0)) + F.when(
        F.col("popularity_rank").isNotNull(),
        F.lit(1.0 - weight) / (F.lit(config.rrf_constant) + F.col("popularity_rank")),
    ).otherwise(F.lit(0.0))
    return (
        joined.withColumn("model_score", score)
        .withColumn(
            "rank",
            F.row_number().over(
                Window.partitionBy("user_index").orderBy(F.desc("model_score"), F.asc("item_index"))
            ),
        )
        .where(F.col("rank") <= config.candidate_k)
        .select("user_index", "item_index", "rank", "model_score")
    )


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


def _fit_and_retrieve(
    spark: Any,
    positive: Any,
    cutoff: int,
    evaluation_start: int,
    evaluation_end: int | None,
    config: SparkAlsBenchmarkConfig,
    scratch_schema: str,
    stage: str,
) -> dict[str, Any]:
    scratch_tables: list[str] = []

    def materialize(name: str, frame: Any) -> Any:
        table = f"{scratch_schema}.als_{stage}_{name}"
        scratch_tables.append(table)
        return _materialize(spark, frame, table)

    train_pairs = materialize("train_pairs", _prepare_training(positive, cutoff, config))
    indexed, users, items = _index_training(train_pairs)
    indexed = materialize("indexed", indexed)
    users = indexed.select("user_id", "user_index").distinct()
    items = indexed.select("parent_asin", "item_index").distinct()
    targets = materialize(
        "targets",
        _evaluation_targets(
            positive,
            train_pairs,
            users,
            items,
            evaluation_start,
            evaluation_end,
            config,
        ),
    )
    if targets.count() == 0:
        raise RuntimeError("temporal evaluation produced no warm, novel targets")
    model = _fit_als(indexed, config)
    seen = materialize("seen", indexed.select("user_index", "item_index").distinct())
    als = materialize("als_candidates", _als_candidates(spark, model, targets, seen, config))
    popularity_plan, item_popularity_plan = _popularity_candidates(indexed, targets, seen, config)
    item_popularity = materialize("item_popularity", item_popularity_plan)
    popularity = materialize("popularity_candidates", popularity_plan)
    return {
        "model": model,
        "train_pairs": train_pairs,
        "indexed": indexed,
        "users": users,
        "items": items,
        "targets": targets,
        "seen": seen,
        "als": als,
        "popularity": popularity,
        "item_popularity": item_popularity,
        "train_events": int(train_pairs.agg({"event_count": "sum"}).first()[0]),
        "train_pairs_count": train_pairs.count(),
        "train_users": users.count(),
        "train_items": items.count(),
        "scratch_tables": scratch_tables,
    }


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
    train_cutoff, validation_cutoff = (
        int(value)
        for value in positive.approxQuantile(
            "review_timestamp", [train_quantile, validation_quantile], 0.0001
        )
    )
    maximum_timestamp = int(positive.agg(F.max("review_timestamp")).first()[0])
    delta_version = int(spark.sql(f"DESCRIBE HISTORY {silver_table} LIMIT 1").first().version)
    implementation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    bootstrap_path = Path(__file__).parents[1] / "evaluation" / "paired_bootstrap.py"
    bootstrap_implementation_sha256 = hashlib.sha256(bootstrap_path.read_bytes()).hexdigest()
    definition = {
        "contract_version": "spark-als-temporal-benchmark/v4",
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
                "contract_version": "spark-als-temporal-benchmark/v4",
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
