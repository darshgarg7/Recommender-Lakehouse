from __future__ import annotations

from typing import Any

from marketplace_recommender.retrieval.spark_als_config import SparkAlsBenchmarkConfig


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
