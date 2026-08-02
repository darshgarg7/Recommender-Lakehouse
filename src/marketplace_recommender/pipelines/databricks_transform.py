from __future__ import annotations

from typing import Any

from marketplace_recommender.pipelines.databricks_common import _ensure_schemas, _merge_table


def build_silver_tables(spark: Any, catalog: str, schema_prefix: str = "") -> None:
    """Canonicalize parent products and interactions using explicit source contracts."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        ArrayType,
        BooleanType,
        DoubleType,
        LongType,
        MapType,
        StringType,
        StructField,
        StructType,
    )

    schemas = _ensure_schemas(spark, catalog, schema_prefix)
    product_schema = StructType(
        [
            StructField("parent_asin", StringType()),
            StructField("asin", StringType()),
            StructField("title", StringType()),
            StructField("store", StringType()),
            StructField("main_category", StringType()),
            StructField("categories", ArrayType(StringType())),
            StructField("description", ArrayType(StringType())),
            StructField("features", ArrayType(StringType())),
            StructField("details", MapType(StringType(), StringType())),
            StructField("price", DoubleType()),
            StructField("bought_together", ArrayType(StringType())),
        ]
    )
    review_schema = StructType(
        [
            StructField("user_id", StringType()),
            StructField("asin", StringType()),
            StructField("parent_asin", StringType()),
            StructField("rating", DoubleType()),
            StructField("verified_purchase", BooleanType()),
            StructField("timestamp", LongType()),
            StructField("title", StringType()),
            StructField("text", StringType()),
            StructField("helpful_vote", LongType()),
        ]
    )
    raw_products = spark.table(f"{schemas['bronze']}.bronze_product_metadata")
    products = (
        raw_products.withColumn("record", F.from_json("raw_payload", product_schema))
        .select(
            F.coalesce("record.parent_asin", "record.asin").alias("parent_asin"),
            F.regexp_extract("source_file", r"meta_([^/.]+)", 1).alias("domain"),
            "record.title",
            F.col("record.store").alias("brand_or_store"),
            F.col("record.categories").alias("category_path"),
            F.col("record.description"),
            F.col("record.features").alias("feature_bullets"),
            F.col("record.details").alias("structured_attributes"),
            F.col("record.price").alias("crawl_price"),
            "source_file",
            F.col("ingested_at").alias("processed_at"),
            F.lit("valid").alias("quality_status"),
        )
        .where(F.col("parent_asin").isNotNull())
        .dropDuplicates(["parent_asin"])
    )
    _merge_table(spark, products, f"{schemas['silver']}.silver_products", "parent_asin")
    variants = (
        raw_products.withColumn("record", F.from_json("raw_payload", product_schema))
        .select("record.asin", F.coalesce("record.parent_asin", "record.asin").alias("parent_asin"))
        .where(F.col("asin").isNotNull())
        .dropDuplicates(["asin"])
    )
    _merge_table(spark, variants, f"{schemas['silver']}.silver_product_variants", "asin")
    reviews = spark.table(f"{schemas['bronze']}.bronze_reviews").withColumn(
        "record", F.from_json("raw_payload", review_schema)
    )
    resolved_parent = F.coalesce(
        F.col("record.parent_asin"), variants.parent_asin, F.col("record.asin")
    )
    interactions = (
        reviews.join(F.broadcast(variants), F.col("record.asin") == variants.asin, "left")
        .select(
            F.sha2(
                F.concat_ws(
                    "\u001f", "record.user_id", "record.asin", "record.timestamp", "record.rating"
                ),
                256,
            ).alias("interaction_id"),
            F.col("record.user_id").alias("user_id"),
            F.col("record.asin").alias("asin"),
            resolved_parent.alias("parent_asin"),
            F.regexp_extract("source_file", r"reviews_([^/.]+)", 1).alias("domain"),
            F.col("record.rating").alias("rating"),
            F.col("record.verified_purchase").alias("verified_purchase"),
            F.col("record.timestamp").alias("review_timestamp"),
            F.col("record.title").alias("review_title"),
            F.col("record.text").alias("review_text"),
            F.col("record.helpful_vote").alias("helpful_votes"),
            "source_file",
            F.col("ingested_at").alias("processed_at"),
            F.when(
                F.coalesce(F.col("record.parent_asin"), variants.parent_asin).isNull(),
                "missing_product_metadata",
            )
            .otherwise("valid")
            .alias("quality_status"),
        )
        .where(
            F.col("user_id").isNotNull()
            & F.col("parent_asin").isNotNull()
            & F.col("rating").between(1, 5)
            & F.col("review_timestamp").between(830_908_800_000, 1_696_118_400_000)
        )
        .dropDuplicates(["interaction_id"])
    )
    _merge_table(spark, interactions, f"{schemas['silver']}.silver_interactions", "interaction_id")


def build_gold_tables(spark: Any, catalog: str, schema_prefix: str = "") -> None:
    """Materialize point-in-time Gold tables with shuffle-bounded window plans."""
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    schemas = _ensure_schemas(spark, catalog, schema_prefix)
    interactions = spark.table(f"{schemas['silver']}.silver_interactions")
    labels = interactions.where(
        (F.col("verified_purchase") == F.lit(True)) & (F.col("rating") != 3)
    ).select(
        "interaction_id",
        "user_id",
        "parent_asin",
        F.col("review_timestamp").alias("label_timestamp"),
        F.when(F.col("rating") >= 4, 1).otherwise(0).alias("label"),
        "rating",
    )
    labels.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        f"{schemas['gold']}.gold_training_labels"
    )

    # A millisecond RANGE ending at -1 excludes every event tied with the label
    # timestamp. Unlike the former label-to-history self-join, this requires one
    # partitioned sort rather than materializing a potentially quadratic join.
    user_history_window = (
        Window.partitionBy("user_id")
        .orderBy(F.col("review_timestamp"))
        .rangeBetween(Window.unboundedPreceding, -1)
    )
    histories = interactions.select(
        "interaction_id",
        "user_id",
        "review_timestamp",
        "parent_asin",
    ).withColumn(
        "historical_events",
        F.slice(
            F.collect_list(
                F.struct(
                    F.col("review_timestamp").alias("ts"),
                    F.col("parent_asin").alias("item"),
                )
            ).over(user_history_window),
            -100,
            100,
        ),
    )
    sequences = labels.select("interaction_id", "user_id", "label_timestamp").join(
        histories.select("interaction_id", "historical_events"),
        "interaction_id",
        "inner",
    )
    sequences.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        f"{schemas['gold']}.gold_user_sequences_asof"
    )

    item_history_window = (
        Window.partitionBy("parent_asin")
        .orderBy(F.col("review_timestamp"))
        .rangeBetween(Window.unboundedPreceding, -1)
    )
    item_features = (
        interactions.select(
            "interaction_id",
            "parent_asin",
            "review_timestamp",
            "rating",
            "verified_purchase",
        )
        .withColumn("feature_timestamp", F.max("review_timestamp").over(item_history_window))
        .withColumn(
            "positive_interactions_lifetime",
            F.coalesce(
                F.sum(
                    F.when(F.col("verified_purchase") & (F.col("rating") >= 4), 1).otherwise(0)
                ).over(item_history_window),
                F.lit(0),
            ),
        )
        .withColumn(
            "historical_average_rating",
            F.avg(F.when(F.col("verified_purchase"), F.col("rating"))).over(item_history_window),
        )
        .withColumn(
            "historical_dissatisfaction_rate",
            F.avg(
                F.when(
                    F.col("verified_purchase"),
                    (F.col("rating") <= 2).cast("double"),
                )
            ).over(item_history_window),
        )
    )
    item_statistics = labels.select("interaction_id", "parent_asin", "label_timestamp").join(
        item_features.select(
            "interaction_id",
            "feature_timestamp",
            "positive_interactions_lifetime",
            "historical_average_rating",
            "historical_dissatisfaction_rate",
        ),
        "interaction_id",
        "inner",
    )
    item_statistics.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(f"{schemas['gold']}.gold_item_statistics_asof")
