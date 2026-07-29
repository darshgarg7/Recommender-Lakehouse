from __future__ import annotations

import re
import hashlib
import json
from datetime import datetime, timezone
from typing import Any


def _merge_table(spark: Any, frame: Any, table: str, key: str) -> None:
    """Idempotent Delta upsert used by bounded backfills and incremental refreshes."""
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


def _merge_manifest(spark: Any, frame: Any, table: str) -> None:
    """Upsert verified objects while preserving an auditable replay count."""
    if not spark.catalog.tableExists(table):
        frame.write.format("delta").mode("append").saveAsTable(table)
        return
    from delta.tables import DeltaTable

    assignments = {name: f"source.`{name}`" for name in frame.columns if name != "replay_attempts"}
    assignments["replay_attempts"] = "coalesce(target.replay_attempts, 0) + 1"
    (
        DeltaTable.forName(spark, table)
        .alias("target")
        .merge(frame.alias("source"), "target.object_path = source.object_path")
        .whenMatchedUpdate(set=assignments)
        .whenNotMatchedInsertAll()
        .execute()
    )


def is_sha256(value: str) -> bool:
    """Return whether a value is a complete hexadecimal SHA-256 digest."""
    return re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _ensure_schemas(spark: Any, catalog: str) -> None:
    if not catalog.replace("_", "").isalnum():
        raise ValueError("unsafe catalog identifier")
    for schema in ("bronze", "silver", "gold", "features", "serving", "monitoring"):
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {catalog}.bronze.bronze_quarantined_records (
            bronze_record_id STRING,
            source_file STRING,
            source_domain STRING,
            source_checksum STRING,
            source_row_number INT,
            raw_payload STRING,
            schema_version INT,
            rescued_data STRING,
            ingested_at TIMESTAMP,
            quarantine_reason STRING
        ) USING DELTA
    """)


def _migrate_bronze_record_id(spark: Any, table: str) -> None:
    """Backfill deterministic record IDs for tables created by an earlier bundle version."""
    if not spark.catalog.tableExists(table):
        return
    if "bronze_record_id" not in spark.table(table).columns:
        spark.sql(f"ALTER TABLE {table} ADD COLUMNS (bronze_record_id STRING)")
    spark.sql(f"""
        UPDATE {table}
        SET bronze_record_id = sha2(
            concat_ws(
                char(31),
                source_checksum,
                CAST(source_row_number AS STRING),
                raw_payload
            ),
            256
        )
        WHERE bronze_record_id IS NULL
    """)


def bootstrap_ingestion_manifest(
    spark: Any,
    catalog: str,
    volume_root: str,
    reviews_checksum: str,
    metadata_checksum: str,
) -> None:
    """Verify landed Amazon Reviews objects before committing their manifest rows."""
    from pyspark.sql import functions as F

    _ensure_schemas(spark, catalog)
    root = volume_root.rstrip("/")
    expected = [
        {
            "source_domain": "Magazine_Subscriptions",
            "source_kind": "reviews",
            "source_url": (
                "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
                "resolve/main/raw/review_categories/Magazine_Subscriptions.jsonl?download=true"
            ),
            "object_path": f"{root}/reviews/reviews_Magazine_Subscriptions.jsonl",
            "filename": "reviews_Magazine_Subscriptions.jsonl",
            "checksum": reviews_checksum,
        },
        {
            "source_domain": "Magazine_Subscriptions",
            "source_kind": "metadata",
            "source_url": (
                "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
                "resolve/main/raw/meta_categories/meta_Magazine_Subscriptions.jsonl?download=true"
            ),
            "object_path": f"{root}/metadata/meta_Magazine_Subscriptions.jsonl",
            "filename": "meta_Magazine_Subscriptions.jsonl",
            "checksum": metadata_checksum,
        },
    ]
    if any(not is_sha256(row["checksum"]) for row in expected):
        raise ValueError("SHA-256 checksums must contain 64 hexadecimal characters")

    landed = (
        spark.read.format("binaryFile")
        .load([row["object_path"] for row in expected])
        .select(
            F.regexp_extract("path", r"([^/]+)$", 1).alias("filename"),
            F.sha2("content", 256).alias("actual_checksum"),
            F.col("length").alias("landed_bytes"),
        )
    )
    verified = spark.createDataFrame(expected).join(landed, "filename", "left")
    failures = (
        verified.where(
            F.col("actual_checksum").isNull() | (F.lower("actual_checksum") != F.lower("checksum"))
        )
        .select("object_path", "checksum", "actual_checksum")
        .collect()
    )
    if failures:
        details = "; ".join(
            f"{row.object_path}: expected={row.checksum}, actual={row.actual_checksum}"
            for row in failures
        )
        raise RuntimeError(f"landed object checksum validation failed: {details}")

    manifest = (
        verified.drop("actual_checksum")
        .withColumn("download_status", F.lit("complete"))
        .withColumn("ingestion_status", F.lit("committed"))
        .withColumn("retry_count", F.lit(0))
        .withColumn("replay_attempts", F.lit(0))
        .withColumn("validated_at", F.current_timestamp())
    )
    _merge_manifest(
        spark,
        manifest,
        f"{catalog}.bronze.bronze_ingestion_manifest",
    )


def ingest_bronze_stream(spark: Any, catalog: str, source_kind: str, source_path: str) -> None:
    """Available-now Auto Loader backfill with checkpointing and rescued columns."""
    from pyspark.sql import functions as F

    if source_kind not in {"reviews", "metadata"}:
        raise ValueError("source_kind must be reviews or metadata")
    _ensure_schemas(spark, catalog)
    manifest_table = f"{catalog}.bronze.bronze_ingestion_manifest"
    if not spark.catalog.tableExists(manifest_table):
        raise RuntimeError(
            f"{manifest_table} must contain checksum-validated committed objects before ingestion"
        )
    state_root = source_path.rstrip("/").rsplit("/", 1)[0]
    schema_location = f"{state_root}/_state/schemas/{source_kind}"
    checkpoint = f"{state_root}/_state/checkpoints/{source_kind}"
    frame = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("cloudFiles.schemaEvolutionMode", "rescue")
        .option("cloudFiles.schemaLocation", schema_location)
        .option("rescuedDataColumn", "rescued_data")
        .load(source_path)
    )
    payload_columns = [name for name in frame.columns if name != "rescued_data"]
    source_file = F.col("_metadata.file_path")
    output = frame.select(
        source_file.alias("source_file"),
        F.regexp_extract(source_file, r"([^/]+)$", 1).alias("source_filename"),
        F.regexp_extract(source_file, r"(?:reviews_|meta_)([^/.]+)", 1).alias("source_domain"),
        F.to_json(F.struct(*(F.col(name) for name in payload_columns))).alias("raw_payload"),
        F.lit(1).alias("schema_version"),
        F.col("rescued_data"),
        F.current_timestamp().alias("ingested_at"),
    )
    committed = (
        spark.table(manifest_table)
        .where(F.col("ingestion_status").isin("pending", "committed"))
        .select(
            F.regexp_extract("object_path", r"([^/]+)$", 1).alias("manifest_filename"),
            "checksum",
        )
        .dropDuplicates(["manifest_filename", "checksum"])
    )
    output = (
        output.join(
            F.broadcast(committed),
            output.source_filename == committed.manifest_filename,
            "left",
        )
        .drop("manifest_filename", "source_filename")
        .withColumnRenamed("checksum", "source_checksum")
    )
    table = (
        f"{catalog}.bronze.bronze_{'reviews' if source_kind == 'reviews' else 'product_metadata'}"
    )
    quarantine_table = f"{catalog}.bronze.bronze_quarantined_records"
    _migrate_bronze_record_id(spark, table)

    def write_verified(batch: Any, _batch_id: int) -> None:
        from pyspark.sql.window import Window

        missing = (
            batch.where(F.col("source_checksum").isNull())
            .select("source_file")
            .distinct()
            .limit(5)
            .collect()
        )
        if missing:
            raise RuntimeError(
                "Auto Loader discovered objects absent from the validated manifest: "
                + ", ".join(row.source_file for row in missing)
            )
        stable_rows = batch.withColumn(
            "source_row_number",
            F.row_number().over(
                Window.partitionBy("source_file").orderBy(F.sha2("raw_payload", 256), "raw_payload")
            ),
        ).withColumn(
            "bronze_record_id",
            F.sha2(
                F.concat_ws(
                    "\u001f",
                    "source_checksum",
                    F.col("source_row_number").cast("string"),
                    "raw_payload",
                ),
                256,
            ),
        )
        quarantined = stable_rows.where(F.col("rescued_data").isNotNull()).withColumn(
            "quarantine_reason", F.lit("schema_rescue_or_malformed_json")
        )
        _merge_table(spark, quarantined, quarantine_table, "bronze_record_id")
        _merge_table(spark, stable_rows, table, "bronze_record_id")

    (
        output.writeStream.foreachBatch(write_verified)
        .option("checkpointLocation", checkpoint)
        .trigger(availableNow=True)
        .start()
        .awaitTermination()
    )


def build_silver_tables(spark: Any, catalog: str) -> None:
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

    _ensure_schemas(spark, catalog)
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
    raw_products = spark.table(f"{catalog}.bronze.bronze_product_metadata")
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
    _merge_table(spark, products, f"{catalog}.silver.silver_products", "parent_asin")
    variants = (
        raw_products.withColumn("record", F.from_json("raw_payload", product_schema))
        .select("record.asin", F.coalesce("record.parent_asin", "record.asin").alias("parent_asin"))
        .where(F.col("asin").isNotNull())
        .dropDuplicates(["asin"])
    )
    _merge_table(spark, variants, f"{catalog}.silver.silver_product_variants", "asin")
    reviews = spark.table(f"{catalog}.bronze.bronze_reviews").withColumn(
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
    _merge_table(spark, interactions, f"{catalog}.silver.silver_interactions", "interaction_id")


def build_gold_tables(spark: Any, catalog: str) -> None:
    """Materialize strictly historical labels and sequence examples in Delta."""
    _ensure_schemas(spark, catalog)
    spark.sql(f"""
        CREATE OR REPLACE TABLE {catalog}.gold.gold_training_labels AS
        SELECT interaction_id, user_id, parent_asin, review_timestamp AS label_timestamp,
               CASE WHEN rating >= 4 THEN 1 ELSE 0 END AS label, rating
        FROM {catalog}.silver.silver_interactions
        WHERE verified_purchase = true AND rating <> 3
    """)
    spark.sql(f"""
        CREATE OR REPLACE TABLE {catalog}.gold.gold_user_sequences_asof AS
        SELECT l.interaction_id, l.user_id, l.label_timestamp,
               slice(
                 array_sort(collect_list(named_struct('ts', h.review_timestamp, 'item', h.parent_asin))),
                 -100, 100
               ) AS historical_events
        FROM {catalog}.gold.gold_training_labels l
        LEFT JOIN {catalog}.silver.silver_interactions h
          ON l.user_id = h.user_id AND h.review_timestamp < l.label_timestamp
        GROUP BY l.interaction_id, l.user_id, l.label_timestamp
    """)
    spark.sql(f"""
        CREATE OR REPLACE TABLE {catalog}.gold.gold_item_statistics_asof AS
        SELECT l.interaction_id, l.parent_asin, l.label_timestamp,
               max(h.review_timestamp) AS feature_timestamp,
               count_if(h.verified_purchase AND h.rating >= 4) AS positive_interactions_lifetime,
               avg(CASE WHEN h.verified_purchase THEN h.rating END) AS historical_average_rating,
               avg(CASE WHEN h.verified_purchase THEN CAST(h.rating <= 2 AS DOUBLE) END)
                 AS historical_dissatisfaction_rate
        FROM {catalog}.gold.gold_training_labels l
        LEFT JOIN {catalog}.silver.silver_interactions h
          ON l.parent_asin = h.parent_asin AND h.review_timestamp < l.label_timestamp
        GROUP BY l.interaction_id, l.parent_asin, l.label_timestamp
    """)


def certify_pipeline_run(spark: Any, catalog: str, job_run_id: str, job_id: str) -> None:
    """Fail closed and persist a content-addressed certificate for the materialized lakehouse."""
    _ensure_schemas(spark, catalog)
    metrics = (
        spark.sql(f"""
        SELECT
          (SELECT count(*) FROM {catalog}.bronze.bronze_ingestion_manifest)
            AS manifest_rows,
          (SELECT min(replay_attempts) FROM {catalog}.bronze.bronze_ingestion_manifest)
            AS manifest_min_replay_attempts,
          (SELECT count(*) FROM {catalog}.bronze.bronze_reviews) AS bronze_reviews_rows,
          (SELECT count(*) - count(DISTINCT bronze_record_id)
             FROM {catalog}.bronze.bronze_reviews) AS bronze_reviews_duplicate_ids,
          (SELECT count(*) FROM {catalog}.bronze.bronze_product_metadata)
            AS bronze_metadata_rows,
          (SELECT count(*) - count(DISTINCT bronze_record_id)
             FROM {catalog}.bronze.bronze_product_metadata) AS bronze_metadata_duplicate_ids,
          (SELECT count(*) FROM {catalog}.bronze.bronze_quarantined_records)
            AS quarantined_rows,
          (SELECT count(*) FROM {catalog}.silver.silver_products) AS silver_products_rows,
          (SELECT count(*) FROM {catalog}.silver.silver_interactions)
            AS silver_interactions_rows,
          (SELECT count(*) - count(DISTINCT interaction_id)
             FROM {catalog}.silver.silver_interactions) AS silver_duplicate_ids,
          (SELECT count_if(quality_status <> 'valid')
             FROM {catalog}.silver.silver_interactions) AS silver_flagged_rows,
          (SELECT count(*) FROM {catalog}.gold.gold_training_labels) AS gold_labels_rows,
          (SELECT count(*) FROM {catalog}.gold.gold_user_sequences_asof)
            AS gold_sequences_rows,
          (SELECT count(*) FROM {catalog}.gold.gold_item_statistics_asof)
            AS gold_item_statistics_rows,
          (SELECT count_if(feature_timestamp >= label_timestamp)
             FROM {catalog}.gold.gold_item_statistics_asof) AS item_feature_leakage_rows,
          (SELECT count_if(exists(historical_events, event -> event.ts >= label_timestamp))
             FROM {catalog}.gold.gold_user_sequences_asof) AS sequence_leakage_rows
    """)
        .first()
        .asDict()
    )
    checks = {
        "manifest_has_two_sources": metrics["manifest_rows"] == 2,
        "bronze_reviews_are_nonempty": metrics["bronze_reviews_rows"] > 0,
        "metadata_matches_canonical_products": (
            metrics["bronze_metadata_rows"] == metrics["silver_products_rows"]
        ),
        "silver_is_bounded_by_bronze": (
            0 < metrics["silver_interactions_rows"] <= metrics["bronze_reviews_rows"]
        ),
        "gold_tables_are_aligned": (
            metrics["gold_labels_rows"]
            == metrics["gold_sequences_rows"]
            == metrics["gold_item_statistics_rows"]
            > 0
        ),
        "bronze_review_ids_are_unique": metrics["bronze_reviews_duplicate_ids"] == 0,
        "bronze_metadata_ids_are_unique": metrics["bronze_metadata_duplicate_ids"] == 0,
        "silver_ids_are_unique": metrics["silver_duplicate_ids"] == 0,
        "silver_quality_is_clean": metrics["silver_flagged_rows"] == 0,
        "item_features_are_strictly_historical": metrics["item_feature_leakage_rows"] == 0,
        "user_histories_are_strictly_historical": metrics["sequence_leakage_rows"] == 0,
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    source_checksums = [
        row.checksum
        for row in spark.table(f"{catalog}.bronze.bronze_ingestion_manifest")
        .select("checksum")
        .distinct()
        .orderBy("checksum")
        .collect()
    ]
    source_set_sha256 = hashlib.sha256("\u001f".join(source_checksums).encode()).hexdigest()
    state = {
        "contract_version": "lakehouse-certification/v1",
        "source_set_sha256": source_set_sha256,
        "metrics": metrics,
        "checks": checks,
    }
    state_sha256 = hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    certificate = spark.createDataFrame(
        [
            {
                "job_run_id": str(job_run_id),
                "job_id": str(job_id),
                "contract_version": "lakehouse-certification/v1",
                "certified_at": datetime.now(timezone.utc),
                "passed": not failed_checks,
                "assertion_count": len(checks),
                "failed_checks_json": json.dumps(failed_checks),
                "source_set_sha256": source_set_sha256,
                "table_state_sha256": state_sha256,
                "metrics_json": json.dumps(metrics, sort_keys=True),
            }
        ]
    )
    _merge_table(
        spark,
        certificate,
        f"{catalog}.monitoring.pipeline_run_certifications",
        "job_run_id",
    )
    if failed_checks:
        raise RuntimeError("lakehouse certification failed: " + ", ".join(failed_checks))
