from __future__ import annotations

import hashlib
import json
import re
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


def _hash_landed_object(path: str, chunk_bytes: int = 8 * 1024 * 1024) -> tuple[str, int]:
    """Hash a landed object with constant memory through the Volumes POSIX interface."""
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    landed_bytes = 0
    with open(path, "rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
            landed_bytes += len(chunk)
    return digest.hexdigest(), landed_bytes


def _ensure_schemas(spark: Any, catalog: str, schema_prefix: str = "") -> dict[str, str]:
    if not catalog.replace("_", "").isalnum():
        raise ValueError("unsafe catalog identifier")
    if schema_prefix and not schema_prefix.replace("_", "").isalnum():
        raise ValueError("unsafe schema prefix")
    layers = ("bronze", "silver", "gold", "features", "serving", "monitoring")
    schemas = {
        layer: f"{catalog}.{schema_prefix + '_' if schema_prefix else ''}{layer}"
        for layer in layers
    }
    for schema in schemas.values():
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {schemas["bronze"]}.bronze_quarantined_records (
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
    return schemas


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
    source_domain: str = "Magazine_Subscriptions",
    schema_prefix: str = "",
) -> None:
    """Verify landed Amazon Reviews objects before committing their manifest rows."""
    from pyspark.sql import functions as F

    if not source_domain.replace("_", "").isalnum():
        raise ValueError("source_domain may contain only letters, numbers, and underscores")
    schemas = _ensure_schemas(spark, catalog, schema_prefix)
    root = volume_root.rstrip("/")
    expected = [
        {
            "source_domain": source_domain,
            "source_kind": "reviews",
            "source_url": (
                "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
                f"resolve/main/raw/review_categories/{source_domain}.jsonl?download=true"
            ),
            "object_path": f"{root}/reviews/reviews_{source_domain}.jsonl",
            "filename": f"reviews_{source_domain}.jsonl",
            "checksum": reviews_checksum,
        },
        {
            "source_domain": source_domain,
            "source_kind": "metadata",
            "source_url": (
                "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
                f"resolve/main/raw/meta_categories/meta_{source_domain}.jsonl?download=true"
            ),
            "object_path": f"{root}/metadata/meta_{source_domain}.jsonl",
            "filename": f"meta_{source_domain}.jsonl",
            "checksum": metadata_checksum,
        },
    ]
    if any(not is_sha256(row["checksum"]) for row in expected):
        raise ValueError("SHA-256 checksums must contain 64 hexadecimal characters")

    # Spark's binaryFile source represents each object as one binary value and can
    # exhaust an executor on a large immutable landing object. Hash through the UC
    # Volumes POSIX interface in bounded chunks instead: same cryptographic gate,
    # constant memory, and no dependency on the serverless executor heap size.
    landed = spark.createDataFrame(
        [
            {
                "filename": row["filename"],
                "actual_checksum": result[0],
                "landed_bytes": result[1],
            }
            for row in expected
            for result in [_hash_landed_object(row["object_path"])]
        ]
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
        f"{schemas['bronze']}.bronze_ingestion_manifest",
    )


def ingest_bronze_stream(
    spark: Any,
    catalog: str,
    source_kind: str,
    source_path: str,
    schema_prefix: str = "",
) -> None:
    """Ingest exact JSONL lines with available-now Auto Loader and checkpoints."""
    from pyspark.sql import functions as F

    if source_kind not in {"reviews", "metadata"}:
        raise ValueError("source_kind must be reviews or metadata")
    schemas = _ensure_schemas(spark, catalog, schema_prefix)
    manifest_table = f"{schemas['bronze']}.bronze_ingestion_manifest"
    if not spark.catalog.tableExists(manifest_table):
        raise RuntimeError(
            f"{manifest_table} must contain checksum-validated committed objects before ingestion"
        )
    state_root = source_path.rstrip("/").rsplit("/", 1)[0]
    schema_location = f"{state_root}/_state/schemas/{source_kind}"
    checkpoint = f"{state_root}/_state/checkpoints/{source_kind}"
    # Bronze stores exact JSONL lines. Inferring and reconstructing a deeply nested
    # product schema here is expensive and makes ingestion sensitive to additive
    # source evolution; explicit parsing and quality policy belong in Silver.
    frame = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "text")
        .option("cloudFiles.schemaLocation", schema_location)
        .load(source_path)
    )
    source_file = F.col("_metadata.file_path")
    output = frame.select(
        source_file.alias("source_file"),
        F.regexp_extract(source_file, r"([^/]+)$", 1).alias("source_filename"),
        F.regexp_extract(source_file, r"(?:reviews_|meta_)([^/.]+)", 1).alias("source_domain"),
        F.col("value").alias("raw_payload"),
        F.lit(1).alias("schema_version"),
        F.lit(None).cast("string").alias("rescued_data"),
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
    table = f"{schemas['bronze']}.bronze_{'reviews' if source_kind == 'reviews' else 'product_metadata'}"
    quarantine_table = f"{schemas['bronze']}.bronze_quarantined_records"
    _migrate_bronze_record_id(spark, table)

    def write_verified(batch: Any, _batch_id: int) -> None:
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
        # Source line order is not semantic. Content addressing keeps identity
        # replay-stable without a single-file global sort, and deliberately
        # collapses byte-identical duplicate records from the same immutable object.
        stable_rows = (
            batch.withColumn("_record_fingerprint", F.sha2("raw_payload", 256))
            .withColumn(
                "source_row_number",
                F.pmod(F.xxhash64("_record_fingerprint"), F.lit(2_147_483_647)).cast("int"),
            )
            .withColumn(
                "bronze_record_id",
                F.sha2(F.concat_ws("\u001f", "source_checksum", "_record_fingerprint"), 256),
            )
            .drop("_record_fingerprint")
            .dropDuplicates(["bronze_record_id"])
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


def certify_pipeline_run(
    spark: Any,
    catalog: str,
    job_run_id: str,
    job_id: str,
    schema_prefix: str = "",
) -> None:
    """Fail closed and persist a content-addressed certificate for the materialized lakehouse."""
    schemas = _ensure_schemas(spark, catalog, schema_prefix)
    metrics = (
        spark.sql(f"""
        SELECT
          (SELECT count(*) FROM {schemas["bronze"]}.bronze_ingestion_manifest)
            AS manifest_rows,
          (SELECT min(replay_attempts) FROM {schemas["bronze"]}.bronze_ingestion_manifest)
            AS manifest_min_replay_attempts,
          (SELECT count(*) FROM {schemas["bronze"]}.bronze_reviews) AS bronze_reviews_rows,
          (SELECT count(*) - count(DISTINCT bronze_record_id)
             FROM {schemas["bronze"]}.bronze_reviews) AS bronze_reviews_duplicate_ids,
          (SELECT count(*) FROM {schemas["bronze"]}.bronze_product_metadata)
            AS bronze_metadata_rows,
          (SELECT count(*) - count(DISTINCT bronze_record_id)
             FROM {schemas["bronze"]}.bronze_product_metadata) AS bronze_metadata_duplicate_ids,
          (SELECT count(*) FROM {schemas["bronze"]}.bronze_quarantined_records)
            AS quarantined_rows,
          (SELECT count(*) FROM {schemas["silver"]}.silver_products) AS silver_products_rows,
          (SELECT count(*) FROM {schemas["silver"]}.silver_interactions)
            AS silver_interactions_rows,
          (SELECT count(*) - count(DISTINCT interaction_id)
             FROM {schemas["silver"]}.silver_interactions) AS silver_duplicate_ids,
          (SELECT count_if(quality_status <> 'valid')
             FROM {schemas["silver"]}.silver_interactions) AS silver_flagged_rows,
          (SELECT count(*) FROM {schemas["gold"]}.gold_training_labels) AS gold_labels_rows,
          (SELECT count(*) FROM {schemas["gold"]}.gold_user_sequences_asof)
            AS gold_sequences_rows,
          (SELECT count(*) FROM {schemas["gold"]}.gold_item_statistics_asof)
            AS gold_item_statistics_rows,
          (SELECT count_if(feature_timestamp >= label_timestamp)
             FROM {schemas["gold"]}.gold_item_statistics_asof) AS item_feature_leakage_rows,
          (SELECT count_if(exists(historical_events, event -> event.ts >= label_timestamp))
             FROM {schemas["gold"]}.gold_user_sequences_asof) AS sequence_leakage_rows
    """)
        .first()
        .asDict()
    )
    checks = {
        "manifest_has_two_sources": metrics["manifest_rows"] == 2,
        "bronze_reviews_are_nonempty": metrics["bronze_reviews_rows"] > 0,
        "canonical_products_are_bounded_by_metadata": (
            0 < metrics["silver_products_rows"] <= metrics["bronze_metadata_rows"]
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
        for row in spark.table(f"{schemas['bronze']}.bronze_ingestion_manifest")
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
        f"{schemas['monitoring']}.pipeline_run_certifications",
        "job_run_id",
    )
    if failed_checks:
        raise RuntimeError("lakehouse certification failed: " + ", ".join(failed_checks))
