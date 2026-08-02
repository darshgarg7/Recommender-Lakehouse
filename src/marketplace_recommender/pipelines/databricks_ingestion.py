from __future__ import annotations

from typing import Any

from marketplace_recommender.pipelines.databricks_common import (
    _ensure_schemas,
    _hash_landed_object,
    _merge_manifest,
    _merge_table,
    _migrate_bronze_record_id,
    is_sha256,
)

BRONZE_STATE_CONTRACT = "raw_text_v1"


def _bronze_state_paths(source_path: str, source_kind: str) -> tuple[str, str]:
    """Version Auto Loader state whenever its source-format contract changes."""
    state_root = source_path.rstrip("/").rsplit("/", 1)[0]
    state_base = f"{state_root}/_state/{BRONZE_STATE_CONTRACT}"
    return (
        f"{state_base}/schemas/{source_kind}",
        f"{state_base}/checkpoints/{source_kind}",
    )


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
    schema_location, checkpoint = _bronze_state_paths(source_path, source_kind)
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
