from __future__ import annotations

import hashlib
import re
from typing import Any

BRONZE_IDENTITY_CONTRACT = "content-sha256-generation-v2"


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
    """Converge legacy Bronze rows onto the content-addressed replay identity."""
    if not spark.catalog.tableExists(table):
        return
    properties = {
        row["key"]: row["value"] for row in spark.sql(f"SHOW TBLPROPERTIES {table}").collect()
    }
    property_name = "marketplace.bronze_identity_contract"
    if properties.get(property_name) == BRONZE_IDENTITY_CONTRACT:
        return
    if "bronze_record_id" not in spark.table(table).columns:
        spark.sql(f"ALTER TABLE {table} ADD COLUMNS (bronze_record_id STRING)")
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    frame = spark.table(table)
    fingerprint = F.sha2("raw_payload", 256)
    expected_id = F.sha2(F.concat_ws("\u001f", "source_checksum", fingerprint), 256)
    has_legacy_identity = (
        frame.where(F.col("bronze_record_id").isNull() | (F.col("bronze_record_id") != expected_id))
        .limit(1)
        .count()
        > 0
    )
    has_replayed_generations = (
        frame.select("source_file", "source_checksum", "ingested_at")
        .dropDuplicates()
        .groupBy("source_file", "source_checksum")
        .count()
        .where(F.col("count") > 1)
        .limit(1)
        .count()
        > 0
    )
    if has_legacy_identity or has_replayed_generations:
        temporary_table = f"{table}__content_identity_migration"
        spark.sql(f"DROP TABLE IF EXISTS {temporary_table}")
        source_generation = Window.partitionBy("source_file", "source_checksum")
        migrated = (
            frame.withColumn("_latest_ingested_at", F.max("ingested_at").over(source_generation))
            .where(F.col("ingested_at") == F.col("_latest_ingested_at"))
            .drop("_latest_ingested_at")
            .withColumn("_record_fingerprint", fingerprint)
            .withColumn(
                "source_row_number",
                F.pmod(F.xxhash64("_record_fingerprint"), F.lit(2_147_483_647)).cast("int"),
            )
            .withColumn(
                "bronze_record_id",
                F.sha2(F.concat_ws("\u001f", "source_checksum", "_record_fingerprint"), 256),
            )
            .withColumn(
                "_identity_rank",
                F.row_number().over(
                    Window.partitionBy("bronze_record_id").orderBy(
                        F.col("ingested_at").asc_nulls_last(),
                        F.col("source_file"),
                    )
                ),
            )
            .where(F.col("_identity_rank") == 1)
            .drop("_record_fingerprint", "_identity_rank")
        )
        migrated.write.format("delta").mode("overwrite").saveAsTable(temporary_table)
        spark.sql(f"CREATE OR REPLACE TABLE {table} USING DELTA AS SELECT * FROM {temporary_table}")
        spark.sql(f"DROP TABLE {temporary_table}")
    spark.sql(
        f"ALTER TABLE {table} SET TBLPROPERTIES ('{property_name}' = '{BRONZE_IDENTITY_CONTRACT}')"
    )
