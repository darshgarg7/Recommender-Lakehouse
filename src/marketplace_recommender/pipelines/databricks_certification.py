from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from marketplace_recommender.pipelines.databricks_common import _ensure_schemas, _merge_table


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
          (SELECT count(*) - count(DISTINCT concat_ws(
              char(31), source_checksum, sha2(raw_payload, 256)))
             FROM {schemas["bronze"]}.bronze_reviews) AS bronze_reviews_duplicate_content_rows,
          (SELECT count(*) FROM {schemas["bronze"]}.bronze_product_metadata)
            AS bronze_metadata_rows,
          (SELECT count(*) - count(DISTINCT bronze_record_id)
             FROM {schemas["bronze"]}.bronze_product_metadata) AS bronze_metadata_duplicate_ids,
          (SELECT count(*) - count(DISTINCT concat_ws(
              char(31), source_checksum, sha2(raw_payload, 256)))
             FROM {schemas["bronze"]}.bronze_product_metadata)
            AS bronze_metadata_duplicate_content_rows,
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
        "bronze_review_content_is_unique": metrics["bronze_reviews_duplicate_content_rows"] == 0,
        "bronze_metadata_content_is_unique": (
            metrics["bronze_metadata_duplicate_content_rows"] == 0
        ),
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
        "contract_version": "lakehouse-certification/v2",
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
                "contract_version": "lakehouse-certification/v2",
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
