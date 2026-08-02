"""Stable Databricks pipeline API assembled from stage-owned modules."""

from marketplace_recommender.pipelines.databricks_certification import certify_pipeline_run
from marketplace_recommender.pipelines.databricks_common import (
    BRONZE_IDENTITY_CONTRACT,
    _ensure_schemas,
    _hash_landed_object,
    is_sha256,
)
from marketplace_recommender.pipelines.databricks_ingestion import (
    BRONZE_STATE_CONTRACT,
    _bronze_state_paths,
    bootstrap_ingestion_manifest,
    ingest_bronze_stream,
)
from marketplace_recommender.pipelines.databricks_transform import (
    build_gold_tables,
    build_silver_tables,
)

__all__ = [
    "BRONZE_IDENTITY_CONTRACT",
    "BRONZE_STATE_CONTRACT",
    "_bronze_state_paths",
    "_ensure_schemas",
    "_hash_landed_object",
    "bootstrap_ingestion_manifest",
    "build_gold_tables",
    "build_silver_tables",
    "certify_pipeline_run",
    "ingest_bronze_stream",
    "is_sha256",
]
