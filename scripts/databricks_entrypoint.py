from __future__ import annotations

import argparse

from marketplace_recommender.pipelines.databricks import (
    bootstrap_ingestion_manifest,
    build_gold_tables,
    build_silver_tables,
    certify_pipeline_run,
    ingest_bronze_stream,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("bootstrap", "bronze", "silver", "gold", "certify"), required=True
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema-prefix", default="")
    parser.add_argument("--source-domain", default="Magazine_Subscriptions")
    parser.add_argument("--volume-root")
    parser.add_argument("--reviews-checksum")
    parser.add_argument("--metadata-checksum")
    parser.add_argument("--source-kind", choices=("reviews", "metadata"))
    parser.add_argument("--source-path")
    parser.add_argument("--job-run-id")
    parser.add_argument("--job-id")
    args = parser.parse_args()
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    if args.stage == "bootstrap":
        if not args.volume_root or not args.reviews_checksum or not args.metadata_checksum:
            parser.error(
                "bootstrap requires --volume-root, --reviews-checksum, and --metadata-checksum"
            )
        bootstrap_ingestion_manifest(
            spark,
            args.catalog,
            args.volume_root,
            args.reviews_checksum,
            args.metadata_checksum,
            args.source_domain,
            args.schema_prefix,
        )
    elif args.stage == "bronze":
        if not args.source_kind or not args.source_path:
            parser.error("bronze requires --source-kind and --source-path")
        ingest_bronze_stream(
            spark, args.catalog, args.source_kind, args.source_path, args.schema_prefix
        )
    elif args.stage == "silver":
        build_silver_tables(spark, args.catalog, args.schema_prefix)
    elif args.stage == "gold":
        build_gold_tables(spark, args.catalog, args.schema_prefix)
    else:
        if not args.job_run_id or not args.job_id:
            parser.error("certify requires --job-run-id and --job-id")
        certify_pipeline_run(spark, args.catalog, args.job_run_id, args.job_id, args.schema_prefix)


if __name__ == "__main__":
    main()
