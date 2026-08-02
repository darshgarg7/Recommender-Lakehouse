from __future__ import annotations

import argparse

from marketplace_recommender.pipelines.databricks import (
    bootstrap_ingestion_manifest,
    build_gold_tables,
    build_silver_tables,
    certify_pipeline_run,
    ingest_bronze_stream,
)
from marketplace_recommender.retrieval.spark_als import (
    SparkAlsBenchmarkConfig,
    train_spark_als_benchmark,
)
from marketplace_recommender.retrieval.sasrec_torch import SasRecConfig, train_sasrec_benchmark
from marketplace_recommender.retrieval.vector_search import (
    VectorSearchBenchmarkConfig,
    run_vector_search_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "bootstrap",
            "bronze",
            "silver",
            "gold",
            "certify",
            "train-als",
            "train-sasrec",
            "benchmark-vector-search",
        ),
        required=True,
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
    parser.add_argument("--model-artifact-path")
    parser.add_argument("--als-rank", type=int, default=64)
    parser.add_argument("--als-max-iter", type=int, default=12)
    parser.add_argument("--evaluation-user-limit", type=int, default=10_000)
    parser.add_argument("--vector-search-endpoint", default="marketplace-recommender-search")
    parser.add_argument(
        "--vector-search-index", default="workspace.scale_serving.als_item_mips_index"
    )
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
    elif args.stage == "certify":
        if not args.job_run_id or not args.job_id:
            parser.error("certify requires --job-run-id and --job-id")
        certify_pipeline_run(spark, args.catalog, args.job_run_id, args.job_id, args.schema_prefix)
    elif args.stage == "train-als":
        if not args.model_artifact_path or not args.job_run_id or not args.job_id:
            parser.error("train-als requires --model-artifact-path, --job-run-id, and --job-id")
        train_spark_als_benchmark(
            spark,
            args.catalog,
            args.schema_prefix,
            args.model_artifact_path,
            args.job_run_id,
            args.job_id,
            SparkAlsBenchmarkConfig(
                rank=args.als_rank,
                max_iter=args.als_max_iter,
                evaluation_user_limit=args.evaluation_user_limit,
            ),
        )
    elif args.stage == "train-sasrec":
        if not args.model_artifact_path or not args.job_run_id or not args.job_id:
            parser.error("train-sasrec requires --model-artifact-path, --job-run-id, and --job-id")
        train_sasrec_benchmark(
            spark,
            args.catalog,
            args.schema_prefix,
            args.model_artifact_path,
            args.job_run_id,
            args.job_id,
            SasRecConfig(evaluation_user_limit=min(args.evaluation_user_limit, 4_000)),
        )
    else:
        if not args.job_run_id or not args.job_id:
            parser.error("benchmark-vector-search requires --job-run-id and --job-id")
        run_vector_search_benchmark(
            spark,
            args.catalog,
            args.schema_prefix,
            args.job_run_id,
            args.job_id,
            VectorSearchBenchmarkConfig(
                endpoint_name=args.vector_search_endpoint,
                index_name=args.vector_search_index,
            ),
        )


if __name__ == "__main__":
    main()
