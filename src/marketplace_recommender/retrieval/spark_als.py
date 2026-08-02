"""Stable public API for the distributed temporal ALS benchmark."""

from marketplace_recommender.retrieval.spark_als_benchmark import train_spark_als_benchmark
from marketplace_recommender.retrieval.spark_als_config import (
    SparkAlsBenchmarkConfig,
    benchmark_fingerprint,
    ndcg_for_single_relevant_rank,
)

__all__ = [
    "SparkAlsBenchmarkConfig",
    "benchmark_fingerprint",
    "ndcg_for_single_relevant_rank",
    "train_spark_als_benchmark",
]
