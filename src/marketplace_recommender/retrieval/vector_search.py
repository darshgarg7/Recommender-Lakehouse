from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote


@dataclass(frozen=True)
class VectorSearchBenchmarkConfig:
    endpoint_name: str = "marketplace-recommender-search"
    index_name: str = "workspace.scale_serving.als_item_mips_index"
    query_count: int = 200
    load_request_count: int = 500
    concurrency: int = 16
    k: int = 10
    ann_candidate_pool_size: int = 500
    ann_probe_scales: tuple[float, ...] = (1.0,)
    minimum_recall_at_10: float = 0.85
    maximum_p95_latency_ms: float = 1_000.0
    poll_timeout_seconds: int = 1_200

    def validate(self) -> None:
        if (
            min(
                self.query_count,
                self.load_request_count,
                self.concurrency,
                self.k,
                self.ann_candidate_pool_size,
            )
            <= 0
        ):
            raise ValueError("query, load, concurrency, and k values must be positive")
        if self.ann_candidate_pool_size < self.k:
            raise ValueError("ANN candidate pool must be at least as large as the final ranking")
        if not self.ann_probe_scales or any(scale <= 0 for scale in self.ann_probe_scales):
            raise ValueError("ANN probe scales must be nonempty and strictly positive")
        if not 0.0 <= self.minimum_recall_at_10 <= 1.0:
            raise ValueError("minimum_recall_at_10 must be in [0, 1]")
        if self.maximum_p95_latency_ms <= 0 or self.poll_timeout_seconds <= 0:
            raise ValueError("latency and timeout values must be positive")


def mips_item_extension(vector: Iterable[float], maximum_norm_squared: float) -> list[float]:
    """Map an item vector so L2 search preserves maximum-inner-product order."""
    values = [float(value) for value in vector]
    norm_squared = sum(value * value for value in values)
    if norm_squared > maximum_norm_squared + 1e-6:
        raise ValueError("maximum norm is smaller than the item vector norm")
    return values + [math.sqrt(max(0.0, maximum_norm_squared - norm_squared))]


def mips_query_extension(vector: Iterable[float], scale: float = 1.0) -> list[float]:
    """Extend a positively scaled query; exact MIPS order remains unchanged."""
    if scale <= 0:
        raise ValueError("MIPS query scale must be strictly positive")
    return [scale * float(value) for value in vector] + [0.0]


def parse_vector_search_rows(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize the Databricks query response into named rows."""
    manifest = response.get("manifest", {})
    columns = manifest.get("columns") or manifest.get("schema", {}).get("columns") or []
    names = [column.get("name") for column in columns]
    data = response.get("result", {}).get("data_array", [])
    if not names and data:
        raise ValueError("AI Search response contains data without a column manifest")
    return [dict(zip(names, row, strict=True)) for row in data]


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_load(latencies_ms: Iterable[float], wall_seconds: float) -> dict[str, float | int]:
    values = [float(value) for value in latencies_ms]
    if not values or wall_seconds <= 0:
        raise ValueError("load summary requires completed requests and positive wall time")
    return {
        "completed_requests": len(values),
        "throughput_qps": len(values) / wall_seconds,
        "latency_mean_ms": sum(values) / len(values),
        "latency_p50_ms": _percentile(values, 0.50),
        "latency_p95_ms": _percentile(values, 0.95),
        "latency_p99_ms": _percentile(values, 0.99),
    }


class _VectorSearchApi:
    def __init__(self, api_client: Any) -> None:
        self._api_client = api_client

    def _do(
        self,
        method: str,
        path: str | None = None,
        *,
        url: str | None = None,
        body: dict[str, Any] | None = None,
        query_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._api_client.do(
            method=method,
            path=path,
            url=url,
            body=body,
            query=query_params,
            headers={"Accept": "application/json"},
        )

    def get_endpoint(self, name: str) -> dict[str, Any]:
        return self._do("GET", f"/api/2.0/vector-search/endpoints/{quote(name, safe='')}")

    def create_endpoint(self, name: str) -> dict[str, Any]:
        return self._do(
            "POST",
            "/api/2.0/vector-search/endpoints",
            body={"name": name, "endpoint_type": "STANDARD"},
        )

    def get_index(self, name: str) -> dict[str, Any]:
        return self._do("GET", f"/api/2.0/vector-search/indexes/{quote(name, safe='')}")

    def create_index(
        self,
        *,
        name: str,
        endpoint_name: str,
        source_table: str,
        primary_key: str,
        vector_column: str,
        dimension: int,
    ) -> dict[str, Any]:
        return self._do(
            "POST",
            "/api/2.0/vector-search/indexes",
            body={
                "name": name,
                "endpoint_name": endpoint_name,
                "primary_key": primary_key,
                "index_type": "DELTA_SYNC",
                "delta_sync_index_spec": {
                    "source_table": source_table,
                    "pipeline_type": "TRIGGERED",
                    "embedding_vector_columns": [
                        {"name": vector_column, "embedding_dimension": dimension}
                    ],
                    "columns_to_sync": [
                        primary_key,
                        "item_index",
                        "benchmark_id",
                        vector_column,
                    ],
                },
            },
        )

    def sync_index(self, name: str) -> dict[str, Any]:
        return self._do(
            "POST", f"/api/2.0/vector-search/indexes/{quote(name, safe='')}/sync", body={}
        )

    def query(
        self, name: str, vector: list[float], k: int, *, index_url: str | None = None
    ) -> list[str]:
        query_url = None
        if index_url:
            query_url = index_url if index_url.startswith("https://") else f"https://{index_url}"
            query_url = f"{query_url.rstrip('/')}/query"
        response = self._do(
            "POST",
            None if query_url else f"/api/2.0/vector-search/indexes/{quote(name, safe='')}/query",
            url=query_url,
            body={
                "columns": ["parent_asin"],
                "query_vector": vector,
                "num_results": k,
                "query_type": "ANN",
            },
        )
        return [str(row["parent_asin"]) for row in parse_vector_search_rows(response)]


def _wait_until(
    read: Callable[[], dict[str, Any]],
    ready: Callable[[dict[str, Any]], bool],
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest = read()
        if ready(latest):
            return latest
        time.sleep(5)
    raise TimeoutError(f"managed AI Search resource did not become ready: {latest}")


def _index_is_current_and_idle(index: dict[str, Any], expected_delta_version: int) -> bool:
    status = index.get("status", {})
    update = status.get("triggered_update_status", {})
    processed_version = int(update.get("last_processed_commit_version") or -1)
    return bool(_index_is_idle(index) and processed_version >= expected_delta_version)


def _index_is_idle(index: dict[str, Any]) -> bool:
    status = index.get("status", {})
    return bool(status.get("ready") and status.get("detailed_state") == "ONLINE_NO_PENDING_UPDATE")


def _get_or_create(
    api: _VectorSearchApi,
    config: VectorSearchBenchmarkConfig,
    table: str,
    expected_delta_version: int,
) -> Any:
    try:
        endpoint = api.get_endpoint(config.endpoint_name)
    except Exception as exc:
        if "404" not in str(exc) and "NOT_FOUND" not in str(exc):
            raise
        endpoint = api.create_endpoint(config.endpoint_name)
    if endpoint.get("endpoint_status", {}).get("state") != "ONLINE":
        _wait_until(
            lambda: api.get_endpoint(config.endpoint_name),
            lambda value: value.get("endpoint_status", {}).get("state") == "ONLINE",
            config.poll_timeout_seconds,
        )
    try:
        api.get_index(config.index_name)
    except Exception as exc:
        if "404" not in str(exc) and "NOT_FOUND" not in str(exc):
            raise
        api.create_index(
            name=config.index_name,
            endpoint_name=config.endpoint_name,
            source_table=table,
            primary_key="parent_asin",
            vector_column="mips_vector",
            dimension=65,
        )
    index = _wait_until(
        lambda: api.get_index(config.index_name),
        _index_is_idle,
        config.poll_timeout_seconds,
    )
    if _index_is_current_and_idle(index, expected_delta_version):
        return index
    api.sync_index(config.index_name)
    # The predicate includes the expected source Delta version. It therefore
    # cannot accept the brief stale READY snapshot before the async sync starts.
    return _wait_until(
        lambda: api.get_index(config.index_name),
        lambda value: _index_is_current_and_idle(value, expected_delta_version),
        config.poll_timeout_seconds,
    )


def run_vector_search_benchmark(
    spark: Any,
    catalog: str,
    schema_prefix: str,
    job_run_id: str,
    job_id: str,
    config: VectorSearchBenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Provision, sync, recall-test, and load-test managed Databricks AI Search."""
    import numpy as np
    from databricks.sdk import WorkspaceClient
    from pyspark.sql import functions as F

    config = config or VectorSearchBenchmarkConfig()
    config.validate()
    prefix = f"{schema_prefix}_" if schema_prefix else ""
    features_schema = f"{catalog}.{prefix}features"
    serving_schema = f"{catalog}.{prefix}serving"
    monitoring_schema = f"{catalog}.{prefix}monitoring"
    item_table = f"{features_schema}.als_item_factors"
    user_table = f"{features_schema}.als_user_factors"
    search_table = f"{serving_schema}.als_item_factor_search_vectors"
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {serving_schema}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {monitoring_schema}")

    items = spark.table(item_table)
    norm = F.aggregate(
        "features",
        F.lit(0.0).cast("double"),
        lambda total, value: total + value.cast("double") * value.cast("double"),
    )
    with_norm = items.withColumn("_norm_squared", norm)
    maximum_norm_squared = float(with_norm.agg(F.max("_norm_squared")).first()[0])
    search_vectors = with_norm.select(
        "parent_asin",
        "item_index",
        "benchmark_id",
        F.concat(
            "features",
            F.array(
                F.sqrt(
                    F.greatest(F.lit(maximum_norm_squared) - F.col("_norm_squared"), F.lit(0.0))
                ).cast("float")
            ),
        ).alias("mips_vector"),
    )
    search_vectors.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(search_table)
    # Capture the data-bearing commit. A following metadata-only CDF property
    # commit is intentionally excluded because AI Search reports the last
    # processed data commit rather than that table-property version.
    source_delta_version = int(
        spark.sql(f"DESCRIBE HISTORY {search_table} LIMIT 1").select("version").first()[0]
    )
    spark.sql(f"ALTER TABLE {search_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

    api = _VectorSearchApi(WorkspaceClient().api_client)
    index = _get_or_create(api, config, search_table, source_delta_version)
    indexed_rows = int(index.get("status", {}).get("indexed_row_count") or 0)
    index_url = str(index.get("status", {}).get("index_url") or "")
    if not index_url:
        raise RuntimeError("managed AI Search index did not expose its optimized query URL")

    item_rows = items.select("parent_asin", "features").orderBy("parent_asin").collect()
    item_ids = np.asarray([row.parent_asin for row in item_rows], dtype=object)
    item_matrix = np.asarray([row.features for row in item_rows], dtype=np.float32)
    item_positions = {str(item_id): position for position, item_id in enumerate(item_ids)}
    query_rows = (
        spark.table(user_table)
        .orderBy(F.xxhash64("user_id"))
        .limit(config.query_count)
        .select("user_id", "features")
        .collect()
    )
    if len(query_rows) != config.query_count:
        raise RuntimeError("not enough learned user factors for the ANN benchmark")
    query_matrix = np.asarray([row.features for row in query_rows], dtype=np.float32)
    exact_scores = query_matrix @ item_matrix.T
    top_columns = np.argpartition(exact_scores, -config.k, axis=1)[:, -config.k :]
    top_scores = np.take_along_axis(exact_scores, top_columns, axis=1)
    top_ids = item_ids[top_columns]
    ordering = np.lexsort((top_ids, -top_scores), axis=1)
    exact_ids = np.take_along_axis(top_ids, ordering, axis=1)

    # Warm the endpoint separately so the load distribution describes steady-state behavior.
    for row in query_rows[: min(5, len(query_rows))]:
        api.query(
            config.index_name,
            mips_query_extension(row.features),
            config.ann_candidate_pool_size,
            index_url=index_url,
        )

    requests = [query_rows[index % len(query_rows)] for index in range(config.load_request_count)]
    latencies: list[float] = []
    responses: dict[int, list[str]] = {}
    raw_responses: dict[int, list[str]] = {}

    def execute(request_index: int) -> tuple[int, list[str], list[str], float]:
        started = time.perf_counter()
        candidates: list[str] = []
        seen_candidates: set[str] = set()
        raw_values: list[str] = []
        for probe_index, scale in enumerate(config.ann_probe_scales):
            probe_candidates = api.query(
                config.index_name,
                mips_query_extension(requests[request_index].features, scale),
                config.ann_candidate_pool_size,
                index_url=index_url,
            )
            if probe_index == 0:
                raw_values = probe_candidates[: config.k]
            for item_id in probe_candidates:
                if item_id not in seen_candidates:
                    seen_candidates.add(item_id)
                    candidates.append(item_id)
        known_candidates = [item_id for item_id in candidates if item_id in item_positions]
        positions = np.asarray(
            [item_positions[item_id] for item_id in known_candidates], dtype=np.int64
        )
        candidate_scores = item_matrix[positions] @ np.asarray(
            requests[request_index].features, dtype=np.float32
        )
        rerank_order = sorted(
            range(len(known_candidates)),
            key=lambda position: (-float(candidate_scores[position]), known_candidates[position]),
        )
        reranked = [known_candidates[position] for position in rerank_order[: config.k]]
        return (
            request_index,
            raw_values,
            reranked,
            (time.perf_counter() - started) * 1_000.0,
        )

    load_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        futures = [executor.submit(execute, index) for index in range(len(requests))]
        for future in as_completed(futures):
            request_index, raw_values, reranked_values, latency_ms = future.result()
            raw_responses[request_index] = raw_values
            responses[request_index] = reranked_values
            latencies.append(latency_ms)
    wall_seconds = time.perf_counter() - load_started
    load = summarize_load(latencies, wall_seconds)
    raw_recalls = [
        len(set(raw_responses[index]).intersection(exact_ids[index])) / config.k
        for index in range(config.query_count)
    ]
    reranked_recalls = [
        len(set(responses[index]).intersection(exact_ids[index])) / config.k
        for index in range(config.query_count)
    ]
    raw_recall_at_10 = sum(raw_recalls) / len(raw_recalls)
    recall_at_10 = sum(reranked_recalls) / len(reranked_recalls)
    implementation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    als_benchmark_id = str(items.select("benchmark_id").first()[0])
    benchmark_id = hashlib.sha256(
        json.dumps(
            {
                "implementation_sha256": implementation_sha256,
                "als_benchmark_id": als_benchmark_id,
                "config": asdict(config),
                "maximum_norm_squared": maximum_norm_squared,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    checks = {
        "managed_index_is_ready": bool(index.get("status", {}).get("ready")),
        "index_matches_source_delta_version": _index_is_current_and_idle(
            index, source_delta_version
        ),
        "all_item_factors_are_indexed": indexed_rows == len(item_rows),
        "mips_transform_has_expected_dimension": all(
            len(row.mips_vector) == 65 for row in spark.table(search_table).limit(20).collect()
        ),
        "ann_recall_meets_contract": recall_at_10 >= config.minimum_recall_at_10,
        "load_test_completed_without_errors": len(responses) == config.load_request_count,
        "p95_latency_meets_contract": float(load["latency_p95_ms"])
        <= config.maximum_p95_latency_ms,
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    summary = {
        "contract_version": "databricks-ai-search-benchmark/v3",
        "benchmark_id": benchmark_id,
        "implementation_sha256": implementation_sha256,
        "job_id": str(job_id),
        "job_run_id": str(job_run_id),
        "endpoint_name": config.endpoint_name,
        "index_name": config.index_name,
        "query_route": "optimized-index-url",
        "source_table": search_table,
        "source_delta_version": source_delta_version,
        "source_items": len(item_rows),
        "indexed_rows": indexed_rows,
        "vector_dimension": 65,
        "transform": "maximum-inner-product-to-l2",
        "query_count": config.query_count,
        "ann_candidate_pool_size": config.ann_candidate_pool_size,
        "ann_probe_scales": list(config.ann_probe_scales),
        "backend_query_count": config.load_request_count * len(config.ann_probe_scales),
        "raw_ann_recall_at_10": raw_recall_at_10,
        "rerank": "exact-inner-product-over-ann-candidates",
        "ann_recall_at_10": recall_at_10,
        "load": load,
        "concurrency": config.concurrency,
        "checks": checks,
        "failed_checks": failed_checks,
    }
    row = {
        "benchmark_id": benchmark_id,
        "contract_version": summary["contract_version"],
        "job_id": str(job_id),
        "job_run_id": str(job_run_id),
        "endpoint_name": config.endpoint_name,
        "index_name": config.index_name,
        "indexed_rows": indexed_rows,
        "ann_recall_at_10": recall_at_10,
        **load,
        "concurrency": config.concurrency,
        "passed": not failed_checks,
        "failed_checks_json": json.dumps(failed_checks, separators=(",", ":")),
        "summary_json": json.dumps(summary, sort_keys=True, separators=(",", ":")),
        "created_at_epoch_ms": int(time.time() * 1_000),
    }
    spark.createDataFrame([row]).write.format("delta").mode("append").saveAsTable(
        f"{monitoring_schema}.vector_search_benchmarks"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failed_checks:
        raise RuntimeError("AI Search benchmark certification failed: " + ", ".join(failed_checks))
    return summary
