import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

from marketplace_recommender.governance.code_identity import source_tree_sha256
from marketplace_recommender.pipelines.databricks import (
    _ensure_schemas,
    _hash_landed_object,
    certify_pipeline_run,
    is_sha256,
)
from marketplace_recommender.retrieval.sasrec_benchmark import SASREC_IMPLEMENTATION_SOURCES
from marketplace_recommender.retrieval.sasrec_torch import SasRecConfig, train_sasrec_benchmark
from marketplace_recommender.retrieval.spark_als import (
    SparkAlsBenchmarkConfig,
    train_spark_als_benchmark,
)
from marketplace_recommender.retrieval.spark_als_benchmark import ALS_IMPLEMENTATION_SOURCES
from marketplace_recommender.retrieval.vector_search import (
    VectorSearchBenchmarkConfig,
    run_vector_search_benchmark,
)


ROOT = Path(__file__).resolve().parents[2]


def task_parameters(task: dict[str, Any]) -> dict[str, str]:
    values = task["spark_python_task"]["parameters"]
    if len(values) % 2:
        raise AssertionError("task parameters must be flag/value pairs")
    return dict(zip(values[::2], values[1::2]))


def function_calls(path: Path, function_name: str) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    return [node for node in ast.walk(function) if isinstance(node, ast.Call)]


def call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


class DatabricksBundleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = yaml.safe_load((ROOT / "databricks.yml").read_text(encoding="utf-8"))
        cls.jobs = cls.bundle["resources"]["jobs"]
        cls.evidence = json.loads(
            (ROOT / "benchmarks/reports/portfolio_evidence.json").read_text(encoding="utf-8")
        )

    def test_bundle_is_serverless_and_checksum_gated(self):
        self.assertNotIn("existing_cluster_id", json.dumps(self.bundle))
        pipeline = self.jobs["marketplace_pipeline"]
        tasks = {task["task_key"]: task for task in pipeline["tasks"]}
        self.assertEqual(
            set(tasks),
            {"bootstrap", "bronze_reviews", "bronze_metadata", "silver", "gold", "certify"},
        )
        self.assertTrue(all(task["environment_key"] == "default" for task in tasks.values()))
        bootstrap = task_parameters(tasks["bootstrap"])
        self.assertEqual(bootstrap["--reviews-checksum"], "${var.reviews_checksum}")
        self.assertEqual(bootstrap["--metadata-checksum"], "${var.metadata_checksum}")
        self.assertEqual(bootstrap["--source-domain"], "${var.source_domain}")

    def test_checksum_contract_requires_hexadecimal_sha256(self):
        self.assertTrue(is_sha256("a" * 64))
        self.assertTrue(is_sha256("AB" * 32))
        self.assertFalse(is_sha256("g" * 64))
        self.assertFalse(is_sha256("a" * 63))

    def test_landed_object_hashing_is_chunked_and_byte_exact(self):
        payload = (b"constant-memory-checksum" * 17) + b"!"
        with tempfile.NamedTemporaryFile() as landed:
            landed.write(payload)
            landed.flush()
            digest, landed_bytes = _hash_landed_object(landed.name, chunk_bytes=7)
        self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
        self.assertEqual(landed_bytes, len(payload))
        with self.assertRaises(ValueError):
            _hash_landed_object("unused", chunk_bytes=0)

    def test_pipeline_dag_ends_in_run_bound_certification(self):
        tasks = {task["task_key"]: task for task in self.jobs["marketplace_pipeline"]["tasks"]}
        self.assertEqual(tasks["certify"]["depends_on"], [{"task_key": "gold"}])
        parameters = task_parameters(tasks["certify"])
        self.assertEqual(parameters["--stage"], "certify")
        self.assertEqual(parameters["--job-run-id"], "{{job.run_id}}")
        self.assertEqual(parameters["--job-id"], "{{job.id}}")
        self.assertTrue(callable(certify_pipeline_run))

    def test_model_benchmarks_are_independent_run_bound_jobs(self):
        expected = {
            "marketplace_als_benchmark": (
                "train_validate_certify_als",
                "train-als",
                train_spark_als_benchmark,
            ),
            "marketplace_sasrec_benchmark": (
                "train_validate_certify_sasrec",
                "train-sasrec",
                train_sasrec_benchmark,
            ),
            "marketplace_vector_search_benchmark": (
                "sync_recall_and_load_test",
                "benchmark-vector-search",
                run_vector_search_benchmark,
            ),
        }
        for job_name, (task_key, stage, entrypoint) in expected.items():
            job = self.jobs[job_name]
            self.assertEqual(len(job["tasks"]), 1)
            task = job["tasks"][0]
            self.assertEqual(task["task_key"], task_key)
            parameters = task_parameters(task)
            self.assertEqual(parameters["--stage"], stage)
            self.assertEqual(parameters["--job-run-id"], "{{job.run_id}}")
            self.assertEqual(parameters["--job-id"], "{{job.id}}")
            self.assertTrue(callable(entrypoint))
        SparkAlsBenchmarkConfig().validate()
        SasRecConfig().validate()
        VectorSearchBenchmarkConfig().validate()

    def test_model_time_boundaries_use_exact_percentiles(self):
        implementations = (
            (
                ROOT / "src/marketplace_recommender/retrieval/spark_als_benchmark.py",
                "train_spark_als_benchmark",
            ),
            (
                ROOT / "src/marketplace_recommender/retrieval/sasrec_benchmark.py",
                "train_sasrec_benchmark",
            ),
        )
        for path, function_name in implementations:
            names = {call_name(call) for call in function_calls(path, function_name)}
            self.assertIn("exact_temporal_cutoffs", names)
            self.assertNotIn("approxQuantile", names)

    def test_local_secrets_are_ignored_and_optional(self):
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        example = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn(".env", ignore)
        self.assertIn("NVIDIA_API_KEY=", example)
        self.assertIn("do not use this key", example)

    def test_scale_target_is_domain_and_schema_isolated(self):
        scale = self.bundle["targets"]["scale"]["variables"]
        self.assertEqual(scale["source_domain"], "Appliances")
        self.assertEqual(scale["schema_prefix"], "scale")
        self.assertTrue(scale["volume_root"].endswith("/scale/appliances"))

        class FakeSpark:
            def __init__(self):
                self.statements: list[str] = []

            def sql(self, statement):
                self.statements.append(statement)

        spark = FakeSpark()
        schemas = _ensure_schemas(spark, "workspace", "scale")
        self.assertEqual(schemas["bronze"], "workspace.scale_bronze")
        self.assertEqual(schemas["monitoring"], "workspace.scale_monitoring")
        with self.assertRaises(ValueError):
            _ensure_schemas(spark, "workspace", "scale;drop")

    def test_published_scale_evidence_is_bound_to_the_bundle(self):
        evidence = self.evidence["scale_benchmark"]
        dataset = evidence["dataset"]
        tables = evidence["tables"]
        initial = evidence["initial_run"]
        replay = evidence["replay_run"]
        scale = self.bundle["targets"]["scale"]["variables"]

        self.assertEqual(
            dataset["landed_lines"], dataset["review_lines"] + dataset["metadata_lines"]
        )
        self.assertEqual(
            dataset["total_bytes"], dataset["review_bytes"] + dataset["metadata_bytes"]
        )
        self.assertEqual(dataset["review_sha256"], scale["reviews_checksum"])
        self.assertEqual(dataset["metadata_sha256"], scale["metadata_checksum"])
        self.assertGreater(dataset["landed_lines"], 2_000_000)
        self.assertEqual(initial["status"], "SUCCESS")
        self.assertEqual(replay["status"], "SUCCESS")
        self.assertEqual(initial["assertion_count"], 11)
        self.assertEqual(replay["assertion_count"], 11)
        self.assertEqual(initial["source_set_sha256"], replay["source_set_sha256"])
        self.assertEqual(tables["gold_training_labels"], tables["gold_user_sequences_asof"])
        self.assertEqual(tables["gold_training_labels"], tables["gold_item_statistics_asof"])

    def test_cross_domain_evidence_uses_one_contract_and_exact_aggregates(self):
        evidence = self.evidence["cross_domain_validation"]
        runs = evidence["runs"]
        aggregate = evidence["aggregate"]

        self.assertEqual(evidence["classification"], "cross-domain schema portability")
        self.assertEqual(len(runs), 2)
        self.assertEqual({run["target"] for run in runs}, {"dev", "scale"})
        self.assertEqual(
            {run["category"] for run in runs},
            {"Magazine_Subscriptions", "Appliances"},
        )
        self.assertEqual({run["contract_version"] for run in runs}, {"lakehouse-certification/v2"})
        self.assertTrue(all(run["assertion_count"] == 13 for run in runs))
        self.assertTrue(all(run["status"] == "SUCCESS" for run in runs))
        for field in ("landed_lines", "silver_interactions", "gold_training_labels"):
            self.assertEqual(aggregate[field], sum(run[field] for run in runs))
        self.assertFalse(evidence["claims_production_multi_domain_scale"])

    def test_published_distributed_model_evidence_is_code_and_data_bound(self):
        evidence = self.evidence["distributed_recommender_benchmark"]
        run = evidence["run"]
        models = {row["label"]: row for row in evidence["models"]}

        self.assertEqual(run["status"], "SUCCESS")
        self.assertEqual(run["contract_version"], "spark-als-temporal-benchmark/v6")
        self.assertEqual(run["check_count"], 15)
        self.assertEqual(run["failed_check_count"], 0)
        self.assertEqual(
            run["implementation_sha256"], source_tree_sha256(ALS_IMPLEMENTATION_SOURCES)
        )
        self.assertEqual(len(run["benchmark_id"]), 64)
        self.assertGreater(evidence["training"]["users"], 100_000)
        self.assertEqual(evidence["evaluation"]["selection_split"], "validation")
        self.assertTrue(models["Popularity"]["is_champion"])
        self.assertTrue(models["Temporal hybrid RRF"]["is_validation_selected"])
        self.assertFalse(evidence["evaluation"]["release_qualified"])
        self.assertGreater(
            models["Temporal hybrid RRF"]["candidate_recall_at_100"],
            models["Popularity"]["candidate_recall_at_100"],
        )
        self.assertGreater(
            evidence["uncertainty_vs_popularity"]["candidate_recall_at_100"]["lower"], 0.0
        )
        self.assertLess(evidence["uncertainty_vs_popularity"]["ndcg_at_10"]["lower"], 0.0)

    def test_published_sequence_model_evidence_is_code_data_and_lineage_bound(self):
        evidence = self.evidence["sequential_recommender_benchmark"]
        run = evidence["run"]

        self.assertEqual(run["status"], "SUCCESS")
        self.assertEqual(run["contract_version"], "sasrec-temporal-benchmark/v3")
        self.assertEqual(run["check_count"], 8)
        self.assertEqual(run["failed_check_count"], 0)
        self.assertEqual(
            run["implementation_sha256"], source_tree_sha256(SASREC_IMPLEMENTATION_SOURCES)
        )
        self.assertGreaterEqual(evidence["dataset"]["training_users"], 30_000)
        self.assertEqual(evidence["deployment"]["status"], "rejected")
        self.assertIsNone(evidence["deployment"]["alias"])
        self.assertFalse(evidence["deployment"]["test_was_used_for_promotion"])
        self.assertEqual(evidence["artifacts"]["registered_model_version"], "2")
        self.assertLess(evidence["test"]["paired_ndcg_delta_vs_popularity"]["upper"], 0.0)

    def test_published_vector_search_evidence_is_code_data_and_slo_bound(self):
        evidence = self.evidence["managed_vector_search_benchmark"]
        implementation = ROOT / "src/marketplace_recommender/retrieval/vector_search.py"
        run = evidence["run"]

        self.assertEqual(run["status"], "SUCCESS")
        self.assertEqual(run["contract_version"], "databricks-ai-search-benchmark/v3")
        self.assertEqual(run["check_count"], 7)
        self.assertEqual(run["failed_check_count"], 0)
        self.assertEqual(
            run["implementation_sha256"], hashlib.sha256(implementation.read_bytes()).hexdigest()
        )
        self.assertEqual(evidence["service"]["indexed_rows"], 24_443)
        self.assertGreaterEqual(
            evidence["quality"]["ann_recall_at_10"],
            evidence["configuration"]["minimum_recall_at_10"],
        )
        self.assertLessEqual(
            evidence["load"]["latency_p95_ms"], evidence["configuration"]["maximum_p95_latency_ms"]
        )
        self.assertEqual(
            evidence["load"]["completed_requests"], evidence["configuration"]["load_request_count"]
        )

    def test_gold_plan_uses_strict_prior_event_windows(self):
        calls = function_calls(
            ROOT / "src/marketplace_recommender/pipelines/databricks_transform.py",
            "build_gold_tables",
        )
        range_calls = [call for call in calls if call_name(call) == "rangeBetween"]
        self.assertEqual(len(range_calls), 2)
        for call in range_calls:
            self.assertIsInstance(call.args[0], ast.Attribute)
            self.assertEqual(call.args[0].attr, "unboundedPreceding")
            self.assertIsInstance(call.args[1], ast.UnaryOp)
            self.assertIsInstance(call.args[1].op, ast.USub)
            self.assertEqual(call.args[1].operand.value, 1)
        self.assertTrue(any(call_name(call) == "slice" for call in calls))

    def test_bronze_plan_uses_content_addressing_without_row_order_identity(self):
        calls = function_calls(
            ROOT / "src/marketplace_recommender/pipelines/databricks_ingestion.py",
            "ingest_bronze_stream",
        )
        names = [call_name(call) for call in calls]
        self.assertNotIn("row_number", names)
        self.assertIn("sha2", names)
        self.assertIn("dropDuplicates", names)
        options = [
            call
            for call in calls
            if call_name(call) == "option"
            and len(call.args) >= 2
            and isinstance(call.args[0], ast.Constant)
        ]
        option_values = {
            (call.args[0].value, call.args[1].value)
            for call in options
            if isinstance(call.args[1], ast.Constant)
        }
        self.assertIn(("cloudFiles.format", "text"), option_values)

    def test_legacy_bronze_migration_converges_to_the_stream_identity(self):
        calls = function_calls(
            ROOT / "src/marketplace_recommender/pipelines/databricks_common.py",
            "_migrate_bronze_record_id",
        )
        names = {call_name(call) for call in calls}

        self.assertIn("sha2", names)
        self.assertIn("row_number", names)
        self.assertIn("saveAsTable", names)
        self.assertIn("count", names)
        self.assertIn("collect", names)
        self.assertIn("groupBy", names)
        self.assertIn("max", names)


if __name__ == "__main__":
    unittest.main()
