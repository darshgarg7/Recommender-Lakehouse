import unittest
import hashlib
import json
import tempfile
from pathlib import Path

from marketplace_recommender.pipelines.databricks import (
    _ensure_schemas,
    _hash_landed_object,
    is_sha256,
)


ROOT = Path(__file__).resolve().parents[2]


class DatabricksBundleContractTests(unittest.TestCase):
    def test_bundle_is_serverless_and_has_checksum_bootstrap(self):
        bundle = (ROOT / "databricks.yml").read_text(encoding="utf-8")
        self.assertNotIn("existing_cluster_id", bundle)
        self.assertIn("task_key: bootstrap", bundle)
        self.assertIn("environment_key: default", bundle)
        self.assertIn("reviews_checksum", bundle)
        self.assertIn("metadata_checksum", bundle)

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

    def test_bundle_ends_in_a_run_bound_certification_task(self):
        bundle = (ROOT / "databricks.yml").read_text(encoding="utf-8")
        entrypoint = (ROOT / "scripts/databricks_entrypoint.py").read_text(encoding="utf-8")
        pipeline = (ROOT / "src/marketplace_recommender/pipelines/databricks.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("task_key: certify", bundle)
        self.assertIn('"{{job.run_id}}"', bundle)
        self.assertIn("certify_pipeline_run", entrypoint)
        self.assertIn("pipeline_run_certifications", pipeline)

    def test_bundle_has_independent_distributed_als_benchmark(self):
        bundle = (ROOT / "databricks.yml").read_text(encoding="utf-8")
        entrypoint = (ROOT / "scripts/databricks_entrypoint.py").read_text(encoding="utf-8")
        benchmark = (ROOT / "src/marketplace_recommender/retrieval/spark_als.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("marketplace_als_benchmark", bundle)
        self.assertIn("task_key: train_validate_certify_als", bundle)
        self.assertIn('"{{job.run_id}}"', bundle)
        self.assertIn('"train-als"', entrypoint)
        self.assertIn("pyspark.ml.recommendation import ALS", benchmark)
        self.assertIn("recommender_benchmark_certifications", benchmark)
        self.assertIn('coldStartStrategy="drop"', benchmark)
        self.assertNotIn("from pyspark.ml.feature import StringIndexer", benchmark)
        self.assertIn("Test never tunes a hyperparameter", benchmark)
        self.assertIn('"implementation_sha256": implementation_sha256', benchmark)
        self.assertIn("paired_bootstrap_mean_difference", benchmark)
        self.assertIn("recommender_benchmark_uncertainty", benchmark)

    def test_bundle_has_real_sequence_model_and_managed_ann_benchmarks(self):
        bundle = (ROOT / "databricks.yml").read_text(encoding="utf-8")
        entrypoint = (ROOT / "scripts/databricks_entrypoint.py").read_text(encoding="utf-8")
        sasrec = (ROOT / "src/marketplace_recommender/retrieval/sasrec_torch.py").read_text(
            encoding="utf-8"
        )
        vector_search = (ROOT / "src/marketplace_recommender/retrieval/vector_search.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("marketplace_sasrec_benchmark", bundle)
        self.assertIn("marketplace_vector_search_benchmark", bundle)
        self.assertIn('"train-sasrec"', entrypoint)
        self.assertIn('"benchmark-vector-search"', entrypoint)
        self.assertIn("nn.TransformerEncoder", sasrec)
        self.assertIn("best_epoch_selected_on_validation", sasrec)
        self.assertIn("registered_model_name", sasrec)
        self.assertIn("test_was_used_for_promotion", sasrec)
        self.assertIn("maximum-inner-product-to-l2", vector_search)
        self.assertIn("ThreadPoolExecutor", vector_search)
        self.assertIn("ann_recall_meets_contract", vector_search)

    def test_local_secrets_are_ignored_and_optional(self):
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        example = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn(".env", ignore)
        self.assertIn("NVIDIA_API_KEY=", example)
        self.assertIn("do not use this key", example)

    def test_scale_target_is_domain_and_schema_isolated(self):
        bundle = (ROOT / "databricks.yml").read_text(encoding="utf-8")
        self.assertIn("source_domain: Appliances", bundle)
        self.assertIn("schema_prefix: scale", bundle)
        self.assertIn("marketplace_landing/scale/appliances", bundle)

        class FakeSpark:
            def __init__(self):
                self.statements = []

            def sql(self, statement):
                self.statements.append(statement)

        spark = FakeSpark()
        schemas = _ensure_schemas(spark, "workspace", "scale")
        self.assertEqual(schemas["bronze"], "workspace.scale_bronze")
        self.assertEqual(schemas["monitoring"], "workspace.scale_monitoring")
        with self.assertRaises(ValueError):
            _ensure_schemas(spark, "workspace", "scale;drop")

    def test_published_scale_evidence_is_bound_to_the_bundle(self):
        bundle = (ROOT / "databricks.yml").read_text(encoding="utf-8")
        evidence = json.loads(
            (ROOT / "benchmarks/reports/portfolio_evidence.json").read_text(encoding="utf-8")
        )["scale_benchmark"]
        dataset = evidence["dataset"]
        tables = evidence["tables"]
        initial = evidence["initial_run"]
        replay = evidence["replay_run"]

        self.assertEqual(
            dataset["landed_lines"], dataset["review_lines"] + dataset["metadata_lines"]
        )
        self.assertEqual(
            dataset["total_bytes"], dataset["review_bytes"] + dataset["metadata_bytes"]
        )
        self.assertIn(dataset["review_sha256"], bundle)
        self.assertIn(dataset["metadata_sha256"], bundle)
        self.assertGreater(dataset["landed_lines"], 2_000_000)
        self.assertEqual(initial["status"], "SUCCESS")
        self.assertEqual(replay["status"], "SUCCESS")
        self.assertEqual(initial["assertion_count"], 11)
        self.assertEqual(replay["assertion_count"], 11)
        self.assertEqual(initial["source_set_sha256"], replay["source_set_sha256"])
        self.assertEqual(
            tables["gold_training_labels"],
            tables["gold_user_sequences_asof"],
        )
        self.assertEqual(
            tables["gold_training_labels"],
            tables["gold_item_statistics_asof"],
        )

    def test_published_distributed_model_evidence_is_code_and_data_bound(self):
        evidence = json.loads(
            (ROOT / "benchmarks/reports/portfolio_evidence.json").read_text(encoding="utf-8")
        )["distributed_recommender_benchmark"]
        implementation = ROOT / "src/marketplace_recommender/retrieval/spark_als.py"
        run = evidence["run"]
        models = {row["label"]: row for row in evidence["models"]}

        self.assertEqual(run["status"], "SUCCESS")
        self.assertEqual(run["contract_version"], "spark-als-temporal-benchmark/v4")
        self.assertEqual(run["check_count"], 15)
        self.assertEqual(run["failed_check_count"], 0)
        self.assertEqual(
            run["implementation_sha256"], hashlib.sha256(implementation.read_bytes()).hexdigest()
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
        interval = evidence["uncertainty_vs_popularity"]["candidate_recall_at_100"]
        self.assertGreater(interval["lower"], 0.0)
        self.assertLess(evidence["uncertainty_vs_popularity"]["ndcg_at_10"]["lower"], 0.0)
        self.assertTrue((ROOT / "assets/distributed-model-benchmark.svg").exists())

    def test_published_sequence_model_evidence_is_code_data_and_lineage_bound(self):
        evidence = json.loads(
            (ROOT / "benchmarks/reports/portfolio_evidence.json").read_text(encoding="utf-8")
        )["sequential_recommender_benchmark"]
        implementation = ROOT / "src/marketplace_recommender/retrieval/sasrec_torch.py"
        run = evidence["run"]

        self.assertEqual(run["status"], "SUCCESS")
        self.assertEqual(run["contract_version"], "sasrec-temporal-benchmark/v1")
        self.assertEqual(run["check_count"], 8)
        self.assertEqual(run["failed_check_count"], 0)
        self.assertEqual(
            run["implementation_sha256"], hashlib.sha256(implementation.read_bytes()).hexdigest()
        )
        self.assertEqual(evidence["dataset"]["source_delta_version"], 2)
        self.assertGreaterEqual(evidence["dataset"]["training_users"], 30_000)
        self.assertEqual(evidence["deployment"]["status"], "rejected")
        self.assertIsNone(evidence["deployment"]["alias"])
        self.assertFalse(evidence["deployment"]["test_was_used_for_promotion"])
        self.assertEqual(evidence["artifacts"]["registered_model_version"], "1")
        self.assertLess(evidence["test"]["paired_ndcg_delta_vs_popularity"]["upper"], 0.0)

    def test_published_vector_search_evidence_is_code_data_and_slo_bound(self):
        evidence = json.loads(
            (ROOT / "benchmarks/reports/portfolio_evidence.json").read_text(encoding="utf-8")
        )["managed_vector_search_benchmark"]
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
        self.assertEqual(evidence["service"]["vector_dimension"], 65)
        self.assertGreaterEqual(
            evidence["quality"]["ann_recall_at_10"],
            evidence["configuration"]["minimum_recall_at_10"],
        )
        self.assertLessEqual(
            evidence["load"]["latency_p95_ms"],
            evidence["configuration"]["maximum_p95_latency_ms"],
        )
        self.assertEqual(
            evidence["load"]["completed_requests"],
            evidence["configuration"]["load_request_count"],
        )

    def test_gold_history_plans_use_strict_point_in_time_windows(self):
        pipeline = (ROOT / "src/marketplace_recommender/pipelines/databricks.py").read_text(
            encoding="utf-8"
        )
        gold_plan = pipeline.split("def build_gold_tables", 1)[1].split(
            "def certify_pipeline_run", 1
        )[0]
        self.assertEqual(gold_plan.count("rangeBetween(Window.unboundedPreceding, -1)"), 2)
        self.assertNotIn("LEFT JOIN", gold_plan)
        self.assertIn("F.slice(", gold_plan)

    def test_bronze_identity_is_parallel_content_addressing(self):
        pipeline = (ROOT / "src/marketplace_recommender/pipelines/databricks.py").read_text(
            encoding="utf-8"
        )
        bronze_plan = pipeline.split("def ingest_bronze_stream", 1)[1].split(
            "def build_silver_tables", 1
        )[0]
        self.assertNotIn("row_number()", bronze_plan)
        self.assertNotIn("F.row_number", bronze_plan)
        self.assertIn('F.sha2("raw_payload", 256)', bronze_plan)
        self.assertIn('.dropDuplicates(["bronze_record_id"])', bronze_plan)
        self.assertIn('.option("cloudFiles.format", "text")', bronze_plan)
        self.assertNotIn("cloudFiles.inferColumnTypes", bronze_plan)
        self.assertIn('F.col("value").alias("raw_payload")', bronze_plan)


if __name__ == "__main__":
    unittest.main()
