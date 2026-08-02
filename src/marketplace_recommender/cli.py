from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from marketplace_recommender.config import PipelineConfig
from marketplace_recommender.evaluation.bootstrap import bootstrap_mean_ci
from marketplace_recommender.evaluation.cohort_metrics import marketplace_metrics
from marketplace_recommender.evaluation.experiment import evaluate_rankings
from marketplace_recommender.evaluation.ranking_metrics import ndcg_at_k
from marketplace_recommender.evaluation.temporal_split import temporal_cutoffs
from marketplace_recommender.governance.promotion import PromotionPolicy, promotion_decision
from marketplace_recommender.governance.receipts import verify_run_receipt, write_run_receipt
from marketplace_recommender.ingestion.manifest import ManifestStore
from marketplace_recommender.ingestion.downloader import BoundedDownloader
from marketplace_recommender.mlops import ExperimentLineage, code_revision
from marketplace_recommender.monitoring.metrics import MetricLog
from marketplace_recommender.pipelines.bronze import ingest_manifest_objects
from marketplace_recommender.pipelines.gold import build_gold
from marketplace_recommender.pipelines.silver import build_silver
from marketplace_recommender.schemas import cold_start_bucket
from marketplace_recommender.serving.batch import RecommendationSystem, serving_projection
from marketplace_recommender.storage import (
    file_checksum,
    read_jsonl,
    records_fingerprint,
    write_jsonl_atomic,
)
from marketplace_recommender.synthetic import generate_local_source


def _register_local_sources(paths: dict[str, Path], manifest: ManifestStore, domain: str) -> None:
    existing = {(row["source_url"], row["checksum"]) for row in manifest.all()}
    now = datetime.now(timezone.utc).isoformat()
    for kind, path in sorted(paths.items()):
        checksum = file_checksum(path)
        url = f"file://{path.resolve()}"
        if (url, checksum) in existing:
            continue
        manifest.upsert(
            {
                "source_url": url,
                "source_domain": domain,
                "source_kind": kind,
                "object_path": str(path.resolve()),
                "compressed_bytes": path.stat().st_size,
                "checksum": checksum,
                "download_started_at": now,
                "download_completed_at": now,
                "download_status": "complete",
                "retry_count": 0,
                "ingestion_status": "pending",
            }
        )


def _model_scope(
    interactions: list[dict[str, Any]], user_limit: int | None, seed: int
) -> list[dict[str, Any]]:
    if user_limit is None:
        return interactions
    counts: dict[str, int] = {}
    for row in interactions:
        counts[row["user_id"]] = counts.get(row["user_id"], 0) + 1
    eligible = [user for user, count in counts.items() if count >= 3]
    eligible.sort(key=lambda user: hashlib.sha256(f"{seed}:{user}".encode()).hexdigest())
    chosen = set(eligible[:user_limit])
    scoped = [row for row in interactions if row["user_id"] in chosen]
    if len({row["review_timestamp"] for row in scoped}) < 3:
        raise RuntimeError("model scope needs at least three distinct event timestamps")
    return scoped


def run_demo(
    config_path: str | Path,
    *,
    source_paths: dict[str, Path] | None = None,
    data_source: str = "deterministic synthetic Amazon-shaped data",
    register_sources: bool = True,
) -> dict[str, Any]:
    config = PipelineConfig.from_file(config_path)
    output = config.output_dir.resolve()
    landing = output / "landing"
    bronze = output / "bronze"
    silver = output / "silver"
    gold = output / "gold"
    serving = output / "serving"
    monitoring = output / "monitoring"
    metrics_log = MetricLog()

    if source_paths is None:
        with metrics_log.timed("source_generation"):
            source_paths = generate_local_source(landing, config.interaction_count, config.seed)
    manifest = ManifestStore(bronze / "manifest.jsonl")
    if register_sources:
        _register_local_sources(source_paths, manifest, config.domains[0])
    with metrics_log.timed("bronze"):
        bronze_counts = ingest_manifest_objects(manifest, bronze)
    with metrics_log.timed("silver"):
        silver_counts = build_silver(bronze, silver)
    etl_interactions = list(read_jsonl(silver / "silver_interactions.jsonl"))
    products = list(read_jsonl(silver / "silver_products.jsonl"))
    if not etl_interactions or not products:
        raise RuntimeError("the vertical slice requires at least one valid interaction and product")
    interactions = _model_scope(etl_interactions, config.model_user_limit, config.seed)
    gold_source = silver
    if len(interactions) != len(etl_interactions):
        gold_source = output / "model_input"
        write_jsonl_atomic(gold_source / "silver_interactions.jsonl", interactions)
        write_jsonl_atomic(gold_source / "silver_products.jsonl", products)
    cutoffs = temporal_cutoffs(interactions, config.validation_fraction, config.test_fraction)
    with metrics_log.timed("gold"):
        gold_counts = build_gold(gold_source, gold, cutoffs, config.sequence_max_length)
    content = list(read_jsonl(gold / "gold_item_content_features.jsonl"))
    graph = list(read_jsonl(silver / "silver_bought_together_edges.jsonl"))
    labels = list(read_jsonl(gold / "gold_training_labels.jsonl"))

    with metrics_log.timed("model_fit"):
        system = RecommendationSystem.fit(
            interactions, products, content, graph, cutoffs.training_end, config.seed
        )
        ranker_rows = system.fit_ranker(
            (row for row in labels if row["split"] == "validation"),
            config.candidate_limit,
            config.negative_count,
        )

    evaluations: dict[str, list[dict[str, Any]]] = {
        "popularity": [],
        "content_similarity": [],
        "hybrid_ranker": [],
        "full_reranked": [],
    }
    evaluation_candidates: list[dict[str, Any]] = []
    test_labels = [row for row in labels if row["split"] == "test" and row["label"] == 1]
    with metrics_log.timed("evaluation"):
        for label in test_labels:
            baselines = system.baseline_rankings(
                label["user_id"], label["label_timestamp"], config.candidate_limit
            )
            relevance, final = system.recommend(
                label["user_id"],
                label["label_timestamp"],
                config.candidate_limit,
                config.recommendation_limit,
                config.rerank,
            )
            cohort = cold_start_bucket(
                sum(
                    row["parent_asin"] == label["parent_asin"]
                    and row["review_timestamp"] < cutoffs.training_end
                    and row["verified_purchase"]
                    and row["rating"] >= 4
                    for row in interactions
                )
            )
            common = {
                "target": label["parent_asin"],
                "user_id": label["user_id"],
                "label_timestamp": label["label_timestamp"],
                "cohort": cohort,
            }
            evaluations["popularity"].append(
                {**common, "ranked": baselines["popularity"], "candidates": baselines["popularity"]}
            )
            evaluations["content_similarity"].append(
                {
                    **common,
                    "ranked": baselines["content_similarity"],
                    "candidates": baselines["content_similarity"],
                }
            )
            evaluations["hybrid_ranker"].append(
                {
                    **common,
                    "ranked": [row["parent_asin"] for row in relevance],
                    "candidates": [row["parent_asin"] for row in relevance],
                }
            )
            evaluations["full_reranked"].append(
                {
                    **common,
                    "ranked": [row["parent_asin"] for row in final],
                    "candidates": [row["parent_asin"] for row in relevance],
                }
            )
            evaluation_candidates.extend(
                {
                    "interaction_id": label["interaction_id"],
                    "user_id": label["user_id"],
                    "label_timestamp": label["label_timestamp"],
                    "target_parent_asin": label["parent_asin"],
                    "parent_asin": row["parent_asin"],
                    "retrieval_score": row["retrieval_score"],
                    "ranking_score": row["ranking_score"],
                    "is_relevant": row["parent_asin"] == label["parent_asin"],
                }
                for row in relevance
            )

    catalog = set(system.products)
    evaluation_report = {
        model: evaluate_rankings(rows, catalog) for model, rows in evaluations.items()
    }
    promotion = promotion_decision(
        evaluation_report,
        policy=PromotionPolicy.from_mapping(config.promotion),
    )
    best_relevance_model = max(
        ("popularity", "content_similarity", "hybrid_ranker"),
        key=lambda name: evaluation_report[name]["ranking"]["ndcg_at_10"],
    )
    deltas_by_user: dict[str, list[float]] = {}
    for baseline_row, full_row in zip(
        evaluations[best_relevance_model], evaluations["full_reranked"]
    ):
        target = {baseline_row["target"]}
        delta = ndcg_at_k(full_row["ranked"], target, 10) - ndcg_at_k(
            baseline_row["ranked"], target, 10
        )
        deltas_by_user.setdefault(baseline_row["user_id"], []).append(delta)
    user_deltas = [sum(values) / len(values) for values in deltas_by_user.values()]
    comparison_ci = bootstrap_mean_ci(user_deltas, samples=1_000, seed=config.seed)
    active_users = sorted({row["user_id"] for row in interactions})
    batch_rows: list[dict[str, Any]] = []
    batch_metric_rows: list[dict[str, Any]] = []
    batch_timestamp = cutoffs.test_end + 1
    with metrics_log.timed("batch_inference"):
        for user_id in active_users:
            recommendations = system.recommend_champion(
                promotion["serving_champion"],
                user_id,
                batch_timestamp,
                config.candidate_limit,
                config.recommendation_limit,
                config.rerank,
                promotion["policy"]["policy_id"],
            )
            batch_metric_rows.extend(recommendations)
            batch_rows.extend(serving_projection(row) for row in recommendations)

    frontier = []
    frontier_limit = min(10, config.recommendation_limit)
    for tail_weight in (0.0, 0.10, 0.25, 0.50, 1.00, 2.00):
        frontier_rows = []
        exposure_rows = []
        frontier_config = {**config.rerank, "long_tail_weight": tail_weight}
        for label in test_labels:
            _, recommendations = system.recommend(
                label["user_id"],
                label["label_timestamp"],
                config.candidate_limit,
                frontier_limit,
                frontier_config,
            )
            frontier_rows.append(
                {
                    "target": label["parent_asin"],
                    "user_id": label["user_id"],
                    "label_timestamp": label["label_timestamp"],
                    "cohort": cold_start_bucket(
                        system.tower.interaction_counts.get(label["parent_asin"], 0)
                    ),
                    "ranked": [row["parent_asin"] for row in recommendations],
                }
            )
            exposure_rows.extend(recommendations)
        frontier.append(
            {
                "long_tail_weight": tail_weight,
                "list_size": frontier_limit,
                "ndcg_at_10": evaluate_rankings(frontier_rows, catalog)["ranking"]["ndcg_at_10"],
                **marketplace_metrics(exposure_rows, catalog),
            }
        )

    write_jsonl_atomic(
        gold / "gold_item_embeddings.jsonl",
        (
            {
                "parent_asin": item,
                "content_embedding": system.content_embeddings[item],
                "collaborative_embedding": system.tower.collaborative_embeddings.get(item),
                "hybrid_embedding": vector,
                "model_version": "local-hybrid-v1",
            }
            for item, vector in sorted(system.generator.ann.vectors.items())
        ),
    )
    write_jsonl_atomic(gold / "gold_evaluation_candidates.jsonl", evaluation_candidates)
    write_jsonl_atomic(serving / "gold_batch_recommendations.jsonl", batch_rows)
    revision = code_revision()
    source_versions = {row["source_kind"]: row["checksum"] for row in manifest.all()}
    lineage = ExperimentLineage(
        source_versions=source_versions,
        training_cutoff=cutoffs.training_end,
        validation_cutoff=cutoffs.validation_end,
        test_cutoff=cutoffs.test_end,
        eligible_domains=list(config.domains),
        user_count=len(active_users),
        item_count=len(products),
        interaction_count=sum(
            row["review_timestamp"] < cutoffs.training_end for row in interactions
        ),
        negative_sampling_policy=(
            f"top-{config.negative_count}-retrieved-hard-negatives-with-future-positive-exclusion"
        ),
        seed=config.seed,
        code_revision=revision,
        compute_configuration="single-process dependency-free local",
        estimated_cost="$0 incremental cloud cost",
    )
    lineage_path = output / "ml" / "local-hybrid-v1.json"
    lineage.write(
        lineage_path,
        {
            "ranker_weights": system.ranker.weights,
            "distillation_scale": system.tower.distiller.scale,
            "embedding_dimension": len(next(iter(system.generator.ann.vectors.values()))),
            "candidate_promoted": promotion["promoted"],
            "serving_champion": promotion["serving_champion"],
            "promotion_policy_id": promotion["policy"]["policy_id"],
        },
    )
    monitoring.mkdir(parents=True, exist_ok=True)
    frontier_path = monitoring / "relevance_long_tail_frontier.json"
    frontier_path.write_text(
        json.dumps(frontier, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verified_claims = {
        "input_interactions": len(etl_interactions),
        "input_products": len(products),
        "model_scope_interactions": len(interactions),
        "model_training_interactions": sum(
            row["review_timestamp"] < cutoffs.training_end for row in interactions
        ),
        "ranker_training_rows": ranker_rows,
        "batch_recommendation_rows": len(batch_rows),
        "serving_champion": promotion["serving_champion"],
    }
    fingerprints = {
        "silver_interactions": records_fingerprint(etl_interactions),
        "gold_training_examples": records_fingerprint(
            read_jsonl(gold / "gold_training_examples.jsonl")
        ),
    }
    receipt_path = monitoring / "run_receipt.json"
    receipt = write_run_receipt(
        receipt_path,
        run_root=output,
        identity={
            "tier": config.tier,
            "data_source": data_source,
            "model_version": "local-hybrid-v1",
            "seed": config.seed,
            "code_revision": revision,
            "config_sha256": file_checksum(Path(config_path)),
        },
        source_contract=source_versions,
        temporal_contract={
            "training_end": cutoffs.training_end,
            "validation_end": cutoffs.validation_end,
            "test_end": cutoffs.test_end,
            "batch_feature_timestamp": batch_timestamp,
            "historical_join_predicate": "event_timestamp < label_timestamp",
        },
        decision_contract=promotion,
        verified_claims={**verified_claims, "fingerprints": fingerprints},
        artifacts={
            "silver_interactions": silver / "silver_interactions.jsonl",
            "gold_training_examples": gold / "gold_training_examples.jsonl",
            "gold_item_embeddings": gold / "gold_item_embeddings.jsonl",
            "gold_evaluation_candidates": gold / "gold_evaluation_candidates.jsonl",
            "batch_recommendations": serving / "gold_batch_recommendations.jsonl",
            "model_lineage": lineage_path,
            "marketplace_frontier": frontier_path,
        },
    )
    receipt_verification = verify_run_receipt(receipt_path, output)
    summary = {
        "tier": config.tier,
        "data_source": data_source,
        "verified_claims": verified_claims,
        "cutoffs": {
            "training_end": cutoffs.training_end,
            "validation_end": cutoffs.validation_end,
            "test_end": cutoffs.test_end,
        },
        "bronze": bronze_counts,
        "silver": silver_counts,
        "gold": gold_counts,
        "metrics": evaluation_report,
        "promotion_decision": promotion,
        "comparison": {
            "metric": f"user-mean full_reranked minus {best_relevance_model} NDCG@10",
            "best_relevance_model": best_relevance_model,
            "mean_delta": comparison_ci[0],
            "aggregate_relative_delta": (
                evaluation_report["full_reranked"]["ranking"]["ndcg_at_10"]
                / evaluation_report[best_relevance_model]["ranking"]["ndcg_at_10"]
                - 1.0
            ),
            "bootstrap_95pct_lower": comparison_ci[1],
            "bootstrap_95pct_upper": comparison_ci[2],
            "users": len(user_deltas),
        },
        "batch_marketplace_metrics": marketplace_metrics(batch_metric_rows, catalog),
        "relevance_long_tail_frontier": frontier,
        "runtime": metrics_log.values,
        "fingerprints": fingerprints,
        "run_receipt": {
            "path": receipt_path.relative_to(output).as_posix(),
            "payload_sha256": receipt["payload_sha256"],
            "verified_artifact_count": len(receipt_verification["verified_artifacts"]),
        },
        "limitations": [
            (
                "Local results use synthetic review events, not impressions or clicks."
                if data_source.startswith("deterministic synthetic")
                else "Amazon review events are observations, not impressions or clicks."
            ),
            "Unobserved products are not confirmed negatives.",
            "No CTR, conversion, causal lift, Databricks scale, latency, or cost claim is made.",
        ],
    }
    (monitoring / "local_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def run_real_integration(config_path: str | Path, category: str) -> dict[str, Any]:
    if not category.replace("_", "").isalnum():
        raise ValueError("category may contain only letters, numbers, and underscores")
    config = PipelineConfig.from_file(config_path)
    output = config.output_dir.resolve()
    landing = output / "landing"
    manifest = ManifestStore(output / "bronze" / "manifest.jsonl")
    base = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"
    objects = [
        {
            "source_url": f"{base}/review_categories/{category}.jsonl?download=true",
            "source_domain": category,
            "source_kind": "reviews",
            "filename": f"reviews_{category}.jsonl",
        },
        {
            "source_url": f"{base}/meta_categories/meta_{category}.jsonl?download=true",
            "source_domain": category,
            "source_kind": "metadata",
            "filename": f"meta_{category}.jsonl",
        },
    ]
    downloaded = BoundedDownloader(manifest, workers=2, retries=3).download_all(objects, landing)
    paths = {str(row["source_kind"]): Path(str(row["object_path"])) for row in downloaded}
    return run_demo(
        config_path,
        source_paths=paths,
        data_source=f"Amazon Reviews 2023 raw {category} category",
        register_sources=False,
    )


def download_real_category(output_dir: str | Path, category: str) -> dict[str, Any]:
    """Download and validate a real category without loading it into the local in-memory model."""
    if not category.replace("_", "").isalnum():
        raise ValueError("category may contain only letters, numbers, and underscores")
    output = Path(output_dir).resolve()
    landing = output / "landing"
    manifest = ManifestStore(output / "bronze" / "manifest.jsonl")
    base = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"
    objects = [
        {
            "source_url": f"{base}/review_categories/{category}.jsonl?download=true",
            "source_domain": category,
            "source_kind": "reviews",
            "filename": f"reviews_{category}.jsonl",
        },
        {
            "source_url": f"{base}/meta_categories/meta_{category}.jsonl?download=true",
            "source_domain": category,
            "source_kind": "metadata",
            "filename": f"meta_{category}.jsonl",
        },
    ]
    downloaded = BoundedDownloader(manifest, workers=2, retries=4).download_all(objects, landing)
    report = {
        "category": category,
        "total_bytes": sum(cast(int, row["compressed_bytes"]) for row in downloaded),
        "total_rows": sum(cast(int, row["validated_rows"]) for row in downloaded),
        "objects": [
            {
                "source_kind": row["source_kind"],
                "object_path": row["object_path"],
                "bytes": row["compressed_bytes"],
                "rows": row["validated_rows"],
                "sha256": row["checksum"],
            }
            for row in sorted(downloaded, key=lambda value: str(value["source_kind"]))
        ],
    }
    report_path = output / "monitoring" / "source_download.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="marketplace-recommender")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="run the deterministic local vertical slice")
    demo.add_argument("--config", default="conf/local.yml")
    real = commands.add_parser(
        "real-demo", help="download and run a real Amazon Reviews 2023 slice"
    )
    real.add_argument("--config", default="conf/real_local.yml")
    real.add_argument("--category", default="Magazine_Subscriptions")
    download = commands.add_parser(
        "download-category", help="download and validate a category without local model training"
    )
    download.add_argument("--category", required=True)
    download.add_argument("--output", required=True)
    verify = commands.add_parser(
        "verify-receipt", help="verify a tamper-evident run receipt and all bound artifacts"
    )
    verify.add_argument("--root", default="artifacts/local")
    verify.add_argument("--receipt", default="artifacts/local/monitoring/run_receipt.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "demo":
        run_demo(args.config)
    elif args.command == "real-demo":
        run_real_integration(args.config, args.category)
    elif args.command == "download-category":
        download_real_category(args.output, args.category)
    elif args.command == "verify-receipt":
        print(json.dumps(verify_run_receipt(args.receipt, args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
