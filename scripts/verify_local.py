from __future__ import annotations

import json
from pathlib import Path

from marketplace_recommender.governance.receipts import verify_run_receipt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary_path = root / "artifacts/local/monitoring/local_run_summary.json"
    serving_path = root / "artifacts/local/serving/gold_batch_recommendations.jsonl"
    lineage_path = root / "artifacts/local/ml/local-hybrid-v1.json"
    receipt_path = root / "artifacts/local/monitoring/run_receipt.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    serving_rows = [
        json.loads(line) for line in serving_path.read_text(encoding="utf-8").splitlines()
    ]
    receipt = verify_run_receipt(receipt_path, root / "artifacts/local")

    assert summary["verified_claims"]["input_interactions"] == 240
    assert summary["verified_claims"]["input_products"] == 24
    assert summary["verified_claims"]["model_training_interactions"] == 144
    assert summary["metrics"]["full_reranked"]["example_count"] == 32
    assert summary["bronze"]["replayed_files"] == 2
    assert summary["promotion_decision"]["promoted"] is False
    assert summary["promotion_decision"]["serving_champion"] == "content_similarity"
    assert summary["verified_claims"]["serving_champion"] == "content_similarity"
    assert all(row["serving_champion"] == "content_similarity" for row in serving_rows)
    assert all(len(row["promotion_policy_id"]) == 64 for row in serving_rows)
    assert lineage["model_state"]["serving_champion"] == "content_similarity"
    assert (
        lineage["model_state"]["promotion_policy_id"]
        == summary["promotion_decision"]["policy"]["policy_id"]
    )
    assert all(row["representation_strategy"] for row in serving_rows)
    assert all(isinstance(row["evidence_capabilities"], list) for row in serving_rows)
    assert all(len(value) == 64 for value in summary["fingerprints"].values())
    assert receipt["valid"] is True
    assert receipt["payload_sha256"] == summary["run_receipt"]["payload_sha256"]
    assert len(receipt["verified_artifacts"]) == 7
    assert serving_path.stat().st_size > 0
    assert lineage_path.stat().st_size > 0
    print(
        "verified deterministic replay, champion routing, capability explanations, "
        "and tamper-evident evidence"
    )


if __name__ == "__main__":
    main()
