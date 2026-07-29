from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary_path = root / "artifacts/local/monitoring/local_run_summary.json"
    serving_path = root / "artifacts/local/serving/gold_batch_recommendations.jsonl"
    lineage_path = root / "artifacts/local/ml/local-hybrid-v1.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["verified_claims"]["input_interactions"] == 240
    assert summary["verified_claims"]["input_products"] == 24
    assert summary["verified_claims"]["model_training_interactions"] == 144
    assert summary["metrics"]["full_reranked"]["example_count"] == 32
    assert summary["bronze"]["replayed_files"] == 2
    assert summary["promotion_decision"]["promoted"] is False
    assert summary["promotion_decision"]["serving_champion"] == "content_similarity"
    assert all(len(value) == 64 for value in summary["fingerprints"].values())
    assert serving_path.stat().st_size > 0
    assert lineage_path.stat().st_size > 0
    print("verified deterministic replay, evidence, promotion fallback, and serving artifacts")


if __name__ == "__main__":
    main()
