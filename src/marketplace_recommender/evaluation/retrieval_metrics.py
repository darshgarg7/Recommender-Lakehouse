from __future__ import annotations

from statistics import mean
from typing import Iterable

from marketplace_recommender.evaluation.ranking_metrics import recall_at_k


def retrieval_metrics(rows: Iterable[dict[str, object]], catalog: set[str]) -> dict[str, float]:
    examples = list(rows)
    retrieved = {item for row in examples for item in row["ranked"]}
    return {
        "recall_at_100": mean(recall_at_k(row["ranked"], row["relevant"], 100) for row in examples)
        if examples
        else 0.0,
        "recall_at_500": mean(recall_at_k(row["ranked"], row["relevant"], 500) for row in examples)
        if examples
        else 0.0,
        "candidate_coverage": len(retrieved) / len(catalog) if catalog else 0.0,
    }
