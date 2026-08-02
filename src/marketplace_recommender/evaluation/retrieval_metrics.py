from __future__ import annotations

from statistics import mean
from typing import Iterable, cast

from marketplace_recommender.evaluation.ranking_metrics import recall_at_k


def retrieval_metrics(rows: Iterable[dict[str, object]], catalog: set[str]) -> dict[str, float]:
    examples = list(rows)
    typed = [(cast(list[str], row["ranked"]), cast(set[str], row["relevant"])) for row in examples]
    retrieved = {item for ranked, _ in typed for item in ranked}
    return {
        "recall_at_100": mean(recall_at_k(ranked, relevant, 100) for ranked, relevant in typed)
        if typed
        else 0.0,
        "recall_at_500": mean(recall_at_k(ranked, relevant, 500) for ranked, relevant in typed)
        if typed
        else 0.0,
        "candidate_coverage": len(retrieved) / len(catalog) if catalog else 0.0,
    }
