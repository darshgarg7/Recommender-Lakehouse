from __future__ import annotations

import math
from statistics import mean
from typing import Iterable, cast


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(index + 2) for index, item in enumerate(ranked[:k]) if item in relevant
    )
    ideal_count = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
    return dcg / ideal if ideal else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for index, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def aggregate_ranking_metrics(rows: Iterable[dict[str, object]]) -> dict[str, float]:
    examples = list(rows)
    if not examples:
        return {
            name: 0.0
            for name in ("ndcg_at_10", "ndcg_at_20", "recall_at_10", "recall_at_20", "mrr")
        }
    typed = [(cast(list[str], row["ranked"]), cast(set[str], row["relevant"])) for row in examples]
    return {
        "ndcg_at_10": mean(ndcg_at_k(ranked, relevant, 10) for ranked, relevant in typed),
        "ndcg_at_20": mean(ndcg_at_k(ranked, relevant, 20) for ranked, relevant in typed),
        "recall_at_10": mean(recall_at_k(ranked, relevant, 10) for ranked, relevant in typed),
        "recall_at_20": mean(recall_at_k(ranked, relevant, 20) for ranked, relevant in typed),
        "mrr": mean(reciprocal_rank(ranked, relevant) for ranked, relevant in typed),
    }
