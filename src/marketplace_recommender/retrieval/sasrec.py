from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


class SequentialCooccurrenceTeacher:
    """Deterministic local sequence oracle; managed benchmarks use the causal SASRec path."""

    def __init__(self) -> None:
        self.transitions: dict[str, dict[str, float]] = defaultdict(dict)

    def fit(
        self, interactions: Iterable[dict[str, Any]], cutoff: int
    ) -> "SequentialCooccurrenceTeacher":
        by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in interactions:
            if row["review_timestamp"] < cutoff and row["verified_purchase"] and row["rating"] >= 4:
                by_user[row["user_id"]].append(row)
        for rows in by_user.values():
            rows.sort(key=lambda row: row["review_timestamp"])
            items = [row["parent_asin"] for row in rows]
            for index, source in enumerate(items):
                for distance, target in enumerate(items[index + 1 : index + 4], start=1):
                    if source != target:
                        neighbors = self.transitions[source]
                        neighbors[target] = neighbors.get(target, 0.0) + 1.0 / distance
        return self

    def score(self, history: list[str]) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        for recency, source in enumerate(reversed(history[-10:]), start=1):
            for target, weight in self.transitions.get(source, {}).items():
                scores[target] += weight / recency
        maximum = max(scores.values(), default=1.0)
        return {item: score / maximum for item, score in scores.items()}
