from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from marketplace_recommender.retrieval.distillation import DiagonalDistiller
from marketplace_recommender.retrieval.vectors import (
    hashed_id_vector,
    mean,
    normalize,
)


def cold_start_gate(interaction_count: int, has_content: bool = True) -> float:
    """Collaboration weight: exactly zero for no history and asymptotic for warm items."""
    if interaction_count <= 0:
        return 0.0
    if not has_content:
        return 1.0
    return interaction_count / (interaction_count + 10.0)


def build_collaborative_embeddings(
    teacher_scores: dict[str, Counter[str]], dimension: int = 48
) -> dict[str, list[float]]:
    output: dict[str, list[float]] = {}
    all_items = set(teacher_scores)
    all_items.update(item for neighbors in teacher_scores.values() for item in neighbors)
    for item in sorted(all_items):
        neighbors = teacher_scores.get(item, {})
        vectors = [hashed_id_vector(neighbor, dimension) for neighbor in neighbors]
        weights = list(neighbors.values())
        output[item] = (
            normalize(mean(vectors, weights)) if vectors else hashed_id_vector(item, dimension)
        )
    return output


@dataclass
class HybridTwoTower:
    content_embeddings: dict[str, list[float]]
    collaborative_embeddings: dict[str, list[float]]
    interaction_counts: Counter[str]
    has_content: dict[str, bool]
    distiller: DiagonalDistiller

    @classmethod
    def fit(
        cls,
        content_embeddings: dict[str, list[float]],
        teacher_scores: dict[str, Counter[str]],
        interactions: Iterable[dict[str, Any]],
        cutoff: int,
        has_content: dict[str, bool],
    ) -> "HybridTwoTower":
        counts = Counter(
            row["parent_asin"]
            for row in interactions
            if row["review_timestamp"] < cutoff and row["verified_purchase"] and row["rating"] >= 4
        )
        collaborative = build_collaborative_embeddings(teacher_scores)
        warm = {item for item, count in counts.items() if count > 1}
        distiller = DiagonalDistiller().fit(content_embeddings, collaborative, warm)
        return cls(content_embeddings, collaborative, counts, has_content, distiller)

    def item_embedding(self, item: str, *, simulate_zero_history: bool = False) -> list[float]:
        content = self.distiller.transform(self.content_embeddings.get(item, [0.0] * 48))
        count = 0 if simulate_zero_history else self.interaction_counts.get(item, 0)
        gate = cold_start_gate(count, self.has_content.get(item, False))
        collaboration = self.collaborative_embeddings.get(item, [0.0] * len(content))
        return normalize(
            [gate * collab + (1.0 - gate) * cont for collab, cont in zip(collaboration, content)]
        )

    def user_embedding(self, history: list[str]) -> list[float]:
        vectors = [self.item_embedding(item) for item in history if item in self.content_embeddings]
        if not vectors:
            return [0.0] * 48
        weights = [math.exp(index / max(len(vectors), 1)) for index in range(len(vectors))]
        return normalize(mean(vectors, weights))
