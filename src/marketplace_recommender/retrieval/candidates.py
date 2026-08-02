from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from marketplace_recommender.retrieval.ann import ExactANNIndex
from marketplace_recommender.retrieval.popularity import recommend_popular
from marketplace_recommender.retrieval.sasrec import SequentialCooccurrenceTeacher
from marketplace_recommender.retrieval.two_tower import HybridTwoTower


@dataclass
class CandidateGenerator:
    tower: HybridTwoTower
    ann: ExactANNIndex
    teacher: SequentialCooccurrenceTeacher
    popularity: dict[str, float]
    graph: dict[str, set[str]]

    def generate(
        self, history: list[str], limit: int, domain: str | None = None
    ) -> list[dict[str, Any]]:
        seen = set(history)
        channels: dict[str, dict[str, float]] = {}

        def add(channel: str, rows: list[tuple[str, float]]) -> None:
            for item, score in rows:
                if item in seen:
                    continue
                channels.setdefault(item, {})[channel] = float(score)

        user_vector = self.tower.user_embedding(history)
        add(
            "hybrid_ann", self.ann.query(user_vector, min(400, limit), excluded=seen, domain=domain)
        )
        sequential = self.teacher.score(history)
        add(
            "cointeraction",
            sorted(sequential.items(), key=lambda pair: (-pair[1], pair[0]))[: min(200, limit)],
        )
        graph_scores: dict[str, float] = defaultdict(float)
        for recency, item in enumerate(reversed(history[-10:]), start=1):
            for neighbor in self.graph.get(item, set()):
                graph_scores[neighbor] += 1.0 / recency
        max_graph = max(graph_scores.values(), default=1.0)
        add("bought_together", [(item, score / max_graph) for item, score in graph_scores.items()])
        add("trend", recommend_popular(self.popularity, seen, min(100, limit)))
        records: list[dict[str, Any]] = []
        for item, scores in channels.items():
            records.append(
                {
                    "parent_asin": item,
                    "retrieval_scores": scores,
                    "retrieval_score": max(scores.values()),
                    "recommendation_channel": "+".join(sorted(scores)),
                }
            )
        return sorted(
            records,
            key=lambda row: (-row["retrieval_score"], row["parent_asin"]),
        )[:limit]
