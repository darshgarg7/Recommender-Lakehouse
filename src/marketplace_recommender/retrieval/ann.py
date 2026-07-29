from __future__ import annotations

from dataclasses import dataclass

from marketplace_recommender.retrieval.vectors import dot


@dataclass
class ExactANNIndex:
    """Exact local index used to measure and validate a production ANN implementation."""

    vectors: dict[str, list[float]]
    domains: dict[str, str]

    def query(
        self,
        vector: list[float],
        limit: int,
        *,
        excluded: set[str] | None = None,
        domain: str | None = None,
    ) -> list[tuple[str, float]]:
        excluded = excluded or set()
        candidates = (
            (item, dot(vector, item_vector))
            for item, item_vector in self.vectors.items()
            if item not in excluded and (domain is None or self.domains.get(item) == domain)
        )
        return sorted(candidates, key=lambda pair: (-pair[1], pair[0]))[:limit]

    def synchronize(self, changed: dict[str, list[float]]) -> int:
        self.vectors.update(changed)
        return len(changed)
