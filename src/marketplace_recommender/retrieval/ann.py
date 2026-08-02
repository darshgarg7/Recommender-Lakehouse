from __future__ import annotations

from dataclasses import dataclass, field

from marketplace_recommender.retrieval.vectors import dot, normalize


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


@dataclass
class SubspacePartitionedANNIndex:
    """Dependency-light local ANN approximation used for contract and recall tests.

    Production serving uses managed Databricks AI Search. This deterministic
    centroid partitioner keeps local tests fast without making a latency claim.
    """

    vectors: dict[str, list[float]]
    domains: dict[str, str]
    num_buckets: int = 16
    n_probes: int = 4
    min_exact_threshold: int = 128
    centroids: list[list[float]] = field(default_factory=list, init=False)
    buckets: dict[int, list[str]] = field(default_factory=dict, init=False)
    query_stats: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._build_index()

    def _build_index(self) -> None:
        self.query_stats = {"total_queries": 0, "evaluations": 0}
        if len(self.vectors) < self.min_exact_threshold:
            self.centroids = []
            self.buckets = {0: list(self.vectors.keys())}
            return

        items = list(self.vectors.items())
        dimension = len(items[0][1]) if items else 0
        if dimension == 0:
            return

        actual_buckets = min(self.num_buckets, len(items))
        step = max(1, len(items) // actual_buckets)
        self.centroids = [normalize(items[i * step][1]) for i in range(actual_buckets)]
        self.buckets = {b: [] for b in range(actual_buckets)}

        for item_id, vec in items:
            best_bucket = max(
                range(len(self.centroids)),
                key=lambda b: dot(vec, self.centroids[b]),
            )
            self.buckets[best_bucket].append(item_id)

    def query(
        self,
        vector: list[float],
        limit: int,
        *,
        excluded: set[str] | None = None,
        domain: str | None = None,
        n_probes: int | None = None,
    ) -> list[tuple[str, float]]:
        excluded = excluded or set()
        probes = n_probes or self.n_probes
        self.query_stats["total_queries"] += 1

        if not self.centroids or len(self.vectors) < self.min_exact_threshold:
            candidate_ids = list(self.vectors.keys())
        else:
            centroid_scores = [
                (idx, dot(vector, c_vec)) for idx, c_vec in enumerate(self.centroids)
            ]
            centroid_scores.sort(key=lambda x: -x[1])
            top_buckets = [idx for idx, _ in centroid_scores[:probes]]
            candidate_ids = [
                item_id for b_idx in top_buckets for item_id in self.buckets.get(b_idx, [])
            ]

        self.query_stats["evaluations"] += len(candidate_ids)

        candidates = (
            (item, dot(vector, self.vectors[item]))
            for item in candidate_ids
            if item not in excluded and (domain is None or self.domains.get(item) == domain)
        )
        return sorted(candidates, key=lambda pair: (-pair[1], pair[0]))[:limit]

    def synchronize(self, changed: dict[str, list[float]]) -> int:
        self.vectors.update(changed)
        self._build_index()
        return len(changed)


class ANNIndexFactory:
    """Factory to instantiate the appropriate ANN Index based on scale and operational mode."""

    @staticmethod
    def create(
        vectors: dict[str, list[float]],
        domains: dict[str, str],
        *,
        scalable: bool = True,
        num_buckets: int = 16,
    ) -> ExactANNIndex | SubspacePartitionedANNIndex:
        if scalable and len(vectors) >= 64:
            return SubspacePartitionedANNIndex(
                vectors=vectors,
                domains=domains,
                num_buckets=num_buckets,
            )
        return ExactANNIndex(vectors=vectors, domains=domains)
