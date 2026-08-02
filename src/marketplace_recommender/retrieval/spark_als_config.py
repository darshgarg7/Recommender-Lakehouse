from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SparkAlsBenchmarkConfig:
    """Configuration for the distributed temporal ALS benchmark."""

    rank: int = 64
    max_iter: int = 12
    reg_param: float = 0.08
    alpha: float = 20.0
    seed: int = 20250308
    validation_fraction: float = 0.10
    test_fraction: float = 0.10
    min_user_items: int = 2
    min_item_users: int = 2
    recommendation_k: int = 10
    candidate_k: int = 200
    evaluation_user_limit: int = 10_000
    rrf_constant: int = 60
    rrf_als_weights: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    bootstrap_samples: int = 10_000
    confidence_level: float = 0.95

    def validate(self) -> None:
        if self.rank <= 0 or self.max_iter <= 0:
            raise ValueError("rank and max_iter must be positive")
        if self.reg_param <= 0 or self.alpha <= 0:
            raise ValueError("reg_param and alpha must be positive")
        if self.validation_fraction <= 0 or self.test_fraction <= 0:
            raise ValueError("validation and test fractions must be positive")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation and test fractions must sum to less than one")
        if self.min_user_items < 2 or self.min_item_users < 2:
            raise ValueError("the collaborative core requires at least two users and items")
        if self.recommendation_k <= 0 or self.candidate_k < self.recommendation_k:
            raise ValueError("candidate_k must be at least recommendation_k")
        if self.evaluation_user_limit <= 0 or self.rrf_constant <= 0:
            raise ValueError("evaluation_user_limit and rrf_constant must be positive")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if not self.rrf_als_weights or any(
            not 0.0 <= value <= 1.0 for value in self.rrf_als_weights
        ):
            raise ValueError("RRF ALS weights must be in [0, 1]")


def benchmark_fingerprint(payload: dict[str, Any]) -> str:
    """Content-address a benchmark definition without run-time metadata."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def ndcg_for_single_relevant_rank(rank: int | None, k: int = 10) -> float:
    if rank is None or rank <= 0 or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)
