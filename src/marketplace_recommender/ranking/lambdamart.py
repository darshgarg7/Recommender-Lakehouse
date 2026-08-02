from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, cast

from marketplace_recommender.ranking.features import FEATURE_NAMES


@dataclass
class PairwiseLinearRanker:
    """Local learned ranker. The integration profile replaces this with XGBoost LambdaMART."""

    seed: int = 20250308
    weights: dict[str, float] = field(default_factory=dict)

    def fit(
        self,
        rows: Iterable[dict[str, object]],
        epochs: int = 80,
        learning_rate: float = 0.04,
        l2: float = 0.001,
    ) -> "PairwiseLinearRanker":
        values = list(rows)
        groups: dict[str, list[dict[str, object]]] = {}
        for row in values:
            groups.setdefault(str(row["group_id"]), []).append(row)
        pairs: list[tuple[dict[str, object], dict[str, object]]] = []
        for group_rows in groups.values():
            positives = [row for row in group_rows if int(cast(int, row["label"])) > 0]
            negatives = [row for row in group_rows if int(cast(int, row["label"])) <= 0]
            pairs.extend((positive, negative) for positive in positives for negative in negatives)
        self.weights = {name: 0.0 for name in FEATURE_NAMES}
        rng = random.Random(self.seed)
        for _ in range(epochs):
            rng.shuffle(pairs)
            for positive, negative in pairs:
                positive_features = cast(dict[str, float], positive["features"])
                negative_features = cast(dict[str, float], negative["features"])
                differences = {
                    name: float(positive_features.get(name, 0.0))
                    - float(negative_features.get(name, 0.0))
                    for name in FEATURE_NAMES
                }
                margin = sum(self.weights[name] * differences[name] for name in FEATURE_NAMES)
                gradient_scale = 1.0 / (1.0 + math.exp(min(30.0, max(-30.0, margin))))
                for name in FEATURE_NAMES:
                    self.weights[name] += learning_rate * (
                        gradient_scale * differences[name] - l2 * self.weights[name]
                    )
        if not pairs:
            self.weights.update(
                hybrid_ann_score=0.45,
                cointeraction_score=0.20,
                trend_score=0.10,
                user_item_similarity=0.25,
            )
        return self

    def predict(self, features: dict[str, float]) -> float:
        return sum(self.weights.get(name, 0.0) * features.get(name, 0.0) for name in FEATURE_NAMES)


class LambdaMARTRanker:
    """Optional XGBoost LambdaMART adapter for the Databricks integration tier."""

    def __init__(self, seed: int = 20250308) -> None:
        self.seed = seed
        self.model: Any | None = None

    def fit(self, rows: list[dict[str, object]]) -> "LambdaMARTRanker":
        try:
            import numpy as np
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("Install the 'ml' extra to use LambdaMARTRanker") from exc
        ordered = sorted(rows, key=lambda row: str(row["group_id"]))
        feature_rows = [
            [
                float(cast(dict[str, float], row["features"]).get(name, 0.0))
                for name in FEATURE_NAMES
            ]
            for row in ordered
        ]
        x = np.asarray(feature_rows)
        y = np.asarray([int(cast(int, row["label"])) for row in ordered])
        group_sizes: list[int] = []
        previous = None
        for row in ordered:
            group_id = str(row["group_id"])
            if group_id != previous:
                group_sizes.append(0)
                previous = group_id
            group_sizes[-1] += 1
        self.model = xgb.XGBRanker(
            objective="rank:ndcg",
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=self.seed,
            tree_method="hist",
        )
        self.model.fit(x, y, group=group_sizes)
        return self

    def predict(self, features: dict[str, float]) -> float:
        if self.model is None:
            raise RuntimeError("ranker has not been fitted")
        import numpy as np

        return float(
            self.model.predict(np.asarray([[features.get(name, 0.0) for name in FEATURE_NAMES]]))[0]
        )
