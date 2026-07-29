from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DiagonalDistiller:
    """Fits a diagonal content-to-collaborative projection by least squares."""

    scale: list[float] | None = None

    def fit(
        self,
        content: dict[str, list[float]],
        collaborative: dict[str, list[float]],
        warm_items: set[str],
    ) -> "DiagonalDistiller":
        common = sorted(warm_items & content.keys() & collaborative.keys())
        if not common:
            dimension = len(next(iter(content.values()))) if content else 48
            self.scale = [1.0] * dimension
            return self
        dimension = len(content[common[0]])
        scale = []
        for index in range(dimension):
            numerator = sum(content[item][index] * collaborative[item][index] for item in common)
            denominator = sum(content[item][index] ** 2 for item in common) + 1e-6
            scale.append(max(-3.0, min(3.0, numerator / denominator)))
        self.scale = scale
        return self

    def transform(self, vector: list[float]) -> list[float]:
        scale = self.scale or [1.0] * len(vector)
        return [value * factor for value, factor in zip(vector, scale)]
