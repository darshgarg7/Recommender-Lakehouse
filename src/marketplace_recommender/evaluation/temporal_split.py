from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class TemporalCutoffs:
    training_end: int
    validation_end: int
    test_end: int

    def split_for(self, timestamp: int) -> str:
        if timestamp < self.training_end:
            return "train"
        if timestamp < self.validation_end:
            return "validation"
        if timestamp <= self.test_end:
            return "test"
        return "outside"


def temporal_cutoffs(
    interactions: Iterable[dict[str, Any]],
    validation_fraction: float,
    test_fraction: float,
) -> TemporalCutoffs:
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must be positive and sum to less than one")
    timestamps = sorted({int(row["review_timestamp"]) for row in interactions})
    if len(timestamps) < 3:
        raise ValueError("at least three distinct timestamps are required")
    train_index = max(1, int(len(timestamps) * (1 - validation_fraction - test_fraction)))
    validation_index = max(train_index + 1, int(len(timestamps) * (1 - test_fraction)))
    validation_index = min(validation_index, len(timestamps) - 1)
    return TemporalCutoffs(
        training_end=timestamps[train_index],
        validation_end=timestamps[validation_index],
        test_end=timestamps[-1],
    )
