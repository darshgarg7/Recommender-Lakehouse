from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def exact_temporal_cutoffs(
    frame: Any,
    column: str,
    quantiles: Sequence[float],
) -> tuple[int, ...]:
    """Return exact, table-layout-independent percentile cutoffs from Spark."""
    if not quantiles or any(not 0.0 <= value <= 1.0 for value in quantiles):
        raise ValueError("quantiles must be a nonempty sequence inside [0, 1]")
    from pyspark.sql import functions as F

    percentages = F.array(*(F.lit(float(value)) for value in quantiles))
    values = frame.agg(F.percentile(F.col(column), percentages).alias("cutoffs")).first()[0]
    if values is None or len(values) != len(quantiles):
        raise RuntimeError("exact temporal percentile calculation returned no cutoffs")
    return tuple(int(value) for value in values)
