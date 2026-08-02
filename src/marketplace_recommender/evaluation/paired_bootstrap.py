from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class PairedBootstrapEstimate:
    """Uncertainty for a candidate-minus-baseline mean on matched examples."""

    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    bootstrap_samples: int
    example_count: int
    probability_of_improvement: float
    two_sided_p_value: float
    method: str = "paired-user-percentile-bootstrap"

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = probability * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction


def paired_bootstrap_mean_difference(
    baseline: Iterable[float],
    candidate: Iterable[float],
    *,
    samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 20250308,
) -> PairedBootstrapEstimate:
    """Bootstrap matched user-level deltas without breaking the pairing.

    Recommendation metrics are highly skewed and the same users are scored by
    both policies. Resampling the paired deltas preserves that dependence and
    avoids the inflated variance of two independent bootstrap samples.
    """

    baseline_values = [float(value) for value in baseline]
    candidate_values = [float(value) for value in candidate]
    if len(baseline_values) != len(candidate_values):
        raise ValueError("baseline and candidate must contain the same matched examples")
    if not baseline_values:
        raise ValueError("paired bootstrap requires at least one matched example")
    if samples < 100:
        raise ValueError("paired bootstrap requires at least 100 samples")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")

    deltas = [right - left for left, right in zip(baseline_values, candidate_values, strict=True)]
    point_estimate = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    estimates = sorted(
        sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(samples)
    )
    alpha = (1.0 - confidence_level) / 2.0
    lower = _quantile(estimates, alpha)
    upper = _quantile(estimates, 1.0 - alpha)
    non_positive = sum(value <= 0.0 for value in estimates)
    non_negative = sum(value >= 0.0 for value in estimates)
    # Add-one correction prevents a misleading p=0 from a finite Monte Carlo run.
    probability = (sum(value > 0.0 for value in estimates) + 1) / (samples + 1)
    p_value = min(1.0, 2.0 * min(non_positive + 1, non_negative + 1) / (samples + 1))
    return PairedBootstrapEstimate(
        point_estimate=point_estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        bootstrap_samples=samples,
        example_count=len(deltas),
        probability_of_improvement=probability,
        two_sided_p_value=p_value,
    )
