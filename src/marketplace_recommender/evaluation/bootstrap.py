from __future__ import annotations

import random
from statistics import mean
from typing import Iterable


def bootstrap_mean_ci(
    values: Iterable[float], samples: int = 1_000, confidence: float = 0.95, seed: int = 20250308
) -> tuple[float, float, float]:
    observations = list(values)
    if not observations:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    estimates = sorted(mean(rng.choice(observations) for _ in observations) for _ in range(samples))
    alpha = (1.0 - confidence) / 2.0
    lower = estimates[int(alpha * (samples - 1))]
    upper = estimates[int((1.0 - alpha) * (samples - 1))]
    return mean(observations), lower, upper
