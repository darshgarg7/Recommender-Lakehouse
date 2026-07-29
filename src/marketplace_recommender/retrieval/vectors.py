from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def add(*vectors: list[float]) -> list[float]:
    if not vectors:
        return []
    return [sum(values) for values in zip(*vectors)]


def mean(vectors: Iterable[list[float]], weights: Iterable[float] | None = None) -> list[float]:
    values = list(vectors)
    if not values:
        return []
    chosen_weights = list(weights) if weights is not None else [1.0] * len(values)
    total = sum(chosen_weights)
    if total <= 0:
        return [0.0] * len(values[0])
    return [
        sum(vector[index] * weight for vector, weight in zip(values, chosen_weights)) / total
        for index in range(len(values[0]))
    ]


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def hashed_text_vector(text: str, dimension: int = 48) -> list[float]:
    vector = [0.0] * dimension
    counts = Counter(tokenize(text))
    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 else -1.0
        vector[bucket] += sign * (1.0 + math.log(count))
    return normalize(vector)


def hashed_id_vector(identifier: str, dimension: int = 48) -> list[float]:
    return hashed_text_vector(" ".join([identifier, f"id-{identifier}"]), dimension)
