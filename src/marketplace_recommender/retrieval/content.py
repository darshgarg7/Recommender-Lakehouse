from __future__ import annotations

from typing import Any, Iterable

from marketplace_recommender.retrieval.vectors import dot, hashed_text_vector, mean, normalize


def build_content_embeddings(
    product_content: Iterable[dict[str, Any]], dimension: int = 48
) -> dict[str, list[float]]:
    return {
        row["parent_asin"]: hashed_text_vector(row.get("content_text", ""), dimension)
        for row in product_content
    }


def user_content_profile(history: list[str], embeddings: dict[str, list[float]]) -> list[float]:
    vectors = [embeddings[item] for item in history if item in embeddings]
    if not vectors:
        return [0.0] * (len(next(iter(embeddings.values()))) if embeddings else 48)
    weights = list(range(1, len(vectors) + 1))
    return normalize(mean(vectors, weights))


def content_candidates(
    user_vector: list[float],
    embeddings: dict[str, list[float]],
    seen: set[str],
    limit: int,
) -> list[tuple[str, float]]:
    return sorted(
        (
            (item, dot(user_vector, vector))
            for item, vector in embeddings.items()
            if item not in seen
        ),
        key=lambda pair: (-pair[1], pair[0]),
    )[:limit]
