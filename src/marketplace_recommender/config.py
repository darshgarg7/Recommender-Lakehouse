from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return {}
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith("["):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return [part.strip().strip("\"'") for part in value[1:-1].split(",") if part.strip()]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value.strip("\"'")


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Load the deliberately simple, dependency-free project configuration."""
    result: dict[str, Any] = {}
    parents: list[tuple[int, dict[str, Any]]] = [(-1, result)]
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        key, value = raw_line.strip().split(":", 1)
        while parents[-1][0] >= indent:
            parents.pop()
        target = parents[-1][1]
        parsed = _scalar(value)
        target[key] = parsed
        if parsed == {}:
            parents.append((indent, parsed))
    return result


@dataclass(frozen=True)
class PipelineConfig:
    tier: str
    seed: int
    output_dir: Path
    domains: tuple[str, ...]
    interaction_count: int
    sequence_max_length: int
    candidate_limit: int
    recommendation_limit: int
    validation_fraction: float
    test_fraction: float
    negative_count: int
    rerank: dict[str, Any]
    promotion: dict[str, Any]
    model_user_limit: int | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "PipelineConfig":
        raw = load_simple_yaml(path)
        return cls(
            tier=str(raw["tier"]),
            seed=int(raw["seed"]),
            output_dir=Path(raw["output_dir"]),
            domains=tuple(raw["domains"]),
            interaction_count=int(raw.get("interaction_count", 1_000_000)),
            sequence_max_length=int(raw["sequence_max_length"]),
            candidate_limit=int(raw["candidate_limit"]),
            recommendation_limit=int(raw["recommendation_limit"]),
            validation_fraction=float(raw["validation_fraction"]),
            test_fraction=float(raw["test_fraction"]),
            negative_count=int(raw["negative_count"]),
            rerank=dict(raw["rerank"]),
            promotion=dict(raw.get("promotion", {})),
            model_user_limit=(
                int(raw["model_user_limit"]) if raw.get("model_user_limit") is not None else None
            ),
        )
