from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def build_sequences(
    interactions: Iterable[dict[str, Any]],
    observation_times: Iterable[tuple[str, int]],
    max_length: int = 100,
) -> list[dict[str, Any]]:
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interactions:
        histories[row["user_id"]].append(row)
    for values in histories.values():
        values.sort(key=lambda row: (row["review_timestamp"], row["interaction_id"]))
    output = []
    for user_id, observation_time in sorted(set(observation_times)):
        eligible = [
            row for row in histories.get(user_id, []) if row["review_timestamp"] < observation_time
        ][-max_length:]
        output.append(
            {
                "user_id": user_id,
                "observation_timestamp": observation_time,
                "historical_parent_asins": [row["parent_asin"] for row in eligible],
                "historical_ratings": [row["rating"] for row in eligible],
                "historical_domains": [row["domain"] for row in eligible],
                "historical_event_times": [row["review_timestamp"] for row in eligible],
                "sequence_length": len(eligible),
            }
        )
    return output
