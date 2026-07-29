from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class MetricLog:
    values: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def timed(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.values[f"{name}_seconds"] = time.perf_counter() - started

    def record(self, name: str, value: float) -> None:
        self.values[name] = float(value)
