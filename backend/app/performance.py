from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class PerformanceTimer:
    """
    Lightweight timing utility for measuring application stages.
    """

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        start = time.perf_counter()

        try:
            yield
        finally:
            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            self.timings[name] = round(
                elapsed_ms,
                3,
            )

    def total_ms(self) -> float:
        return round(
            sum(self.timings.values()),
            3,
        )

    def result(self) -> dict[str, float]:
        return dict(self.timings)