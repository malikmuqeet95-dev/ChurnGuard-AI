from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from time import monotonic
from typing import Any
from statistics import quantiles

LATENCY_HISTORY_LIMIT = 1000

@dataclass
class OperationalMetrics:
    """Thread-safe in-process operational metrics."""

    requests_total: int = 0
    requests_successful: int = 0
    requests_failed: int = 0
    predictions_total: int = 0
    what_if_total: int = 0

    prediction_successful: int = 0
    prediction_failed: int = 0
    what_if_successful: int = 0
    what_if_failed: int = 0

    http_2xx_total: int = 0
    http_3xx_total: int = 0
    http_4xx_total: int = 0
    http_5xx_total: int = 0

    latency_total_ms: float = 0.0
    latency_samples: int = 0

    latency_samples_ms: deque[float] = field(
    default_factory=lambda: deque(
        maxlen=LATENCY_HISTORY_LIMIT,
    ),
    init=False,
    repr=False,
    )

    _started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
        init=False,
        repr=False,
    )
    _started_monotonic: float = field(
        default_factory=monotonic,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._lock = Lock()

    def record_request(
        self,
        *,
        latency_ms: float,
        status_code: int,
    ) -> None:
        """Record a completed HTTP request."""

        with self._lock:
            self.requests_total += 1

            if status_code >= 400:
                self.requests_failed += 1
            else:
                self.requests_successful += 1

            if 200 <= status_code < 300:
                self.http_2xx_total += 1
            elif 300 <= status_code < 400:
                self.http_3xx_total += 1
            elif 400 <= status_code < 500:
                self.http_4xx_total += 1
            elif 500 <= status_code < 600:
                self.http_5xx_total += 1

            latency_value = max(float(latency_ms), 0.0)

            self.latency_total_ms += latency_value
            self.latency_samples += 1
            self.latency_samples_ms.append(latency_value)

    def record_prediction(self, count: int = 1) -> None:
        with self._lock:
            self.predictions_total += int(count)

    def record_what_if(self) -> None:
        """Record a what-if simulation request."""

        with self._lock:
            self.what_if_total += 1

    def record_prediction_result(
        self,
        *,
        success: bool,
    ) -> None:
        """Record the outcome of a prediction request."""

        with self._lock:
            if success:
                self.prediction_successful += 1
            else:
                self.prediction_failed += 1


    def record_what_if_result(
        self,
        *,
        success: bool,
    ) -> None:
        """Record the outcome of a what-if request."""

        with self._lock:
            if success:
                self.what_if_successful += 1
            else:
                self.what_if_failed += 1

    def snapshot(self) -> dict[str, Any]:
        """Return a consistent point-in-time metrics snapshot."""

        with self._lock:
            average_latency_ms = (
                self.latency_total_ms / self.latency_samples
                if self.latency_samples
                else 0.0
            )

            uptime_seconds = max(
                monotonic() - self._started_monotonic,
                0.0,
            )

            success_rate_pct = (
                (self.requests_successful / self.requests_total) * 100
                if self.requests_total
                else 0.0
            )

            error_rate_pct = (
                (self.requests_failed / self.requests_total) * 100
                if self.requests_total
                else 0.0
            )

            requests_per_second = (
                self.requests_total / uptime_seconds
                if uptime_seconds > 0
                else 0.0
            )

            request_status_total = (
                self.http_2xx_total
                + self.http_3xx_total
                + self.http_4xx_total
                + self.http_5xx_total
            )

            prediction_outcome_total = (
                self.prediction_successful
                + self.prediction_failed
            )

            what_if_outcome_total = (
                self.what_if_successful
                + self.what_if_failed
            )

            if self.latency_samples_ms:
                if len(self.latency_samples_ms) >= 2:
                    percentile_values = quantiles(
                        self.latency_samples_ms,
                        n=100,
                    )

                    p95_latency_ms = percentile_values[94]
                    p99_latency_ms = percentile_values[98]
                else:
                    p95_latency_ms = self.latency_samples_ms[0]
                    p99_latency_ms = self.latency_samples_ms[0]
            else:
                p95_latency_ms = 0.0
                p99_latency_ms = 0.0
            
            return {
                "started_at": self._started_at.isoformat(),
                "uptime_seconds": round(uptime_seconds, 2),
                "requests_total": self.requests_total,
                "requests_successful": self.requests_successful,
                "requests_failed": self.requests_failed,

                "http_2xx_total": self.http_2xx_total,
                "http_3xx_total": self.http_3xx_total,
                "http_4xx_total": self.http_4xx_total,
                "http_5xx_total": self.http_5xx_total,

                "success_rate_pct": round(success_rate_pct, 2),
                "error_rate_pct": round(error_rate_pct, 2),
                "requests_per_second": round(requests_per_second, 2),
                "predictions_total": self.predictions_total,
                "what_if_total": self.what_if_total,
                "prediction_successful": self.prediction_successful,
                "prediction_failed": self.prediction_failed,
                
                "what_if_successful": self.what_if_successful,
                "what_if_failed": self.what_if_failed,   
                "request_status_total": request_status_total,
                "prediction_outcome_total": prediction_outcome_total,
                "what_if_outcome_total": what_if_outcome_total,
                "average_latency_ms": round(
                    average_latency_ms,
                    2,
                ),
                 "p95_latency_ms": round(
                    p95_latency_ms,
                    2,
                ),
                "p99_latency_ms": round(
                    p99_latency_ms,
                    2,
                ),
            }

    def reset(self) -> None:
        """Reset counters.

        Intended primarily for tests and local development.
        """

        with self._lock:
            self.requests_total = 0
            self.requests_successful = 0
            self.requests_failed = 0
            self.predictions_total = 0
            self.what_if_total = 0

            self.http_2xx_total = 0
            self.http_3xx_total = 0
            self.http_4xx_total = 0
            self.http_5xx_total = 0

            
            self.prediction_successful = 0
            self.prediction_failed = 0
            self.what_if_successful = 0
            self.what_if_failed = 0

            self.latency_total_ms = 0.0
            self.latency_samples = 0
            self.latency_samples_ms.clear()


operational_metrics = OperationalMetrics()