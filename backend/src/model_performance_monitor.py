
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict


@dataclass(frozen=True)
class PerformanceMonitoringConfig:
    """
    Configuration for post-deployment performance monitoring.

    c_index_drop_tolerance:
        Maximum permitted absolute C-index decrease.

    max_brier_score_increase:
        Maximum permitted absolute increase in Brier score.

    minimum_c_index:
        Minimum acceptable C-index.
    """

    c_index_drop_tolerance: float = 0.05
    max_brier_score_increase: float = 0.05
    minimum_c_index: float = 0.60


class ModelPerformanceMonitor:
    """
    Compares current model performance with baseline metrics.

    This component only creates a monitoring assessment.
    It does not retrain, promote, or replace a model.
    """

    def __init__(
        self,
        config: PerformanceMonitoringConfig | None = None,
    ) -> None:
        self.config = config or PerformanceMonitoringConfig()

    @staticmethod
    def _safe_float(
        value: Any,
        default: float | None = None,
    ) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _utc_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def evaluate(
        self,
        baseline_metrics: Dict[str, Any],
        current_metrics: Dict[str, Any],
        model_version: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Compare baseline and current model metrics.

        Expected metric structure:

        {
            "c_index": 0.91,
            "brier_score": 0.12
        }
        """

        if not isinstance(baseline_metrics, dict):
            return self._invalid_result(
                model_version=model_version,
                reason="Baseline metrics must be a dictionary.",
            )

        if not isinstance(current_metrics, dict):
            return self._invalid_result(
                model_version=model_version,
                reason="Current metrics must be a dictionary.",
            )

        baseline_c_index = self._safe_float(
            baseline_metrics.get("c_index")
        )
        current_c_index = self._safe_float(
            current_metrics.get("c_index")
        )

        baseline_brier = self._safe_float(
            baseline_metrics.get("brier_score")
        )
        current_brier = self._safe_float(
            current_metrics.get("brier_score")
        )

        missing_metrics = []

        if baseline_c_index is None or current_c_index is None:
            missing_metrics.append("c_index")

        if baseline_brier is None or current_brier is None:
            missing_metrics.append("brier_score")

        if missing_metrics:
            return self._invalid_result(
                model_version=model_version,
                reason=(
                    "Required performance metrics are missing or invalid: "
                    + ", ".join(sorted(set(missing_metrics)))
                ),
            )

        c_index_drop = baseline_c_index - current_c_index
        brier_increase = current_brier - baseline_brier

        c_index_below_minimum = (
            current_c_index < self.config.minimum_c_index
        )

        c_index_degraded = (
            c_index_drop > self.config.c_index_drop_tolerance
        )

        brier_degraded = (
            brier_increase > self.config.max_brier_score_increase
        )

        critical = c_index_below_minimum
        warning = c_index_degraded or brier_degraded

        if critical:
            status = "CRITICAL"
            decision = "HUMAN_REVIEW_REQUIRED"
            reason = (
                "Current C-index is below the configured minimum."
            )

        elif warning:
            status = "WARNING"
            decision = "PERFORMANCE_REVIEW_RECOMMENDED"
            reason = (
                "One or more performance metrics exceeded the "
                "configured degradation tolerance."
            )

        else:
            status = "HEALTHY"
            decision = "NO_ACTION"
            reason = (
                "Current performance remains within configured "
                "monitoring tolerances."
            )

        return {
            "monitoring_id": (
                f"performance-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            ),
            "created_at_utc": self._utc_timestamp(),
            "model_version": model_version,
            "status": status,
            "decision": decision,
            "reason": reason,
            "metrics": {
                "baseline_c_index": baseline_c_index,
                "current_c_index": current_c_index,
                "c_index_drop": round(c_index_drop, 6),
                "baseline_brier_score": baseline_brier,
                "current_brier_score": current_brier,
                "brier_score_increase": round(brier_increase, 6),
            },
            "thresholds": asdict(self.config),
            "disclaimer": (
                "Performance monitoring identifies metric changes. "
                "It does not establish the cause of degradation "
                "or guarantee future model performance."
            ),
        }

    def _invalid_result(
        self,
        model_version: str,
        reason: str,
    ) -> Dict[str, Any]:
        return {
            "monitoring_id": (
                f"performance-invalid-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            ),
            "created_at_utc": self._utc_timestamp(),
            "model_version": model_version,
            "status": "CRITICAL",
            "decision": "HUMAN_REVIEW_REQUIRED",
            "reason": reason,
            "metrics": {},
            "thresholds": asdict(self.config),
            "disclaimer": (
                "Invalid or incomplete monitoring data requires "
                "human review."
            ),
        }