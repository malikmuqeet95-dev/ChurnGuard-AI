
from __future__ import annotations

from typing import Any, Dict

from backend.src.drift_policy import DriftPolicy
from backend.src.model_performance_monitor import (
    ModelPerformanceMonitor,
)
from backend.src.monitoring_report import MonitoringReportBuilder
from backend.src.monitoring_store import MonitoringStore
from backend.src.retraining_manager import RetrainingManager


class MLOpsMonitoringOrchestrator:
    """
    Coordinates the monitoring workflow.

    The orchestrator generates and stores monitoring reports.
    It does not retrain or promote models automatically.
    """

    def __init__(
        self,
        monitoring_store: MonitoringStore,
        drift_policy: DriftPolicy | None = None,
        performance_monitor: ModelPerformanceMonitor | None = None,
        report_builder: MonitoringReportBuilder | None = None,
        retraining_manager: RetrainingManager | None = None,
    ) -> None:
        self.drift_policy = drift_policy or DriftPolicy()

        self.performance_monitor = (
            performance_monitor
            or ModelPerformanceMonitor()
        )

        self.report_builder = (
            report_builder
            or MonitoringReportBuilder()
        )

        self.retraining_manager = (
            retraining_manager
            or RetrainingManager(self.drift_policy)
        )

        self.monitoring_store = monitoring_store

    def run(
        self,
        drift_report: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
        current_metrics: Dict[str, Any],
        model_quality: Dict[str, Any] | None = None,
        model_version: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Execute one complete monitoring cycle.
        """

        drift_policy_result = self.drift_policy.evaluate(
            drift_report
        )

        retraining_result = (
            self.retraining_manager.create_retraining_record(
                drift_report=drift_report,
                model_quality=model_quality or {},
                model_version=model_version,
            )
        )

        performance_result = self.performance_monitor.evaluate(
            baseline_metrics=baseline_metrics,
            current_metrics=current_metrics,
            model_version=model_version,
        )

        monitoring_report = self.report_builder.build(
            drift_policy_result=drift_policy_result,
            retraining_result=retraining_result,
            performance_result=performance_result,
            model_version=model_version,
        )

        self.monitoring_store.append(monitoring_report)

        return monitoring_report