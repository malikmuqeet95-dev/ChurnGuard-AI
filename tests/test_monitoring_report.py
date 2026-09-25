
from backend.src.drift_policy import DriftPolicy
from backend.src.monitoring_report import MonitoringReportBuilder
from backend.src.retraining_manager import RetrainingManager
from backend.src.model_performance_monitor import (
    ModelPerformanceMonitor,
)


def _build_components():
    drift_policy = DriftPolicy()
    retraining_manager = RetrainingManager(drift_policy)
    performance_monitor = ModelPerformanceMonitor()

    return (
        drift_policy,
        retraining_manager,
        performance_monitor,
    )


def test_healthy_monitoring_report_requires_no_action():
    (
        drift_policy,
        retraining_manager,
        performance_monitor,
    ) = _build_components()

    drift_report = {
        "features": {
            "MonthlyCharges": {
                "psi": 0.02,
                "missing_rate": 0.0,
            }
        }
    }

    drift_result = drift_policy.evaluate(drift_report)

    retraining_result = retraining_manager.create_retraining_record(
        drift_report=drift_report,
        model_quality={"quality_passed": True},
        model_version="2.0.0",
    )

    performance_result = performance_monitor.evaluate(
        baseline_metrics={
            "c_index": 0.91,
            "brier_score": 0.12,
        },
        current_metrics={
            "c_index": 0.90,
            "brier_score": 0.13,
        },
        model_version="2.0.0",
    )

    result = MonitoringReportBuilder().build(
        drift_policy_result=drift_result,
        retraining_result=retraining_result,
        performance_result=performance_result,
        model_version="2.0.0",
    )

    assert result["overall_status"] == "HEALTHY"
    assert result["decision"] == "NO_ACTION"
    assert result["human_review_required"] is False
    assert result["retraining_requested"] is False


def test_warning_monitoring_report_recommends_retraining():
    (
        drift_policy,
        retraining_manager,
        performance_monitor,
    ) = _build_components()

    drift_report = {
        "features": {
            "MonthlyCharges": {
                "psi": 0.15,
                "missing_rate": 0.0,
            }
        }
    }

    drift_result = drift_policy.evaluate(drift_report)

    retraining_result = retraining_manager.create_retraining_record(
        drift_report=drift_report,
        model_quality={"quality_passed": True},
    )

    performance_result = performance_monitor.evaluate(
        baseline_metrics={
            "c_index": 0.91,
            "brier_score": 0.12,
        },
        current_metrics={
            "c_index": 0.90,
            "brier_score": 0.13,
        },
    )

    result = MonitoringReportBuilder().build(
        drift_policy_result=drift_result,
        retraining_result=retraining_result,
        performance_result=performance_result,
    )

    assert result["overall_status"] == "WARNING"
    assert result["decision"] == "RETRAINING_QUEUED"
    assert result["retraining_requested"] is True


def test_critical_performance_requires_human_review():
    (
        drift_policy,
        retraining_manager,
        performance_monitor,
    ) = _build_components()

    drift_report = {
        "features": {
            "MonthlyCharges": {
                "psi": 0.02,
                "missing_rate": 0.0,
            }
        }
    }

    drift_result = drift_policy.evaluate(drift_report)

    retraining_result = retraining_manager.create_retraining_record(
        drift_report=drift_report,
        model_quality={"quality_passed": True},
    )

    performance_result = performance_monitor.evaluate(
        baseline_metrics={
            "c_index": 0.91,
            "brier_score": 0.12,
        },
        current_metrics={
            "c_index": 0.55,
            "brier_score": 0.13,
        },
    )

    result = MonitoringReportBuilder().build(
        drift_policy_result=drift_result,
        retraining_result=retraining_result,
        performance_result=performance_result,
    )

    assert result["overall_status"] == "CRITICAL"
    assert result["decision"] == "HUMAN_REVIEW_REQUIRED"
    assert result["human_review_required"] is True


def test_report_contains_all_monitoring_sections():
    (
        drift_policy,
        retraining_manager,
        performance_monitor,
    ) = _build_components()

    drift_report = {"features": {}}

    drift_result = drift_policy.evaluate(drift_report)

    retraining_result = retraining_manager.create_retraining_record(
        drift_report=drift_report,
        model_quality={"quality_passed": True},
    )

    performance_result = performance_monitor.evaluate(
        baseline_metrics={
            "c_index": 0.91,
            "brier_score": 0.12,
        },
        current_metrics={
            "c_index": 0.91,
            "brier_score": 0.12,
        },
    )

    result = MonitoringReportBuilder().build(
        drift_policy_result=drift_result,
        retraining_result=retraining_result,
        performance_result=performance_result,
    )

    assert "drift_policy" in result["sections"]
    assert "retraining" in result["sections"]
    assert "performance" in result["sections"]
    assert result["report_id"].startswith("monitoring-")


def test_invalid_monitoring_section_requires_review():
    result = MonitoringReportBuilder().build(
        drift_policy_result=[],
        retraining_result={},
        performance_result={},
    )

    assert result["overall_status"] == "CRITICAL"
    assert result["decision"] == "HUMAN_REVIEW_REQUIRED"
    assert "drift_policy" in result["invalid_sections"]