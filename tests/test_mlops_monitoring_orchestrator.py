
from backend.src.mlops_monitoring_orchestrator import (
    MLOpsMonitoringOrchestrator,
)
from backend.src.monitoring_store import MonitoringStore


def _healthy_inputs():
    drift_report = {
        "features": {
            "MonthlyCharges": {
                "psi": 0.02,
                "missing_rate": 0.0,
            }
        }
    }

    baseline_metrics = {
        "c_index": 0.91,
        "brier_score": 0.12,
    }

    current_metrics = {
        "c_index": 0.90,
        "brier_score": 0.13,
    }

    model_quality = {
        "quality_passed": True,
    }

    return (
        drift_report,
        baseline_metrics,
        current_metrics,
        model_quality,
    )


def test_healthy_monitoring_cycle_is_stored(tmp_path):
    store = MonitoringStore(
        tmp_path / "monitoring.jsonl"
    )

    orchestrator = MLOpsMonitoringOrchestrator(
        monitoring_store=store
    )

    (
        drift_report,
        baseline_metrics,
        current_metrics,
        model_quality,
    ) = _healthy_inputs()

    result = orchestrator.run(
        drift_report=drift_report,
        baseline_metrics=baseline_metrics,
        current_metrics=current_metrics,
        model_quality=model_quality,
        model_version="2.0.0",
    )

    assert result["overall_status"] == "HEALTHY"
    assert result["decision"] == "NO_ACTION"
    assert result["model_version"] == "2.0.0"

    assert store.count() == 1
    assert store.latest()["report_id"] == result["report_id"]


def test_drift_cycle_requests_retraining(tmp_path):
    store = MonitoringStore(
        tmp_path / "monitoring.jsonl"
    )

    orchestrator = MLOpsMonitoringOrchestrator(
        monitoring_store=store
    )

    drift_report = {
        "features": {
            "MonthlyCharges": {
                "psi": 0.15,
                "missing_rate": 0.0,
            }
        }
    }

    result = orchestrator.run(
        drift_report=drift_report,
        baseline_metrics={
            "c_index": 0.91,
            "brier_score": 0.12,
        },
        current_metrics={
            "c_index": 0.90,
            "brier_score": 0.13,
        },
        model_quality={
            "quality_passed": True,
        },
        model_version="2.0.0",
    )

    assert result["overall_status"] == "WARNING"
    assert result["decision"] == "RETRAINING_QUEUED"
    assert result["retraining_requested"] is True
    assert store.count() == 1


def test_performance_failure_requires_human_review(tmp_path):
    store = MonitoringStore(
        tmp_path / "monitoring.jsonl"
    )

    orchestrator = MLOpsMonitoringOrchestrator(
        monitoring_store=store
    )

    (
        drift_report,
        baseline_metrics,
        _current_metrics,
        model_quality,
    ) = _healthy_inputs()

    result = orchestrator.run(
        drift_report=drift_report,
        baseline_metrics=baseline_metrics,
        current_metrics={
            "c_index": 0.55,
            "brier_score": 0.13,
        },
        model_quality=model_quality,
        model_version="2.0.0",
    )

    assert result["overall_status"] == "CRITICAL"
    assert result["decision"] == "HUMAN_REVIEW_REQUIRED"
    assert result["human_review_required"] is True
    assert store.count() == 1


def test_failed_model_quality_requires_human_review(tmp_path):
    store = MonitoringStore(tmp_path / "monitoring.jsonl")
    orchestrator = MLOpsMonitoringOrchestrator(monitoring_store=store)

    (
        drift_report,
        baseline_metrics,
        current_metrics,
        _model_quality,
    ) = _healthy_inputs()

    result = orchestrator.run(
        drift_report=drift_report,
        baseline_metrics=baseline_metrics,
        current_metrics=current_metrics,
        model_quality={"quality_passed": False},
        model_version="2.0.0",
    )

    # Correct assertion: failed quality must flag the overall report as CRITICAL
    assert result["overall_status"] == "CRITICAL"
    assert result["decision"] == "HUMAN_REVIEW_REQUIRED"
    assert result["human_review_required"] is True
    assert (
        result["sections"]["retraining"]["decision"]
        == "HUMAN_REVIEW_REQUIRED"
    )
    assert store.count() == 1


def test_multiple_monitoring_cycles_are_persisted(tmp_path):
    store = MonitoringStore(
        tmp_path / "monitoring.jsonl"
    )

    orchestrator = MLOpsMonitoringOrchestrator(
        monitoring_store=store
    )

    (
        drift_report,
        baseline_metrics,
        current_metrics,
        model_quality,
    ) = _healthy_inputs()

    orchestrator.run(
        drift_report=drift_report,
        baseline_metrics=baseline_metrics,
        current_metrics=current_metrics,
        model_quality=model_quality,
        model_version="2.0.0",
    )

    orchestrator.run(
        drift_report=drift_report,
        baseline_metrics=baseline_metrics,
        current_metrics=current_metrics,
        model_quality=model_quality,
        model_version="2.0.1",
    )

    assert store.count() == 2
    assert store.latest()["model_version"] == "2.0.1"