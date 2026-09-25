
from backend.src.model_performance_monitor import (
    ModelPerformanceMonitor,
    PerformanceMonitoringConfig,
)


def test_healthy_performance_requires_no_action():
    monitor = ModelPerformanceMonitor()

    result = monitor.evaluate(
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

    assert result["status"] == "HEALTHY"
    assert result["decision"] == "NO_ACTION"
    assert result["model_version"] == "2.0.0"


def test_c_index_degradation_triggers_warning():
    monitor = ModelPerformanceMonitor()

    result = monitor.evaluate(
        baseline_metrics={
            "c_index": 0.91,
            "brier_score": 0.12,
        },
        current_metrics={
            "c_index": 0.84,
            "brier_score": 0.13,
        },
    )

    assert result["status"] == "WARNING"
    assert result["decision"] == "PERFORMANCE_REVIEW_RECOMMENDED"
    assert result["metrics"]["c_index_drop"] == 0.07


def test_low_c_index_requires_human_review():
    monitor = ModelPerformanceMonitor()

    result = monitor.evaluate(
        baseline_metrics={
            "c_index": 0.91,
            "brier_score": 0.12,
        },
        current_metrics={
            "c_index": 0.55,
            "brier_score": 0.13,
        },
    )

    assert result["status"] == "CRITICAL"
    assert result["decision"] == "HUMAN_REVIEW_REQUIRED"


def test_brier_score_degradation_triggers_warning():
    monitor = ModelPerformanceMonitor()

    result = monitor.evaluate(
        baseline_metrics={
            "c_index": 0.91,
            "brier_score": 0.10,
        },
        current_metrics={
            "c_index": 0.90,
            "brier_score": 0.20,
        },
    )

    assert result["status"] == "WARNING"
    assert result["decision"] == "PERFORMANCE_REVIEW_RECOMMENDED"


def test_missing_metrics_require_human_review():
    monitor = ModelPerformanceMonitor()

    result = monitor.evaluate(
        baseline_metrics={
            "c_index": 0.91,
        },
        current_metrics={
            "c_index": 0.90,
        },
    )

    assert result["status"] == "CRITICAL"
    assert result["decision"] == "HUMAN_REVIEW_REQUIRED"


def test_custom_thresholds_are_applied():
    monitor = ModelPerformanceMonitor(
        PerformanceMonitoringConfig(
            c_index_drop_tolerance=0.01,
            max_brier_score_increase=0.01,
            minimum_c_index=0.70,
        )
    )

    result = monitor.evaluate(
        baseline_metrics={
            "c_index": 0.91,
            "brier_score": 0.10,
        },
        current_metrics={
            "c_index": 0.89,
            "brier_score": 0.11,
        },
    )

    assert result["status"] == "WARNING"