
from backend.src.retraining_manager import RetrainingManager


def test_low_drift_does_not_request_retraining():
    manager = RetrainingManager()

    report = {
        "features": {
            "MonthlyCharges": {
                "psi": 0.02,
                "missing_rate": 0.0,
            },
        }
    }

    result = manager.create_retraining_record(
        drift_report=report,
        model_quality={"quality_passed": True},
        model_version="2.0.0",
    )

    assert result["decision"] == "NO_RETRAINING_REQUIRED"
    assert result["retraining_requested"] is False
    assert result["human_review_required"] is False
    assert result["model_version"] == "2.0.0"


def test_moderate_drift_queues_retraining():
    manager = RetrainingManager()

    report = {
        "features": {
            "MonthlyCharges": {
                "psi": 0.15,
                "missing_rate": 0.0,
            },
        }
    }

    result = manager.create_retraining_record(
        drift_report=report,
        model_quality={"quality_passed": True},
        model_version="2.0.0",
    )

    assert result["decision"] == "RETRAINING_QUEUED"
    assert result["retraining_requested"] is True
    assert result["human_review_required"] is False


def test_critical_drift_requires_human_review():
    manager = RetrainingManager()

    report = {
        "features": {
            "MonthlyCharges": {
                "psi": 0.40,
                "missing_rate": 0.0,
            },
        }
    }

    result = manager.create_retraining_record(
        drift_report=report,
        model_quality={"quality_passed": True},
    )

    assert result["decision"] == "HUMAN_REVIEW_REQUIRED"
    assert result["retraining_requested"] is False
    assert result["human_review_required"] is True


def test_failed_quality_check_requires_human_review():
    manager = RetrainingManager()

    report = {
        "features": {
            "MonthlyCharges": {
                "psi": 0.02,
                "missing_rate": 0.0,
            },
        }
    }

    result = manager.create_retraining_record(
        drift_report=report,
        model_quality={"quality_passed": False},
    )

    assert result["decision"] == "HUMAN_REVIEW_REQUIRED"
    assert result["retraining_requested"] is False
    assert result["human_review_required"] is True


def test_record_contains_traceability_fields():
    manager = RetrainingManager()

    result = manager.create_retraining_record(
        drift_report={"features": {}},
        model_quality={"quality_passed": True},
        model_version="2.0.0",
    )

    assert result["record_id"].startswith("retrain-")
    assert result["created_at_utc"]
    assert result["model_version"] == "2.0.0"
    assert result["drift_policy"]
    assert result["disclaimer"]