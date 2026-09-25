from backend.src.drift_policy import (
    DriftPolicy,
    DriftPolicyConfig,
)


def test_low_drift_requires_no_action():
    policy = DriftPolicy()

    report = {
        "features": {
            "MonthlyCharges": {
                "psi": 0.02,
                "missing_rate": 0.0,
            },
            "TotalCharges": {
                "psi": 0.03,
                "missing_rate": 0.01,
            },
        }
    }

    result = policy.evaluate(report)

    assert result["decision"] == "NO_ACTION"
    assert result["severity"] == "LOW"
    assert result["retraining_recommended"] is False


def test_moderate_drift_recommends_retraining():
    policy = DriftPolicy()

    report = {
        "features": {
            "MonthlyCharges": {
                "psi": 0.15,
                "missing_rate": 0.0,
            },
            "TotalCharges": {
                "psi": 0.02,
                "missing_rate": 0.0,
            },
        }
    }

    result = policy.evaluate(report)

    assert result["decision"] == "RETRAINING_RECOMMENDED"
    assert result["severity"] == "WARNING"
    assert result["retraining_recommended"] is True
    assert "MonthlyCharges" in result["drifted_features"]


def test_critical_drift_requires_human_review():
    policy = DriftPolicy()

    report = {
        "features": {
            "MonthlyCharges": {
                "psi": 0.40,
                "missing_rate": 0.0,
            },
        }
    }

    result = policy.evaluate(report)

    assert result["decision"] == "HUMAN_REVIEW_REQUIRED"
    assert result["severity"] == "CRITICAL"
    assert result["retraining_recommended"] is True
    assert "MonthlyCharges" in result["critical_features"]


def test_high_missing_rate_requires_human_review():
    policy = DriftPolicy()

    report = {
        "features": {
            "TotalCharges": {
                "psi": 0.02,
                "missing_rate": 0.25,
            },
        }
    }

    result = policy.evaluate(report)

    assert result["decision"] == "HUMAN_REVIEW_REQUIRED"
    assert "TotalCharges" in result["critical_features"]


def test_invalid_feature_structure_requires_review():
    policy = DriftPolicy()

    result = policy.evaluate(
        {
            "features": [],
        }
    )

    assert result["decision"] == "HUMAN_REVIEW_REQUIRED"
    assert result["severity"] == "CRITICAL"


def test_custom_policy_thresholds_are_applied():
    policy = DriftPolicy(
        DriftPolicyConfig(
            warning_psi=0.05,
            critical_psi=0.15,
        )
    )

    report = {
        "features": {
            "MonthlyCharges": {
                "psi": 0.08,
                "missing_rate": 0.0,
            },
        }
    }

    result = policy.evaluate(report)

    assert result["decision"] == "RETRAINING_RECOMMENDED"