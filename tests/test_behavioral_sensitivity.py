"""
Phase 6 - Step 4
Feature Sensitivity & Behavioral Testing (Aligned with Production Schema)

Verifies that important customer feature changes propagate through the survival
prediction and decision-intelligence pipeline.
"""
from __future__ import annotations

from copy import deepcopy
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)
ENDPOINT = "/api/predict"

BASE_CUSTOMER = {
    "tenure": 12,
    "forecast_horizon": 12,
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "MonthlyCharges": 79.85,
    "TotalCharges": 958.20,
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
}


def predict(customer: dict) -> dict:
    """Return prediction JSON from the mounted endpoint."""
    response = client.post(ENDPOINT, json=customer)
    assert response.status_code == 200
    return response.json()


def test_contract_change_affects_prediction():
    """Month-to-month versus Two year must alter the model output."""
    baseline = predict(BASE_CUSTOMER)

    changed = deepcopy(BASE_CUSTOMER)
    changed["Contract"] = "Two year"

    alternative = predict(changed)

    assert (
        baseline["hazard_ratio"] != alternative["hazard_ratio"]
        or baseline["projected_churn"] != alternative["projected_churn"]
        or baseline["risk_tier"] != alternative["risk_tier"]
        or baseline["recommended_action"] != alternative["recommended_action"]
    )


def test_payment_method_change_affects_prediction():
    """Electronic check versus automatic credit card must alter model outputs."""
    baseline = predict(BASE_CUSTOMER)

    changed = deepcopy(BASE_CUSTOMER)
    changed["PaymentMethod"] = "Credit card (automatic)"

    alternative = predict(changed)

    assert (
        baseline["hazard_ratio"] != alternative["hazard_ratio"]
        or baseline["projected_churn"] != alternative["projected_churn"]
        or baseline["risk_tier"] != alternative["risk_tier"]
        or baseline["recommended_action"] != alternative["recommended_action"]
    )


def test_tech_support_change_affects_prediction():
    """TechSupport changes should propagate to downstream model outputs."""
    baseline = predict(BASE_CUSTOMER)

    changed = deepcopy(BASE_CUSTOMER)
    changed["TechSupport"] = "Yes"

    alternative = predict(changed)

    assert (
        baseline["hazard_ratio"] != alternative["hazard_ratio"]
        or baseline["projected_churn"] != alternative["projected_churn"]
        or baseline["risk_tier"] != alternative["risk_tier"]
        or baseline["recommended_action"] != alternative["recommended_action"]
    )


def test_security_feature_change_affects_prediction():
    """OnlineSecurity changes should propagate through the system."""
    baseline = predict(BASE_CUSTOMER)

    changed = deepcopy(BASE_CUSTOMER)
    changed["OnlineSecurity"] = "Yes"

    alternative = predict(changed)

    assert (
        baseline["hazard_ratio"] != alternative["hazard_ratio"]
        or baseline["projected_churn"] != alternative["projected_churn"]
        or baseline["risk_tier"] != alternative["risk_tier"]
        or baseline["recommended_action"] != alternative["recommended_action"]
    )


def test_forecast_horizon_changes_target_month():
    """Changing forecast horizon must update target month."""
    baseline = predict(BASE_CUSTOMER)

    changed = deepcopy(BASE_CUSTOMER)
    changed["forecast_horizon"] = 6

    alternative = predict(changed)

    assert baseline["target_month"] != alternative["target_month"]
    assert baseline["target_month"] == 24
    assert alternative["target_month"] == 18


def test_tenure_change_is_reflected_in_prediction():
    """Changing current tenure must be reflected in the response."""
    baseline = predict(BASE_CUSTOMER)

    changed = deepcopy(BASE_CUSTOMER)
    changed["tenure"] = 24

    alternative = predict(changed)

    assert baseline["current_tenure"] == 12
    assert alternative["current_tenure"] == 24
    assert baseline["target_month"] != alternative["target_month"]


def test_multiple_feature_changes_propagate():
    """Multiple simultaneous customer changes produce a valid altered state."""
    baseline = predict(BASE_CUSTOMER)

    changed = deepcopy(BASE_CUSTOMER)
    changed.update(
        {
            "Contract": "Two year",
            "PaymentMethod": "Credit card (automatic)",
            "TechSupport": "Yes",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
        }
    )

    alternative = predict(changed)

    assert "retention_decision" in alternative or "decision" in alternative
    assert (
        baseline["hazard_ratio"] != alternative["hazard_ratio"]
        or baseline["projected_churn"] != alternative["projected_churn"]
        or baseline["risk_tier"] != alternative["risk_tier"]
        or baseline["recommended_action"] != alternative["recommended_action"]
    )


def test_prediction_contains_complete_downstream_pipeline():
    """A valid prediction contains all decision-intelligence layers."""
    data = predict(BASE_CUSTOMER)

    assert "forecast_curve" in data or "survival_curve" in data
    assert "explanation" in data or "explainability" in data

    decision = data.get("retention_decision") or data.get("decision")
    assert decision is not None
    assert "evaluated_interventions" in decision
    assert isinstance(decision["evaluated_interventions"], list)