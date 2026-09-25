"""
Phase 6 - Step 10
Final End-to-End QA & Release Sign-Off (Aligned with Production Schema)

Validates that the complete API response contains required decision layers
and remains stable across customer profiles.
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
    """Execute prediction request against mounted endpoint."""
    response = client.post(ENDPOINT, json=customer)
    assert response.status_code == 200
    return response.json()


def test_final_response_contains_all_decision_layers():
    """The final API response exposes all layers required by dashboard."""
    data = predict(BASE_CUSTOMER)

    required_sections = [
        "current_tenure",
        "forecast_horizon",
        "target_month",
        "hazard_ratio",
        "projected_churn",
        "projected_retention",
        "risk_tier",
        "recommended_action",
    ]

    for section in required_sections:
        assert section in data

    assert "forecast_curve" in data or "survival_curve" in data
    assert "explanation" in data or "explainability" in data
    assert "retention_decision" in data or "decision" in data


def test_final_prediction_values_are_valid():
    """Core prediction values must remain within expected mathematical ranges."""
    data = predict(BASE_CUSTOMER)

    assert data["current_tenure"] >= 1
    assert data["forecast_horizon"] >= 1
    assert data["target_month"] == data["current_tenure"] + data["forecast_horizon"]
    assert float(data["hazard_ratio"]) > 0

    churn = float(data["projected_churn"])
    retention = float(data["projected_retention"])

    assert 0 <= churn <= 100
    assert 0 <= retention <= 100
    assert abs((churn + retention) - 100.0) <= 0.1


def test_final_explainability_layer_is_present():
    """Explainability provides structured risk and protective driver information."""
    data = predict(BASE_CUSTOMER)
    explanation = data.get("explanation") or data.get("explainability")

    assert isinstance(explanation, dict)
    assert "risk_drivers" in explanation
    assert "protective_drivers" in explanation


def test_final_intervention_layer_is_present():
    """Intervention evaluation is available inside the retention decision."""
    data = predict(BASE_CUSTOMER)
    decision = data.get("retention_decision") or data.get("decision")
    assert decision is not None

    interventions = decision.get("evaluated_interventions", [])
    assert isinstance(interventions, list)

    for intervention in interventions:
        assert "name" in intervention or "intervention_id" in intervention
        assert "priority" in intervention
        assert "roi" in intervention


def test_final_decision_layer_is_operational():
    """Retention decision provides an operational recommendation and status."""
    data = predict(BASE_CUSTOMER)
    decision = data.get("retention_decision") or data.get("decision")

    assert isinstance(decision, dict)
    assert "decision_status" in decision
    assert "risk_tier" in decision
    assert "projected_churn" in decision
    assert "operational_priority" in decision
    assert "recommendation" in decision
    assert "reason" in decision
    assert "evaluated_interventions" in decision
    assert "decision_disclaimer" in decision or "disclaimer" in decision


def test_high_risk_customer_produces_valid_decision():
    """A representative high-risk customer produces a valid decision response."""
    customer = deepcopy(BASE_CUSTOMER)
    customer.update(
        {
            "tenure": 1,
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "MonthlyCharges": 150.0,
            "TotalCharges": 150.0,
        }
    )

    data = predict(customer)
    decision = data.get("retention_decision") or data.get("decision")

    assert 0 <= float(data["projected_churn"]) <= 100
    assert data["risk_tier"] in {"CRITICAL", "HIGH", "MODERATE", "HEALTHY", "LOW"}
    assert decision["recommendation"] is not None


def test_stable_customer_produces_valid_decision():
    """A representative stable customer produces a valid decision response."""
    customer = deepcopy(BASE_CUSTOMER)
    customer.update(
        {
            "tenure": 60,
            "Contract": "Two year",
            "PaymentMethod": "Credit card (automatic)",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "MonthlyCharges": 55.0,
            "TotalCharges": 3300.0,
        }
    )

    data = predict(customer)
    decision = data.get("retention_decision") or data.get("decision")

    assert data["risk_tier"] in {"CRITICAL", "HIGH", "MODERATE", "HEALTHY", "LOW"}
    assert decision["decision_status"] is not None
    assert decision["recommendation"] is not None


def test_frontend_required_prediction_fields_remain_available():
    """Fields consumed by dashboard remain present after pipeline execution."""
    data = predict(BASE_CUSTOMER)

    frontend_fields = [
        "projected_churn",
        "projected_retention",
        "risk_tier",
        "recommended_action",
    ]

    for field in frontend_fields:
        assert field in data

    assert "forecast_curve" in data or "survival_curve" in data
    assert "explanation" in data or "explainability" in data
    assert "retention_decision" in data or "decision" in data


def test_final_api_health_endpoint_is_ready():
    """Health endpoint must return status 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_final_api_readiness_endpoint_is_operational():
    """Readiness check endpoint must respond."""
    response = client.get("/ready")
    assert response.status_code in {200, 503}
    assert isinstance(response.json(), dict)


def test_final_prediction_is_deterministic():
    """Final response core decision outputs are identical for same input."""
    first = predict(BASE_CUSTOMER)
    second = predict(BASE_CUSTOMER)

    assert (
        first["hazard_ratio"],
        first["projected_churn"],
        first["projected_retention"],
        first["risk_tier"],
    ) == (
        second["hazard_ratio"],
        second["projected_churn"],
        second["projected_retention"],
        second["risk_tier"],
    )


def test_final_pipeline_does_not_mutate_customer_payload():
    """The local input dictionary must remain unchanged after prediction."""
    customer = deepcopy(BASE_CUSTOMER)
    original = deepcopy(customer)
    predict(customer)
    assert customer == original