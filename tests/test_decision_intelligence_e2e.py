"""
Phase 6 - Step 6
Decision Intelligence End-to-End Integration Tests (Aligned with Production Schema)

Verifies that a prediction successfully propagates through:
Prediction -> Explainability -> Interventions -> ROI -> Operational Priority -> Retention Decision
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
    """Execute prediction and return response JSON from mounted route."""
    response = client.post(ENDPOINT, json=customer)
    assert response.status_code == 200
    return response.json()


def get_decision(data: dict) -> dict:
    """Support the project's current decision field naming."""
    decision = data.get("retention_decision") or data.get("decision")
    assert decision is not None
    return decision


def test_prediction_contains_complete_decision_pipeline():
    """Verify that all major downstream layers are present."""
    data = predict(BASE_CUSTOMER)

    assert "explanation" in data or "explainability" in data

    decision = get_decision(data)
    assert "decision_status" in decision
    assert "risk_tier" in decision
    assert "projected_churn" in decision
    assert "operational_priority" in decision
    assert "recommendation" in decision
    assert "reason" in decision
    assert "evaluated_interventions" in decision


def test_decision_risk_matches_prediction_risk():
    """Decision engine must use the same risk tier as prediction."""
    data = predict(BASE_CUSTOMER)
    decision = get_decision(data)
    assert decision["risk_tier"] == data["risk_tier"]


def test_decision_churn_matches_prediction():
    """Decision projected churn should correspond to the prediction result."""
    data = predict(BASE_CUSTOMER)
    decision = get_decision(data)

    prediction_churn = float(data["projected_churn"])
    decision_churn = float(decision["projected_churn"])

    # Handle decimal vs percentage scale difference safely
    if prediction_churn <= 1.0 and decision_churn > 1.0:
        prediction_churn *= 100
    elif decision_churn <= 1.0 and prediction_churn > 1.0:
        decision_churn *= 100

    assert abs(prediction_churn - decision_churn) <= 0.1


def test_interventions_have_required_operational_fields():
    """Every evaluated intervention contains required operational attributes."""
    data = predict(BASE_CUSTOMER)
    decision = get_decision(data)
    interventions = decision["evaluated_interventions"]

    assert isinstance(interventions, list)

    for intervention in interventions:
        assert intervention.get("intervention_id") or intervention.get("name")
        assert "category" in intervention or "channel" in intervention
        assert "reason" in intervention
        assert "eligibility" in intervention
        assert "priority" in intervention
        assert "recommended_action" in intervention


def test_intervention_ranking_is_present():
    """Interventions should have valid ranking information."""
    data = predict(BASE_CUSTOMER)
    decision = get_decision(data)
    interventions = decision["evaluated_interventions"]

    for intervention in interventions:
        rank = intervention.get("decision_rank") or intervention.get("rank")
        assert rank is not None
        assert int(rank) >= 1


def test_intervention_roi_is_structurally_valid():
    """When ROI is present, economic projection fields must be valid."""
    data = predict(BASE_CUSTOMER)
    decision = get_decision(data)
    interventions = decision["evaluated_interventions"]

    for intervention in interventions:
        roi = intervention.get("roi")
        if roi is None:
            continue

        assert float(roi.get("estimated_cost", 0)) >= 0
        assert isinstance(float(roi.get("modeled_churn_improvement_pp", 0)), float)
        assert isinstance(float(roi.get("estimated_net_value", 0)), float)
        assert isinstance(float(roi.get("estimated_roi_pct", 0)), float)


def test_selected_intervention_matches_evaluated_interventions():
    """If an intervention is selected, it must exist among evaluated candidates."""
    data = predict(BASE_CUSTOMER)
    decision = get_decision(data)
    selected = decision.get("selected_intervention")

    if selected is None:
        return

    selected_id = selected.get("intervention_id") or selected.get("name")
    evaluated_ids = {
        item.get("intervention_id") or item.get("name")
        for item in decision["evaluated_interventions"]
    }

    assert selected_id in evaluated_ids


def test_decision_contains_non_causal_disclaimer():
    """The system must preserve its decision-support disclaimer."""
    data = predict(BASE_CUSTOMER)
    decision = get_decision(data)
    disclaimer = decision.get("decision_disclaimer") or data.get("disclaimer", "")

    assert isinstance(disclaimer, str)
    assert len(disclaimer.strip()) > 0


def test_high_risk_customer_reaches_decision_engine():
    """High-risk configuration must produce a complete decision result."""
    customer = deepcopy(BASE_CUSTOMER)
    customer.update(
        {
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "TechSupport": "No",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
        }
    )

    data = predict(customer)
    decision = get_decision(data)

    assert decision["decision_status"]
    assert decision["operational_priority"]
    assert decision["recommendation"]
    assert decision["reason"]


def test_lower_risk_customer_reaches_decision_engine():
    """Low-risk configuration must also reach decision layer."""
    customer = deepcopy(BASE_CUSTOMER)
    customer.update(
        {
            "Contract": "Two year",
            "PaymentMethod": "Credit card (automatic)",
            "TechSupport": "Yes",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
        }
    )

    data = predict(customer)
    decision = get_decision(data)

    assert decision["decision_status"]
    assert decision["operational_priority"]
    assert decision["recommendation"]
    assert decision["reason"]


def test_decision_changes_when_customer_profile_changes():
    """Different customer profiles should produce different decision outputs."""
    baseline = predict(BASE_CUSTOMER)

    changed = deepcopy(BASE_CUSTOMER)
    changed.update(
        {
            "Contract": "Two year",
            "PaymentMethod": "Credit card (automatic)",
            "TechSupport": "Yes",
            "OnlineSecurity": "Yes",
        }
    )

    alternative = predict(changed)

    baseline_decision = get_decision(baseline)
    alternative_decision = get_decision(alternative)

    assert (
        baseline_decision["decision_status"] != alternative_decision["decision_status"]
        or baseline_decision["operational_priority"] != alternative_decision["operational_priority"]
        or baseline_decision["recommendation"] != alternative_decision["recommendation"]
        or baseline["projected_churn"] != alternative["projected_churn"]
    )