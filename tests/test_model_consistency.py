"""
Phase 6 - Step 5
Model Consistency Testing (Aligned with Production Schema)

Validates internal consistency, monotonicity, and numerical stability of
the survival prediction layer.
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
    """Execute a prediction and return JSON from mounted route."""
    response = client.post(ENDPOINT, json=customer)
    assert response.status_code == 200
    return response.json()


def extract_survival_values(data: dict) -> list[float]:
    """Extract survival probabilities whether curve is dict or list of points."""
    curve = data.get("forecast_curve") or data.get("survival_curve") or {}

    if isinstance(curve, dict) and "survival_prob" in curve:
        return [float(v) for v in curve["survival_prob"]]

    values = []
    if isinstance(curve, list):
        for point in curve:
            if isinstance(point, (int, float)):
                values.append(float(point))
            elif isinstance(point, dict):
                for key in [
                    "survival_probability",
                    "survival",
                    "retention_probability",
                    "value",
                ]:
                    if key in point:
                        values.append(float(point[key]))
                        break
    return values


def test_same_customer_produces_deterministic_prediction():
    """The same input should produce identical model output."""
    first = predict(BASE_CUSTOMER)
    second = predict(BASE_CUSTOMER)

    assert first["hazard_ratio"] == second["hazard_ratio"]
    assert first["projected_churn"] == second["projected_churn"]
    assert first["projected_retention"] == second["projected_retention"]
    assert first["risk_tier"] == second["risk_tier"]


def test_survival_curve_exists_and_is_nonempty():
    """Every valid prediction must contain a survival curve."""
    data = predict(BASE_CUSTOMER)
    curve = data.get("forecast_curve") or data.get("survival_curve")
    assert curve is not None
    values = extract_survival_values(data)
    assert len(values) > 0


def test_survival_probabilities_are_bounded():
    """Survival probabilities must remain within [0, 1] (or [0, 100])."""
    data = predict(BASE_CUSTOMER)
    values = extract_survival_values(data)
    assert len(values) > 0

    for value in values:
        assert 0 <= value <= 100.0


def test_survival_curve_is_monotonic_non_increasing():
    """
    Survival probability should not increase as time progresses.
    Small floating-point tolerance is allowed.
    """
    data = predict(BASE_CUSTOMER)
    values = extract_survival_values(data)
    assert len(values) > 0

    for previous, current in zip(values, values[1:]):
        assert current <= previous + 1e-5


def test_churn_and_retention_are_complementary():
    """Projected churn and retention should approximately sum to 100%."""
    data = predict(BASE_CUSTOMER)

    churn = float(data["projected_churn"])
    retention = float(data["projected_retention"])

    assert abs((churn + retention) - 100.0) <= 0.1


def test_target_month_is_current_tenure_plus_horizon():
    """Target month must equal tenure + forecast horizon."""
    data = predict(BASE_CUSTOMER)

    assert data["target_month"] == (
        data["current_tenure"] + data["forecast_horizon"]
    )


def test_hazard_ratio_is_positive():
    """Hazard multiplier must be a positive finite value."""
    data = predict(BASE_CUSTOMER)

    hazard = float(data["hazard_ratio"])
    assert hazard > 0
    assert hazard == hazard
    assert hazard != float("inf")


def test_risk_tier_matches_hazard_threshold():
    """Risk tier must agree with hazard ranges (CRITICAL, HIGH, MODERATE, or HEALTHY/LOW)."""
    data = predict(BASE_CUSTOMER)

    hazard = float(data["hazard_ratio"])
    risk_tier = data["risk_tier"].upper()

    if hazard >= 2.0:
        assert risk_tier in ["CRITICAL", "HIGH"]
    elif hazard >= 1.2:
        assert risk_tier in ["MODERATE", "MEDIUM"]
    else:
        assert risk_tier in ["HEALTHY", "LOW"]


def test_shorter_and_longer_forecast_horizons_have_correct_targets():
    """Different horizons must produce the correct target months."""
    short = deepcopy(BASE_CUSTOMER)
    short["forecast_horizon"] = 6

    long = deepcopy(BASE_CUSTOMER)
    long["forecast_horizon"] = 24

    short_result = predict(short)
    long_result = predict(long)

    assert short_result["target_month"] == 18
    assert long_result["target_month"] == 36


def test_prediction_remains_stable_for_extreme_valid_inputs():
    """A valid boundary configuration must still produce finite, bounded outputs."""
    customer = deepcopy(BASE_CUSTOMER)
    customer.update(
        {
            "tenure": 72,
            "forecast_horizon": 36,
            "MonthlyCharges": 250.0,
            "TotalCharges": 20000.0,
            "SeniorCitizen": 1,
        }
    )

    data = predict(customer)

    assert 0 <= data["projected_churn"] <= 100
    assert 0 <= data["projected_retention"] <= 100
    assert float(data["hazard_ratio"]) > 0

    values = extract_survival_values(data)
    assert len(values) > 0