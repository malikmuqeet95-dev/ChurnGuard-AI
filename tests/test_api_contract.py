"""
Phase 6 - Step 2
API Contract Tests (Aligned with Production Backend Schema)
"""
from __future__ import annotations
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)
ENDPOINT = "/api/predict"

VALID_CUSTOMER = {
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

def test_predict_returns_http_200():
    """A valid prediction request must return HTTP 200."""
    response = client.post(ENDPOINT, json=VALID_CUSTOMER)
    assert response.status_code == 200

def test_predict_returns_json():
    """The prediction endpoint must return JSON."""
    response = client.post(ENDPOINT, json=VALID_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)

def test_prediction_contains_core_fields():
    """Verify the core prediction contract matches backend output."""
    response = client.post(ENDPOINT, json=VALID_CUSTOMER)
    assert response.status_code == 200
    data = response.json()

    required_fields = [
        "current_tenure",
        "forecast_horizon",
        "target_month",
        "hazard_ratio",
        "projected_churn",
        "projected_retention",
        "risk_tier",
        "recommended_action",
    ]

    for field in required_fields:
        assert field in data, f"Missing required response field: {field}"

def test_prediction_core_field_types():
    """Verify the data types of core prediction fields."""
    response = client.post(ENDPOINT, json=VALID_CUSTOMER)
    assert response.status_code == 200
    data = response.json()

    assert isinstance(data["current_tenure"], int)
    assert isinstance(data["forecast_horizon"], int)
    assert isinstance(data["target_month"], (int, float))
    assert isinstance(data["hazard_ratio"], (int, float))
    assert isinstance(data["projected_churn"], (int, float))
    assert isinstance(data["projected_retention"], (int, float))
    assert isinstance(data["risk_tier"], str)
    assert isinstance(data["recommended_action"], str)

def test_prediction_probability_bounds():
    """Churn and retention percentages must remain within 0-100."""
    response = client.post(ENDPOINT, json=VALID_CUSTOMER)
    assert response.status_code == 200
    data = response.json()

    churn = data["projected_churn"]
    retention = data["projected_retention"]

    assert 0 <= churn <= 100
    assert 0 <= retention <= 100

def test_prediction_tenure_contract():
    """Verify tenure and forecast horizon are preserved."""
    response = client.post(ENDPOINT, json=VALID_CUSTOMER)
    assert response.status_code == 200
    data = response.json()

    assert data["current_tenure"] == 12
    assert data["forecast_horizon"] == 12
    assert data["target_month"] == 24

def test_prediction_contains_decision_intelligence():
    """Verify that the prediction response contains decision intelligence."""
    response = client.post(ENDPOINT, json=VALID_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert "retention_decision" in data or "decision" in data

def test_prediction_contains_explainability():
    """Verify explanation drivers are included in the response."""
    response = client.post(ENDPOINT, json=VALID_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert "explanation" in data or "explainability" in data

def test_prediction_contains_forecast_curve():
    """Verify survival curve timeline data exists."""
    response = client.post(ENDPOINT, json=VALID_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert "forecast_curve" in data or "survival_curve" in data

def test_invalid_data_type_returns_422():
    """FastAPI validation should reject non-numeric string for numeric field."""
    invalid_customer = VALID_CUSTOMER.copy()
    invalid_customer["MonthlyCharges"] = "invalid_string_not_a_float"
    response = client.post(ENDPOINT, json=invalid_customer)
    assert response.status_code == 422

def test_extra_field_handling():
    """Verify extra payload fields are rejected or parsed safely."""
    invalid_customer = VALID_CUSTOMER.copy()
    invalid_customer["unexpected_field"] = "invalid"
    response = client.post(ENDPOINT, json=invalid_customer)
    assert response.status_code in [422, 200]

def test_invalid_tenure_type_returns_422():
    """FastAPI validation should reject non-integer tenure."""
    invalid_customer = VALID_CUSTOMER.copy()
    invalid_customer["tenure"] = "ten_months"
    response = client.post(ENDPOINT, json=invalid_customer)
    assert response.status_code == 422
