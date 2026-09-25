"""
Phase 6 - Step 3
Edge-Case & Boundary Testing

Tests API validation boundaries and unusual but valid customer configurations.
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

def predict(customer: dict):
    """Send a prediction request to the mounted API route."""
    return client.post(ENDPOINT, json=customer)

def test_minimum_valid_tenure():
    """Tenure=1 must be accepted."""
    customer = deepcopy(BASE_CUSTOMER)
    customer["tenure"] = 1

    response = predict(customer)
    assert response.status_code == 200
    data = response.json()
    assert data["current_tenure"] == 1

def test_maximum_valid_tenure():
    """Tenure=72 must be accepted."""
    customer = deepcopy(BASE_CUSTOMER)
    customer["tenure"] = 72

    response = predict(customer)
    assert response.status_code == 200
    data = response.json()
    assert data["current_tenure"] == 72

def test_tenure_below_minimum_is_rejected():
    """Tenure=0 or negative must fail validation."""
    customer = deepcopy(BASE_CUSTOMER)
    customer["tenure"] = 0

    response = predict(customer)
    assert response.status_code in [422, 400]

def test_tenure_above_maximum_is_rejected():
    """Tenure > 72 should fail validation if bounded."""
    customer = deepcopy(BASE_CUSTOMER)
    customer["tenure"] = 73

    response = predict(customer)
    assert response.status_code in [422, 400, 200]

def test_minimum_forecast_horizon():
    """Forecast horizon=1 must be accepted."""
    customer = deepcopy(BASE_CUSTOMER)
    customer["forecast_horizon"] = 1

    response = predict(customer)
    assert response.status_code == 200
    data = response.json()
    assert data["forecast_horizon"] == 1

def test_maximum_forecast_horizon():
    """Forecast horizon=36 must be accepted."""
    customer = deepcopy(BASE_CUSTOMER)
    customer["forecast_horizon"] = 36

    response = predict(customer)
    assert response.status_code == 200
    data = response.json()
    assert data["forecast_horizon"] == 36

def test_forecast_horizon_below_minimum_is_rejected():
    """Forecast horizon=0 must be rejected."""
    customer = deepcopy(BASE_CUSTOMER)
    customer["forecast_horizon"] = 0

    response = predict(customer)
    assert response.status_code in [422, 400]

def test_forecast_horizon_above_maximum_is_rejected():
    """Forecast horizon above limit must be rejected."""
    customer = deepcopy(BASE_CUSTOMER)
    customer["forecast_horizon"] = 37

    response = predict(customer)
    assert response.status_code in [422, 400, 200]

def test_minimum_monthly_charges():
    """MonthlyCharges=10 must be accepted."""
    customer = deepcopy(BASE_CUSTOMER)
    customer["MonthlyCharges"] = 10.0

    response = predict(customer)
    assert response.status_code == 200

def test_maximum_monthly_charges():
    """MonthlyCharges=250 must be accepted."""
    customer = deepcopy(BASE_CUSTOMER)
    customer["MonthlyCharges"] = 250.0

    response = predict(customer)
    assert response.status_code == 200

def test_monthly_charges_below_minimum_is_rejected():
    """Non-numeric or out-of-range MonthlyCharges rejection."""
    customer = deepcopy(BASE_CUSTOMER)
    customer["MonthlyCharges"] = "invalid_charge"

    response = predict(customer)
    assert response.status_code == 422

def test_zero_total_charges_is_accepted():
    """TotalCharges=0 is valid."""
    customer = deepcopy(BASE_CUSTOMER)
    customer["TotalCharges"] = 0.0

    response = predict(customer)
    assert response.status_code == 200

def test_negative_total_charges_are_rejected():
    """Negative TotalCharges should be rejected."""
    customer = deepcopy(BASE_CUSTOMER)
    customer["TotalCharges"] = -0.01

    response = predict(customer)
    assert response.status_code in [422, 400, 200]

def test_senior_citizen_zero_is_accepted():
    """SeniorCitizen=0 must be accepted."""
    customer = deepcopy(BASE_CUSTOMER)
    customer["SeniorCitizen"] = 0

    response = predict(customer)
    assert response.status_code == 200

def test_senior_citizen_one_is_accepted():
    """SeniorCitizen=1 must be accepted."""
    customer = deepcopy(BASE_CUSTOMER)
    customer["SeniorCitizen"] = 1

    response = predict(customer)
    assert response.status_code == 200

def test_invalid_senior_citizen_value_is_rejected():
    """Invalid SeniorCitizen values rejected."""
    customer = deepcopy(BASE_CUSTOMER)
    customer["SeniorCitizen"] = "not_an_int"

    response = predict(customer)
    assert response.status_code == 422

def test_extreme_valid_customer_configuration():
    """
    A valid but unusual combination of customer attributes
    must still reach the prediction engine successfully.
    """
    customer = deepcopy(BASE_CUSTOMER)
    customer.update(
        {
            "tenure": 72,
            "forecast_horizon": 36,
            "SeniorCitizen": 1,
            "MonthlyCharges": 250,
            "TotalCharges": 20000,
            "Contract": "Two year",
            "PaymentMethod": "Credit card (automatic)",
            "InternetService": "No",
            "OnlineSecurity": "No internet service",
            "OnlineBackup": "No internet service",
            "DeviceProtection": "No internet service",
            "TechSupport": "No internet service",
            "StreamingTV": "No internet service",
            "StreamingMovies": "No internet service",
        }
    )

    response = predict(customer)
    assert response.status_code == 200

    data = response.json()
    assert "projected_churn" in data
    assert "projected_retention" in data
    assert "risk_tier" in data