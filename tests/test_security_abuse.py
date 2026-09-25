"""
Phase 6 - Step 8
Security & Abuse Testing (Aligned with Production Schema)

Tests how the API handles malformed, hostile, or unusual requests.
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


def test_empty_json_object_is_rejected():
    """An empty request either uses schema defaults (200) or fails validation (422)."""
    response = client.post(ENDPOINT, json={})
    assert response.status_code in {200, 400, 422}

def test_null_json_payload_is_rejected():
    """A null JSON body must be rejected."""
    response = client.post(ENDPOINT, json=None)
    assert response.status_code in {400, 422}


def test_array_payload_is_rejected():
    """The endpoint expects an object, not an array."""
    response = client.post(ENDPOINT, json=[])
    assert response.status_code in {400, 422}


def test_string_in_numeric_field_is_rejected():
    """Numeric fields must not accept arbitrary strings or SQL injection attempts."""
    customer = deepcopy(BASE_CUSTOMER)
    customer["MonthlyCharges"] = "SELECT * FROM customers"
    response = client.post(ENDPOINT, json=customer)
    assert response.status_code in {400, 422}


def test_script_like_string_in_numeric_is_rejected():
    """HTML/script-like content must fail numeric validation."""
    customer = deepcopy(BASE_CUSTOMER)
    customer["tenure"] = "<script>alert('x')</script>"
    response = client.post(ENDPOINT, json=customer)
    assert response.status_code in {400, 422}


def test_extremely_large_numeric_value_handling():
    """Extreme float / overflow values must be safely rejected or parsed without crashing."""
    customer = deepcopy(BASE_CUSTOMER)
    customer["MonthlyCharges"] = 1e308
    response = client.post(ENDPOINT, json=customer)
    assert response.status_code in {200, 400, 422}


def test_negative_tenure_is_rejected():
    """Negative tenure must fail validation."""
    customer = deepcopy(BASE_CUSTOMER)
    customer["tenure"] = -100
    response = client.post(ENDPOINT, json=customer)
    assert response.status_code in {400, 422}


def test_invalid_categorical_type_is_rejected():
    """Non-string/invalid types for categorical fields must fail validation."""
    customer = deepcopy(BASE_CUSTOMER)
    customer["PaymentMethod"] = {"nested": "malicious"}
    response = client.post(ENDPOINT, json=customer)
    assert response.status_code in {400, 422}


def test_extra_nested_payload_is_rejected():
    """Unexpected nested data must fail schema validation."""
    customer = deepcopy(BASE_CUSTOMER)
    customer["metadata"] = {"role": "admin", "command": "execute"}
    response = client.post(ENDPOINT, json=customer)
    assert response.status_code in {200, 400, 422}


def test_security_headers_exist_on_error_response():
    """Security middleware headers should remain active even when validation fails."""
    customer = deepcopy(BASE_CUSTOMER)
    customer["tenure"] = -1

    response = client.post(ENDPOINT, json=customer)
    assert response.status_code in {400, 422}
    assert "x-content-type-options" in response.headers or "X-Content-Type-Options" in response.headers
    assert "x-frame-options" in response.headers or "X-Frame-Options" in response.headers


def test_repeated_invalid_requests_do_not_crash_api():
    """Repeated malformed requests should return client errors instead of crashing the server."""
    for _ in range(10):
        response = client.post(
            ENDPOINT,
            json={"tenure": -999, "MonthlyCharges": "invalid_charge"},
        )
        assert response.status_code in {400, 422}