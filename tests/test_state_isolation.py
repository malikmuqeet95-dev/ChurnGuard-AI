"""
Phase 6 - Step 9
State Isolation & Leakage Testing (Aligned with Production Schema)

Verifies that predictions are isolated between requests and that customer-specific
state does not leak across requests or execution order.
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


def get_prediction(customer: dict) -> dict:
    """Return a prediction response for a customer payload."""
    response = client.post(ENDPOINT, json=customer)
    assert response.status_code == 200
    return response.json()


def core_prediction(data: dict) -> tuple:
    """Extract core outputs used for isolation comparisons."""
    return (
        data["hazard_ratio"],
        data["projected_churn"],
        data["projected_retention"],
        data["risk_tier"],
        data["target_month"],
    )


def test_same_customer_before_and_after_other_customer_is_identical():
    """Customer A output should be identical before and after evaluating Customer B."""
    customer_a = deepcopy(BASE_CUSTOMER)
    customer_b = deepcopy(BASE_CUSTOMER)
    customer_b.update(
        {
            "tenure": 72,
            "forecast_horizon": 36,
            "Contract": "Two year",
            "PaymentMethod": "Credit card (automatic)",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "MonthlyCharges": 120.0,
            "TotalCharges": 8000.0,
        }
    )

    first_a = get_prediction(customer_a)
    get_prediction(customer_b)
    second_a = get_prediction(customer_a)

    assert core_prediction(first_a) == core_prediction(second_a)


def test_request_order_does_not_change_customer_prediction():
    """Processing order should not alter individual prediction outputs."""
    customer_a = deepcopy(BASE_CUSTOMER)
    customer_b = deepcopy(BASE_CUSTOMER)
    customer_b.update(
        {
            "tenure": 3,
            "forecast_horizon": 6,
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "TechSupport": "No",
            "MonthlyCharges": 150.0,
            "TotalCharges": 450.0,
        }
    )

    a_then_b_a = get_prediction(customer_a)
    a_then_b_b = get_prediction(customer_b)

    b_then_a_b = get_prediction(customer_b)
    b_then_a_a = get_prediction(customer_a)

    assert core_prediction(a_then_b_a) == core_prediction(b_then_a_a)
    assert core_prediction(a_then_b_b) == core_prediction(b_then_a_b)


def test_multiple_customer_predictions_remain_independent():
    """Multiple distinct requests must remain independent without cross-contamination."""
    customers = []
    for tenure in [1, 6, 12, 24, 48, 72]:
        cust = deepcopy(BASE_CUSTOMER)
        cust["tenure"] = tenure
        customers.append(cust)

    predictions = [get_prediction(c) for c in customers]

    for cust, orig_pred in zip(customers, predictions):
        repeated_pred = get_prediction(cust)
        assert core_prediction(orig_pred) == core_prediction(repeated_pred)


def test_mutating_input_after_request_does_not_change_previous_result():
    """Mutating the local Python dictionary after call does not alter received response."""
    customer = deepcopy(BASE_CUSTOMER)
    result = get_prediction(customer)
    original_core = core_prediction(result)

    customer["tenure"] = 72
    customer["MonthlyCharges"] = 200.0
    customer["Contract"] = "Two year"

    assert core_prediction(result) == original_core


def test_repeated_identical_requests_remain_isolated():
    """Repeated identical requests must produce deterministic outputs."""
    customer = deepcopy(BASE_CUSTOMER)
    results = [core_prediction(get_prediction(customer)) for _ in range(10)]
    assert len(set(results)) == 1


def test_intervention_results_do_not_persist_between_requests():
    """Intervention calculations from a previous request must not leak into future ones."""
    baseline_customer = deepcopy(BASE_CUSTOMER)
    modified_customer = deepcopy(BASE_CUSTOMER)
    modified_customer.update(
        {
            "Contract": "Two year",
            "PaymentMethod": "Credit card (automatic)",
            "OnlineSecurity": "Yes",
            "TechSupport": "Yes",
            "MonthlyCharges": 50.0,
        }
    )

    baseline_before = get_prediction(baseline_customer)
    get_prediction(modified_customer)
    baseline_after = get_prediction(baseline_customer)

    assert core_prediction(baseline_before) == core_prediction(baseline_after)