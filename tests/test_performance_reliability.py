"""
Phase 6 - Step 7
Performance & Reliability Testing (Aligned with Production Schema)

Establishes a practical local performance baseline for the ChurnGuard prediction API.
"""
from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor
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


def predict():
    """Execute one valid prediction against mounted endpoint."""
    return client.post(ENDPOINT, json=BASE_CUSTOMER)


def test_single_prediction_completes_within_reasonable_time():
    """A single prediction should complete within a practical local threshold."""
    start = time.perf_counter()
    response = predict()
    elapsed = time.perf_counter() - start

    assert response.status_code == 200
    assert elapsed < 10.0


def test_repeated_predictions_remain_successful():
    """Repeated prediction calls should remain successful."""
    for _ in range(10):
        response = predict()
        assert response.status_code == 200
        data = response.json()
        assert "projected_churn" in data
        assert "projected_retention" in data
        assert "risk_tier" in data


def test_repeated_predictions_remain_deterministic():
    """Repeated identical requests should produce the same core prediction outputs."""
    results = []

    for _ in range(5):
        response = predict()
        assert response.status_code == 200
        data = response.json()
        results.append(
            (
                data["hazard_ratio"],
                data["projected_churn"],
                data["projected_retention"],
                data["risk_tier"],
            )
        )

    assert len(set(results)) == 1


def test_prediction_latency_stays_below_regression_threshold():
    """Measure several predictions and ensure average latency stays below threshold."""
    timings = []

    for _ in range(5):
        start = time.perf_counter()
        response = predict()
        elapsed = time.perf_counter() - start
        assert response.status_code == 200
        timings.append(elapsed)

    average_latency = statistics.mean(timings)
    assert average_latency < 10.0


def test_concurrent_predictions_are_successful():
    """A small concurrent request burst should be handled successfully."""
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(predict) for _ in range(5)]
        responses = [future.result() for future in futures]

    assert len(responses) == 5

    for response in responses:
        assert response.status_code == 200
        data = response.json()
        assert "projected_churn" in data
        assert "retention_decision" in data or "decision" in data


def test_concurrent_predictions_are_consistent():
    """Concurrent identical requests should not corrupt shared prediction state."""
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(predict) for _ in range(5)]
        responses = [future.result() for future in futures]

    results = []

    for response in responses:
        assert response.status_code == 200
        data = response.json()
        results.append(
            (
                data["hazard_ratio"],
                data["projected_churn"],
                data["projected_retention"],
                data["risk_tier"],
            )
        )

    assert len(set(results)) == 1