from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.operational_metrics import operational_metrics


client = TestClient(app)


def setup_function():
    operational_metrics.reset()


def test_metrics_endpoint_returns_200():
    response = client.get("/metrics")

    assert response.status_code == 200


def test_metrics_endpoint_returns_expected_structure():
    response = client.get("/metrics")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert "environment" in data
    assert "metrics" in data

    metrics = data["metrics"]

    assert "requests_total" in metrics
    assert "requests_successful" in metrics
    assert "requests_failed" in metrics
    assert "http_2xx_total" in metrics
    assert "http_3xx_total" in metrics
    assert "http_4xx_total" in metrics
    assert "http_5xx_total" in metrics
    assert "predictions_total" in metrics
    assert "what_if_total" in metrics

    assert "average_latency_ms" in metrics
    assert "p95_latency_ms" in metrics
    assert "p99_latency_ms" in metrics
    assert "success_rate_pct" in metrics
    assert "error_rate_pct" in metrics
    assert "requests_per_second" in metrics
    assert "prediction_successful" in metrics
    assert "prediction_failed" in metrics
    assert "what_if_successful" in metrics
    assert "what_if_failed" in metrics

    assert "request_status_total" in metrics
    assert "prediction_outcome_total" in metrics
    assert "what_if_outcome_total" in metrics


def test_metrics_endpoint_does_not_expose_sensitive_fields():
    response = client.get("/metrics")

    data = response.json()

    serialized = str(data).lower()

    assert "api_key" not in serialized
    assert "authorization" not in serialized
    assert "customerpayload" not in serialized


def test_metrics_endpoint_is_not_customer_data():
    response = client.get("/metrics")

    data = response.json()

    metrics = data["metrics"]

    assert isinstance(metrics["requests_total"], int)
    assert isinstance(metrics["requests_successful"], int)
    assert isinstance(metrics["requests_failed"], int)
    assert isinstance(metrics["predictions_total"], int)
    assert isinstance(metrics["what_if_total"], int)
    assert isinstance(metrics["average_latency_ms"], (int, float))
    assert isinstance(metrics["p95_latency_ms"], (int, float))
    assert isinstance(metrics["p99_latency_ms"], (int, float))

