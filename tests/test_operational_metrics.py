from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.operational_metrics import OperationalMetrics


def test_metrics_start_at_zero():
    metrics = OperationalMetrics()

    snapshot = metrics.snapshot()

    assert snapshot["requests_total"] == 0
    assert snapshot["requests_successful"] == 0
    assert snapshot["requests_failed"] == 0
    assert snapshot["predictions_total"] == 0
    assert snapshot["what_if_total"] == 0
    assert snapshot["average_latency_ms"] == 0.0

    assert "started_at" in snapshot
    assert snapshot["uptime_seconds"] >= 0.0


def test_uptime_is_non_negative():
    metrics = OperationalMetrics()

    snapshot = metrics.snapshot()

    assert snapshot["uptime_seconds"] >= 0.0
    assert isinstance(snapshot["started_at"], str)


def test_request_metrics_are_recorded():
    metrics = OperationalMetrics()

    metrics.record_request(
        latency_ms=100.0,
        status_code=200,
    )

    metrics.record_request(
        latency_ms=200.0,
        status_code=500,
    )

    snapshot = metrics.snapshot()

    assert snapshot["requests_total"] == 2
    assert snapshot["requests_successful"] == 1
    assert snapshot["requests_failed"] == 1
    assert snapshot["average_latency_ms"] == 150.0


def test_request_status_counts_match_total():
    metrics = OperationalMetrics()

    for _ in range(3):
        metrics.record_request(
            latency_ms=10.0,
            status_code=200,
        )

    for _ in range(2):
        metrics.record_request(
            latency_ms=20.0,
            status_code=500,
        )

    snapshot = metrics.snapshot()

    assert snapshot["requests_total"] == 5
    assert snapshot["requests_successful"] == 3
    assert snapshot["requests_failed"] == 2

    assert (
        snapshot["requests_successful"]
        + snapshot["requests_failed"]
        == snapshot["requests_total"]
    )


def test_prediction_counter():
    metrics = OperationalMetrics()

    metrics.record_prediction()
    metrics.record_prediction()

    assert metrics.snapshot()["predictions_total"] == 2


def test_what_if_counter():
    metrics = OperationalMetrics()

    metrics.record_what_if()

    assert metrics.snapshot()["what_if_total"] == 1


def test_metrics_reset():
    metrics = OperationalMetrics()

    metrics.record_prediction()
    metrics.record_what_if()
    metrics.record_request(
        latency_ms=50.0,
        status_code=200,
    )

    metrics.reset()

    snapshot = metrics.snapshot()

    assert snapshot["requests_total"] == 0
    assert snapshot["requests_successful"] == 0
    assert snapshot["requests_failed"] == 0
    assert snapshot["predictions_total"] == 0
    assert snapshot["what_if_total"] == 0
    assert snapshot["average_latency_ms"] == 0.0
    assert snapshot["p95_latency_ms"] == 0.0
    assert snapshot["p99_latency_ms"] == 0.0


def test_latency_percentiles_are_recorded():
    metrics = OperationalMetrics()

    for latency in range(1, 101):
        metrics.record_request(
            latency_ms=float(latency),
            status_code=200,
        )

    snapshot = metrics.snapshot()

    assert snapshot["requests_total"] == 100
    assert snapshot["requests_successful"] == 100
    assert snapshot["requests_failed"] == 0
    assert snapshot["average_latency_ms"] == 50.5
    assert 94.0 <= snapshot["p95_latency_ms"] <= 96.0
    assert 98.0 <= snapshot["p99_latency_ms"] <= 100.0


def test_latency_percentiles_start_at_zero():
    metrics = OperationalMetrics()

    snapshot = metrics.snapshot()

    assert snapshot["p95_latency_ms"] == 0.0
    assert snapshot["p99_latency_ms"] == 0.0


def test_single_latency_sample_is_used_for_percentiles():
    metrics = OperationalMetrics()

    metrics.record_request(
        latency_ms=42.0,
        status_code=200,
    )

    snapshot = metrics.snapshot()

    assert snapshot["p95_latency_ms"] == 42.0
    assert snapshot["p99_latency_ms"] == 42.0


def test_metrics_endpoint_shape():
    app = FastAPI()

    metrics = OperationalMetrics()

    @app.get("/metrics")
    def metrics_endpoint():
        return {
            "status": "ok",
            "environment": "test",
            "metrics": metrics.snapshot(),
        }

    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["environment"] == "test"
    assert "metrics" in data

    assert "requests_total" in data["metrics"]
    assert "requests_successful" in data["metrics"]
    assert "requests_failed" in data["metrics"]
    assert "predictions_total" in data["metrics"]
    assert "what_if_total" in data["metrics"]
    assert "average_latency_ms" in data["metrics"]
    assert "p95_latency_ms" in data["metrics"]
    assert "p99_latency_ms" in data["metrics"]

def test_request_rates_start_at_zero():
    metrics = OperationalMetrics()

    snapshot = metrics.snapshot()

    assert snapshot["success_rate_pct"] == 0.0
    assert snapshot["error_rate_pct"] == 0.0
    assert snapshot["requests_per_second"] >= 0.0


def test_request_rates_are_calculated():
    metrics = OperationalMetrics()

    for _ in range(8):
        metrics.record_request(
            latency_ms=10.0,
            status_code=200,
        )

    for _ in range(2):
        metrics.record_request(
            latency_ms=20.0,
            status_code=500,
        )

    snapshot = metrics.snapshot()

    assert snapshot["requests_total"] == 10
    assert snapshot["requests_successful"] == 8
    assert snapshot["requests_failed"] == 2

    assert snapshot["success_rate_pct"] == 80.0
    assert snapshot["error_rate_pct"] == 20.0
    assert snapshot["requests_per_second"] >= 0.0

def test_latency_history_is_bounded():
    metrics = OperationalMetrics()

    for latency in range(1, 1501):
        metrics.record_request(
            latency_ms=float(latency),
            status_code=200,
        )

    snapshot = metrics.snapshot()

    assert snapshot["requests_total"] == 1500
    assert snapshot["requests_successful"] == 1500
    assert snapshot["requests_failed"] == 0

    assert len(metrics.latency_samples_ms) == 1000

    assert min(metrics.latency_samples_ms) == 501.0
    assert max(metrics.latency_samples_ms) == 1500.0

def test_latency_history_is_cleared_on_reset():
    metrics = OperationalMetrics()

    for latency in range(1, 11):
        metrics.record_request(
            latency_ms=float(latency),
            status_code=200,
        )

    assert len(metrics.latency_samples_ms) == 10

    metrics.reset()

    assert len(metrics.latency_samples_ms) == 0

def test_http_status_class_metrics_are_recorded():
    metrics = OperationalMetrics()

    metrics.record_request(
        latency_ms=10.0,
        status_code=200,
    )

    metrics.record_request(
        latency_ms=10.0,
        status_code=201,
    )

    metrics.record_request(
        latency_ms=10.0,
        status_code=302,
    )

    metrics.record_request(
        latency_ms=10.0,
        status_code=400,
    )

    metrics.record_request(
        latency_ms=10.0,
        status_code=404,
    )

    metrics.record_request(
        latency_ms=10.0,
        status_code=500,
    )

    metrics.record_request(
        latency_ms=10.0,
        status_code=503,
    )

    snapshot = metrics.snapshot()

    assert snapshot["requests_total"] == 7

    assert snapshot["http_2xx_total"] == 2
    assert snapshot["http_3xx_total"] == 1
    assert snapshot["http_4xx_total"] == 2
    assert snapshot["http_5xx_total"] == 2

def test_http_status_class_metrics_reset():
    metrics = OperationalMetrics()

    metrics.record_request(
        latency_ms=10.0,
        status_code=200,
    )

    metrics.record_request(
        latency_ms=10.0,
        status_code=400,
    )

    metrics.record_request(
        latency_ms=10.0,
        status_code=500,
    )

    metrics.reset()

    snapshot = metrics.snapshot()

    assert snapshot["http_2xx_total"] == 0
    assert snapshot["http_3xx_total"] == 0
    assert snapshot["http_4xx_total"] == 0
    assert snapshot["http_5xx_total"] == 0

def test_prediction_outcome_metrics():
    metrics = OperationalMetrics()

    metrics.record_prediction()
    metrics.record_prediction_result(success=True)

    metrics.record_prediction()
    metrics.record_prediction_result(success=False)

    snapshot = metrics.snapshot()

    assert snapshot["predictions_total"] == 2
    assert snapshot["prediction_successful"] == 1
    assert snapshot["prediction_failed"] == 1

def test_what_if_outcome_metrics():
    metrics = OperationalMetrics()

    metrics.record_what_if()
    metrics.record_what_if_result(success=True)

    metrics.record_what_if()
    metrics.record_what_if_result(success=False)

    snapshot = metrics.snapshot()

    assert snapshot["what_if_total"] == 2
    assert snapshot["what_if_successful"] == 1
    assert snapshot["what_if_failed"] == 1

def test_endpoint_outcome_metrics_reset():
    metrics = OperationalMetrics()

    metrics.record_prediction()
    metrics.record_prediction_result(success=True)

    metrics.record_what_if()
    metrics.record_what_if_result(success=False)

    metrics.reset()

    snapshot = metrics.snapshot()

    assert snapshot["prediction_successful"] == 0
    assert snapshot["prediction_failed"] == 0
    assert snapshot["what_if_successful"] == 0
    assert snapshot["what_if_failed"] == 0

def test_metrics_consistency_totals():
    metrics = OperationalMetrics()

    metrics.record_request(
        latency_ms=10.0,
        status_code=200,
    )

    metrics.record_request(
        latency_ms=10.0,
        status_code=201,
    )

    metrics.record_request(
        latency_ms=10.0,
        status_code=400,
    )

    metrics.record_request(
        latency_ms=10.0,
        status_code=500,
    )

    metrics.record_prediction()
    metrics.record_prediction_result(success=True)

    metrics.record_prediction()
    metrics.record_prediction_result(success=False)

    metrics.record_what_if()
    metrics.record_what_if_result(success=True)

    snapshot = metrics.snapshot()

    assert snapshot["requests_total"] == 4

    assert (
        snapshot["request_status_total"]
        == snapshot["requests_total"]
    )

    assert (
        snapshot["prediction_outcome_total"]
        == snapshot["predictions_total"]
    )

    assert (
        snapshot["what_if_outcome_total"]
        == snapshot["what_if_total"]
    )

def test_metrics_consistency_totals_start_at_zero():
    metrics = OperationalMetrics()

    snapshot = metrics.snapshot()

    assert snapshot["request_status_total"] == 0
    assert snapshot["prediction_outcome_total"] == 0
    assert snapshot["what_if_outcome_total"] == 0

