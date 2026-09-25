from concurrent.futures import ThreadPoolExecutor

from backend.app.operational_metrics import OperationalMetrics


def test_concurrent_prediction_updates_are_safe():
    metrics = OperationalMetrics()

    def record_prediction():
        metrics.record_prediction()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(record_prediction)
            for _ in range(100)
        ]

        for future in futures:
            future.result()

    snapshot = metrics.snapshot()

    assert snapshot["predictions_total"] == 100


def test_concurrent_what_if_updates_are_safe():
    metrics = OperationalMetrics()

    def record_what_if():
        metrics.record_what_if()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(record_what_if)
            for _ in range(100)
        ]

        for future in futures:
            future.result()

    snapshot = metrics.snapshot()

    assert snapshot["what_if_total"] == 100


def test_concurrent_request_updates_are_safe():
    metrics = OperationalMetrics()

    def record_request():
        metrics.record_request(
            latency_ms=10.0,
            status_code=200,
        )

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(record_request)
            for _ in range(100)
        ]

        for future in futures:
            future.result()

    snapshot = metrics.snapshot()

    assert snapshot["requests_total"] == 100
    assert snapshot["requests_failed"] == 0
    assert snapshot["average_latency_ms"] == 10.0


def test_concurrent_success_and_failure_requests_are_safe():
    metrics = OperationalMetrics()

    def record_success():
        metrics.record_request(
            latency_ms=10.0,
            status_code=200,
        )

    def record_failure():
        metrics.record_request(
            latency_ms=20.0,
            status_code=500,
        )

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []

        for _ in range(50):
            futures.append(executor.submit(record_success))
            futures.append(executor.submit(record_failure))

        for future in futures:
            future.result()

    snapshot = metrics.snapshot()

    assert snapshot["requests_total"] == 100
    assert snapshot["requests_failed"] == 50
    assert snapshot["average_latency_ms"] == 15.0