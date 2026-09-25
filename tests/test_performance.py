import time

from backend.app.performance import PerformanceTimer


def test_timer_records_stage():
    timer = PerformanceTimer()

    with timer.measure("test_stage"):
        time.sleep(0.001)

    result = timer.result()

    assert "test_stage" in result
    assert result["test_stage"] >= 0


def test_timer_records_multiple_stages():
    timer = PerformanceTimer()

    with timer.measure("stage_one"):
        pass

    with timer.measure("stage_two"):
        pass

    result = timer.result()

    assert "stage_one" in result
    assert "stage_two" in result


def test_total_time_is_non_negative():
    timer = PerformanceTimer()

    with timer.measure("stage"):
        pass

    assert timer.total_ms() >= 0


def test_result_returns_copy():
    timer = PerformanceTimer()

    with timer.measure("stage"):
        pass

    result = timer.result()

    result["new_stage"] = 999

    assert "new_stage" not in timer.result()