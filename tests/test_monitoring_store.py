
import json

import pytest

from backend.src.monitoring_store import MonitoringStore


def test_missing_storage_file_returns_empty_history(tmp_path):
    store = MonitoringStore(tmp_path / "monitoring.jsonl")

    assert store.read_all() == []
    assert store.count() == 0
    assert store.latest() is None


def test_append_and_read_monitoring_report(tmp_path):
    store = MonitoringStore(tmp_path / "monitoring.jsonl")

    report = {
        "report_id": "monitoring-001",
        "overall_status": "HEALTHY",
        "decision": "NO_ACTION",
    }

    stored_report = store.append(report)

    assert stored_report == report
    assert store.read_all() == [report]
    assert store.count() == 1
    assert store.latest() == report


def test_multiple_reports_preserve_order(tmp_path):
    store = MonitoringStore(tmp_path / "monitoring.jsonl")

    first_report = {
        "report_id": "monitoring-001",
        "overall_status": "HEALTHY",
    }

    second_report = {
        "report_id": "monitoring-002",
        "overall_status": "WARNING",
    }

    store.append(first_report)
    store.append(second_report)

    assert store.read_all() == [
        first_report,
        second_report,
    ]

    assert store.latest() == second_report
    assert store.count() == 2


def test_nested_report_data_is_preserved(tmp_path):
    store = MonitoringStore(tmp_path / "monitoring.jsonl")

    report = {
        "report_id": "monitoring-003",
        "sections": {
            "performance": {
                "status": "HEALTHY",
                "metrics": {
                    "c_index": 0.91,
                },
            }
        },
    }

    store.append(report)

    result = store.latest()

    assert result is not None
    assert result["sections"]["performance"]["metrics"]["c_index"] == 0.91


def test_non_dictionary_report_is_rejected(tmp_path):
    store = MonitoringStore(tmp_path / "monitoring.jsonl")

    with pytest.raises(TypeError):
        store.append([])


def test_invalid_json_is_rejected(tmp_path):
    storage_path = tmp_path / "monitoring.jsonl"
    storage_path.write_text(
        '{"report_id": "valid"}\n'
        'invalid-json\n',
        encoding="utf-8",
    )

    store = MonitoringStore(storage_path)

    with pytest.raises(ValueError, match="Invalid JSON"):
        store.read_all()


def test_clear_removes_monitoring_history(tmp_path):
    store = MonitoringStore(tmp_path / "monitoring.jsonl")

    store.append(
        {
            "report_id": "monitoring-001",
            "overall_status": "HEALTHY",
        }
    )

    assert store.count() == 1

    store.clear()

    assert store.read_all() == []
    assert store.latest() is None