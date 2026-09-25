from pathlib import Path

from backend.app.health_service import HealthService


class DummyPredictionService:
    pass


def test_liveness_returns_ok():
    service = HealthService()

    result = service.liveness()

    assert result["status"] == "ok"


def test_readiness_is_ready_when_required_files_exist(monkeypatch):
    service = HealthService(DummyPredictionService())

    monkeypatch.setattr(
        "backend.app.health_service.settings.model_path",
        Path(__file__),
    )
    monkeypatch.setattr(
        "backend.app.health_service.settings.scaler_path",
        Path(__file__),
    )
    monkeypatch.setattr(
        "backend.app.health_service.settings.features_path",
        Path(__file__),
    )

    result = service.readiness()

    assert result["status"] == "ready"
    assert result["checks"]["model"] is True
    assert result["checks"]["scaler"] is True
    assert result["checks"]["features"] is True
    assert result["checks"]["prediction_engine"] is True


def test_readiness_fails_when_model_is_missing(monkeypatch):
    service = HealthService(DummyPredictionService())

    monkeypatch.setattr(
        "backend.app.health_service.settings.model_path",
        Path("missing_model_file.pkl"),
    )
    monkeypatch.setattr(
        "backend.app.health_service.settings.scaler_path",
        Path(__file__),
    )
    monkeypatch.setattr(
        "backend.app.health_service.settings.features_path",
        Path(__file__),
    )

    result = service.readiness()

    assert result["status"] == "not_ready"
    assert result["checks"]["model"] is False


def test_readiness_fails_without_prediction_engine(monkeypatch):
    service = HealthService(None)

    monkeypatch.setattr(
        "backend.app.health_service.settings.model_path",
        Path(__file__),
    )
    monkeypatch.setattr(
        "backend.app.health_service.settings.scaler_path",
        Path(__file__),
    )
    monkeypatch.setattr(
        "backend.app.health_service.settings.features_path",
        Path(__file__),
    )

    result = service.readiness()

    assert result["status"] == "not_ready"
    assert result["checks"]["prediction_engine"] is False