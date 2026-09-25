from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api_key_security import verify_api_key


def create_test_app():
    app = FastAPI()

    @app.get("/protected")
    def protected(
        _: None = __import__("fastapi").Depends(
            verify_api_key
        ),
    ):
        return {"status": "ok"}

    return app


def test_api_key_not_required_when_disabled(monkeypatch):
    monkeypatch.setattr(
        "backend.app.api_key_security.settings.require_api_key",
        False,
    )

    client = TestClient(create_test_app())

    response = client.get("/protected")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_missing_api_key_returns_401(monkeypatch):
    monkeypatch.setattr(
        "backend.app.api_key_security.settings.require_api_key",
        True,
    )

    monkeypatch.setattr(
        "backend.app.api_key_security.settings.api_key",
        "test-secret-key",
    )

    client = TestClient(create_test_app())

    response = client.get("/protected")

    assert response.status_code == 401


def test_invalid_api_key_returns_401(monkeypatch):
    monkeypatch.setattr(
        "backend.app.api_key_security.settings.require_api_key",
        True,
    )

    monkeypatch.setattr(
        "backend.app.api_key_security.settings.api_key",
        "test-secret-key",
    )

    client = TestClient(create_test_app())

    response = client.get(
        "/protected",
        headers={
            "X-API-Key": "wrong-key",
        },
    )

    assert response.status_code == 401


def test_valid_api_key_returns_200(monkeypatch):
    monkeypatch.setattr(
        "backend.app.api_key_security.settings.require_api_key",
        True,
    )

    monkeypatch.setattr(
        "backend.app.api_key_security.settings.api_key",
        "test-secret-key",
    )

    client = TestClient(create_test_app())

    response = client.get(
        "/protected",
        headers={
            "X-API-Key": "test-secret-key",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_missing_server_api_key_configuration_returns_503(
    monkeypatch,
):
    monkeypatch.setattr(
        "backend.app.api_key_security.settings.require_api_key",
        True,
    )

    monkeypatch.setattr(
        "backend.app.api_key_security.settings.api_key",
        "",
    )

    client = TestClient(create_test_app())

    response = client.get(
        "/protected",
    )

    assert response.status_code == 503