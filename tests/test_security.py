from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.security_middleware import (
    SecurityHeadersMiddleware,
    RequestSizeLimitMiddleware,
)


def create_security_app() -> FastAPI:
    app = FastAPI()

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestSizeLimitMiddleware)

    @app.get("/test")
    def test_endpoint():
        return {"status": "ok"}

    return app


def test_security_headers_are_present():
    client = TestClient(create_security_app())

    response = client.get("/test")

    assert response.status_code == 200

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"

    assert (
        response.headers["Permissions-Policy"]
        == "camera=(), microphone=(), geolocation=()"
    )


def test_normal_request_is_allowed():
    client = TestClient(create_security_app())

    response = client.get("/test")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_oversized_request_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "backend.app.security_middleware.MAX_REQUEST_BODY_BYTES",
        10,
    )

    client = TestClient(create_security_app())

    response = client.post(
        "/test",
        content="this request is definitely too large",
        headers={
            "content-length": "100"
        },
    )

    assert response.status_code == 413

    body = response.json()

    assert body["status"] == "error"
    assert body["error_code"] == "REQUEST_TOO_LARGE"