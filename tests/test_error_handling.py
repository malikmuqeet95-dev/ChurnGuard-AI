from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.error_handlers import register_exception_handlers
from backend.app.exceptions import (
    ApplicationError,
    PredictionApplicationError,
)


def create_test_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/application-error")
    def application_error():
        raise ApplicationError(
            "Controlled application failure.",
            status_code=400,
            error_code="TEST_APPLICATION_ERROR",
        )

    @app.get("/prediction-error")
    def prediction_error():
        raise PredictionApplicationError(
            "Prediction processing failed."
        )

    @app.get("/unexpected-error")
    def unexpected_error():
        raise RuntimeError("This is an internal failure.")

    @app.get("/validation")
    def validation(value: int):
        return {"value": value}

    return app


def test_application_error_returns_standard_response():
    client = TestClient(create_test_app())

    response = client.get("/application-error")

    assert response.status_code == 400

    body = response.json()

    assert body["status"] == "error"
    assert body["error_code"] == "TEST_APPLICATION_ERROR"
    assert body["detail"] == "Controlled application failure."


def test_prediction_error_returns_standard_response():
    client = TestClient(create_test_app())

    response = client.get("/prediction-error")

    assert response.status_code == 400

    body = response.json()

    assert body["status"] == "error"
    assert body["error_code"] == "PREDICTION_ERROR"
    assert body["detail"] == "Prediction processing failed."


def test_unexpected_error_does_not_expose_internal_details():
    client = TestClient(
        create_test_app(),
        raise_server_exceptions=False,
    )

    response = client.get("/unexpected-error")

    assert response.status_code == 500

    body = response.json()

    assert body["status"] == "error"
    assert body["error_code"] == "INTERNAL_SERVER_ERROR"
    assert body["detail"] == "An unexpected internal error occurred."

    assert "This is an internal failure" not in response.text
    assert "RuntimeError" not in response.text


def test_request_validation_has_standard_response():
    client = TestClient(create_test_app())

    response = client.get(
        "/validation",
        params={"value": "not-a-number"},
    )

    assert response.status_code == 422

    body = response.json()

    assert body["status"] == "error"
    assert body["error_code"] == "REQUEST_VALIDATION_ERROR"
    assert body["detail"] == "Request validation failed."