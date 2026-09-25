from fastapi import FastAPI

from fastapi.testclient import TestClient

from backend.app.request_context import (
    get_request_id,
)

from backend.app.request_middleware import (
    REQUEST_ID_HEADER,
    RequestIDMiddleware,
)


def create_test_app() -> FastAPI:
    app = FastAPI()

    app.add_middleware(RequestIDMiddleware)

    @app.get("/test")
    def test_endpoint():
        return {
            "request_id": get_request_id(),
        }

    return app


def test_request_id_is_generated_when_missing():
    client = TestClient(create_test_app())

    response = client.get("/test")

    assert response.status_code == 200

    response_request_id = response.headers.get(
        REQUEST_ID_HEADER
    )

    body_request_id = response.json().get(
        "request_id"
    )

    assert response_request_id
    assert body_request_id
    assert response_request_id == body_request_id


def test_client_request_id_is_preserved():
    client = TestClient(create_test_app())

    supplied_request_id = "test-request-12345"

    response = client.get(
        "/test",
        headers={
            REQUEST_ID_HEADER: supplied_request_id
        },
    )

    assert response.status_code == 200

    assert (
        response.headers.get(REQUEST_ID_HEADER)
        == supplied_request_id
    )

    assert (
        response.json()["request_id"]
        == supplied_request_id
    )


def test_request_id_is_unique_for_separate_requests():
    client = TestClient(create_test_app())

    response_one = client.get("/test")
    response_two = client.get("/test")

    request_id_one = response_one.headers.get(
        REQUEST_ID_HEADER
    )

    request_id_two = response_two.headers.get(
        REQUEST_ID_HEADER
    )

    assert request_id_one
    assert request_id_two
    assert request_id_one != request_id_two


def test_request_id_is_available_inside_request_context():
    client = TestClient(create_test_app())

    response = client.get("/test")

    assert response.json()["request_id"]