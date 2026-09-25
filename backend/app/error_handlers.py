from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.app.exceptions import ApplicationError


logger = logging.getLogger(__name__)


def _get_request_id(request: Request) -> str | None:
    """
    Retrieve the request ID assigned by the request tracing middleware.
    """

    return getattr(request.state, "request_id", None)


def _error_response(
    request: Request,
    *,
    status_code: int,
    error_code: str,
    message: str,
) -> JSONResponse:
    """
    Build the standard API error response.
    """

    request_id = _get_request_id(request)

    content: dict[str, Any] = {
        "status": "error",
        "error_code": error_code,
        "detail": message,
    }

    if request_id:
        content["request_id"] = request_id

    return JSONResponse(
        status_code=status_code,
        content=content,
    )


async def application_error_handler(
    request: Request,
    exc: ApplicationError,
) -> JSONResponse:
    """
    Handle known application exceptions.
    """

    request_id = _get_request_id(request)

    logger.warning(
        "Application error | error_code=%s | status_code=%s | request_id=%s | path=%s | message=%s",
        exc.error_code,
        exc.status_code,
        request_id,
        request.url.path,
        exc.message,
    )

    return _error_response(
        request,
        status_code=exc.status_code,
        error_code=exc.error_code,
        message=exc.message,
    )


async def request_validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Handle FastAPI/Pydantic request validation failures.

    Detailed validation internals are logged, while the client receives
    a controlled response.
    """

    request_id = _get_request_id(request)

    logger.warning(
        "Request validation error | request_id=%s | path=%s | errors=%s",
        request_id,
        request.url.path,
        exc.errors(),
    )

    return _error_response(
        request,
        status_code=422,
        error_code="REQUEST_VALIDATION_ERROR",
        message="Request validation failed.",
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Handle unexpected exceptions.

    The full traceback is logged server-side, but internal details
    are never exposed to the client.
    """

    request_id = _get_request_id(request)

    logger.exception(
        "Unhandled application exception | request_id=%s | method=%s | path=%s",
        request_id,
        request.method,
        request.url.path,
    )

    return _error_response(
        request,
        status_code=500,
        error_code="INTERNAL_SERVER_ERROR",
        message="An unexpected internal error occurred.",
    )


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register all centralized exception handlers.
    """

    app.add_exception_handler(
        ApplicationError,
        application_error_handler,
    )

    app.add_exception_handler(
        RequestValidationError,
        request_validation_error_handler,
    )

    app.add_exception_handler(
        Exception,
        unhandled_exception_handler,
    )