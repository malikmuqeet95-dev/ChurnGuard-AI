from __future__ import annotations

import logging

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from backend.app.config import settings

logger = logging.getLogger(__name__)

MAX_REQUEST_BODY_BYTES = settings.max_request_body_bytes


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds baseline HTTP security headers to API responses.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"

        response.headers[
            "Permissions-Policy"
        ] = "camera=(), microphone=(), geolocation=()"

        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Rejects requests whose declared Content-Length exceeds
    the configured maximum.

    This protects the API from unnecessarily large request bodies.
    """

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")

        if content_length:
            try:
                content_length_value = int(content_length)
            except ValueError:
                content_length_value = None

            if (
                content_length_value is not None
                and content_length_value > MAX_REQUEST_BODY_BYTES
            ):
                logger.warning(
                    "Request rejected because body is too large | "
                    "content_length=%s | limit=%s | path=%s",
                    content_length_value,
                    MAX_REQUEST_BODY_BYTES,
                    request.url.path,
                )

                return JSONResponse(
                    status_code=413,
                    content={
                        "status": "error",
                        "error_code": "REQUEST_TOO_LARGE",
                        "detail": (
                            "Request body exceeds the configured "
                            "maximum size."
                        ),
                    },
                )

        return await call_next(request)