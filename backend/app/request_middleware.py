from __future__ import annotations

import logging
import time
from urllib import response
import uuid

from backend.app.operational_metrics import operational_metrics
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from backend.app.request_context import (
    clear_request_id,
    set_request_id,
)


logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Adds request-level traceability.

    If the client provides X-Request-ID, it is preserved.

    Otherwise a new UUID is generated.

    The request ID is:
        - stored on request.state
        - stored in the request context
        - returned in the response header
        - included in request lifecycle logs
    """

    async def dispatch(self, request: Request, call_next):
        incoming_request_id = request.headers.get(REQUEST_ID_HEADER)

        request_id = (
            incoming_request_id.strip()
            if incoming_request_id and incoming_request_id.strip()
            else str(uuid.uuid4())
        )

        request.state.request_id = request_id
        set_request_id(request_id)

        start_time = time.perf_counter()

        logger.info(
            "Request started | request_id=%s | method=%s | path=%s",
            request_id,
            request.method,
            request.url.path,
        )

        try:
            response = await call_next(request)

            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            response.headers[REQUEST_ID_HEADER] = request_id

            operational_metrics.record_request(
                latency_ms=latency_ms,
                status_code=response.status_code,
            )

            logger.info(
                "Request completed | request_id=%s | method=%s | path=%s | status_code=%s | latency_ms=%.2f",
                request_id,
                request.method,
                request.url.path,
                response.status_code,
                latency_ms,
            )

            return response

        except Exception:
            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000
            operational_metrics.record_request(
                latency_ms=latency_ms,
                status_code=500,
            )
            logger.exception(
                "Request failed | request_id=%s | method=%s | path=%s | latency_ms=%.2f",
                request_id,
                request.method,
                request.url.path,
                latency_ms,
            )

            raise

        finally:
            clear_request_id()