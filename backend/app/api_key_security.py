from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from backend.app.config import settings


def verify_api_key(
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key",
    ),
) -> None:
    """
    Validate the API key when API-key protection is enabled.

    Development:
        REQUIRE_API_KEY=false
        Requests are allowed without an API key.

    Production:
        REQUIRE_API_KEY=true
        A valid X-API-Key header is required.
    """

    if not settings.require_api_key:
        return

    if not settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured.",
        )

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required.",
        )

    if not secrets.compare_digest(
        x_api_key,
        settings.api_key,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )