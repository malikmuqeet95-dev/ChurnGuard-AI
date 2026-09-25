from __future__ import annotations

from contextvars import ContextVar


_request_id_context: ContextVar[str | None] = ContextVar(
    "request_id",
    default=None,
)


def set_request_id(request_id: str) -> None:
    """
    Store the request ID for the current execution context.
    """

    _request_id_context.set(request_id)


def get_request_id() -> str | None:
    """
    Retrieve the request ID for the current execution context.
    """

    return _request_id_context.get()


def clear_request_id() -> None:
    """
    Clear the request ID from the current execution context.
    """

    _request_id_context.set(None)