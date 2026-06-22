"""
core/error_handlers.py
──────────────────────
Centralised error response schema and FastAPI exception handlers.

Every error path in the API uses the ErrorResponse Pydantic model so clients
always receive a predictable, machine-readable payload.  Stack traces are
logged server-side via loguru and are NEVER returned to the client.

Registration in main.py:
    from fastapi.exceptions import RequestValidationError
    from core.error_handlers import (
        http_exception_handler,
        validation_error_handler,
        generic_exception_handler,
    )
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
"""

from datetime import datetime, timezone

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.logger import logger


# ── Standard error response schema ───────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Uniform error payload returned by every error handler."""
    status: str = "error"
    code: int
    detail: str
    request_id: str
    timestamp: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_request_id(request: Request) -> str:
    """Extract request_id from request.state (set by RequestIDMiddleware)."""
    return getattr(request.state, "request_id", "unknown")


def _build_response(
    request: Request,
    status_code: int,
    detail: str,
) -> JSONResponse:
    """Construct a JSONResponse using the ErrorResponse schema."""
    payload = ErrorResponse(
        code=status_code,
        detail=detail,
        request_id=_get_request_id(request),
        timestamp=_now_iso(),
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(),
    )


# ── Individual exception handlers ─────────────────────────────────────────────

async def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """
    Handler for all FastAPI HTTPException instances.

    Logs 4xx at WARNING level and 5xx at ERROR level.
    Never exposes internal details beyond the exception's own .detail.
    """
    request_id = _get_request_id(request)
    status_code = exc.status_code

    # Flatten detail — may be str or dict (e.g. from scraper JSON parse fail).
    if isinstance(exc.detail, dict):
        detail_str = exc.detail.get("error", str(exc.detail))
    else:
        detail_str = str(exc.detail)

    if status_code >= 500:
        logger.error(
            "[{}] HTTP {} — {} {}: {}",
            request_id,
            status_code,
            request.method,
            request.url.path,
            detail_str,
        )
    else:
        logger.warning(
            "[{}] HTTP {} — {} {}: {}",
            request_id,
            status_code,
            request.method,
            request.url.path,
            detail_str,
        )

    return _build_response(request, status_code, detail_str)


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handler for Pydantic / FastAPI request validation errors (422).

    Flattens the validation error list into a single human-readable string.
    """
    request_id = _get_request_id(request)
    errors = exc.errors()
    # Build a concise summary: "field: message; field2: message2"
    parts = []
    for err in errors:
        loc = " → ".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "invalid")
        parts.append(f"{loc}: {msg}" if loc else msg)
    detail_str = "; ".join(parts) or "Request validation failed."

    logger.warning(
        "[{}] HTTP 422 — {} {}: {}",
        request_id,
        request.method,
        request.url.path,
        detail_str,
    )
    return _build_response(request, 422, detail_str)


async def generic_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    Catch-all handler for any unhandled exception.

    Logs the full traceback server-side.  Returns a generic 500 message
    to the client — never exposes the stack trace or internal details.
    """
    request_id = _get_request_id(request)
    logger.exception(
        "[{}] Unhandled exception on {} {}: {}",
        request_id,
        request.method,
        request.url.path,
        repr(exc),
    )
    return _build_response(
        request,
        500,
        "An unexpected internal error occurred. Please try again later.",
    )


__all__ = [
    "ErrorResponse",
    "http_exception_handler",
    "validation_error_handler",
    "generic_exception_handler",
]
