"""
middleware/request_id.py
────────────────────────
Request ID middleware for end-to-end log tracing.

For every incoming request:
  1. Generates a UUID4 request ID.
  2. Attaches it to request.state.request_id.
  3. Adds it to the response as the X-Request-ID header.

This must be the FIRST middleware registered in main.py so every request
has an ID before any other middleware or handler runs.

Usage in log lines:
    from core.logger import logger
    request_id = getattr(request.state, "request_id", "unknown")
    logger.info("[{}] Something happened", request_id)
"""

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware that assigns a unique request ID to every HTTP request.

    The ID is:
      - Stored on request.state.request_id (readable by all handlers).
      - Returned in the X-Request-ID response header.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id: str = str(uuid.uuid4())
        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


__all__ = ["RequestIDMiddleware"]
