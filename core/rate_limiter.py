"""
core/rate_limiter.py
────────────────────
Per-user rate limiting via slowapi.

Rules:
  - 5 requests per user per hour on /api/v1/generate
  - Rate limit key: Firebase user UID (set on request.state by
    middleware/firebase_auth.py after successful token verification)
  - Fallback key: remote IP address (for unauthenticated probes)

Wiring into main.py (see main.py for the actual integration):
  1. Import `limiter` and `rate_limit_exceeded_handler` from this module.
  2. Attach limiter to the app:  app.state.limiter = limiter
  3. Register the exception handler:
       app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
  4. Add SlowAPI middleware:
       app.add_middleware(SlowAPIASGIMiddleware)
  5. On the route, decorate with: @limiter.limit("5/hour")
     and add `request: Request` as the first parameter.

Note: The key_func MUST be synchronous — slowapi calls it synchronously
during request processing. The UID must already be in request.state before
the rate limiter evaluates the key. This is guaranteed because FastAPI
resolves `Depends(verify_firebase_token)` before the route body executes,
and verify_firebase_token sets request.state.user_uid as a side-effect.
"""

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


def _get_user_uid_key(request: Request) -> str:
    """
    Rate limit key function.

    Returns the authenticated Firebase UID stored on request.state, falling
    back to the remote IP if the UID is not available (e.g. during health
    check routes that bypass auth).
    """
    uid: str | None = getattr(request.state, "user_uid", None)
    if uid:
        return uid
    return get_remote_address(request)


# ── Singleton Limiter instance ────────────────────────────────────────────────
limiter = Limiter(key_func=_get_user_uid_key)


# ── Custom 429 exception handler ──────────────────────────────────────────────
def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """
    Return a structured JSON 429 response instead of slowapi's plain-text
    default. The detail message is user-facing and surfaced in the API.
    """
    return JSONResponse(
        status_code=429,
        content={
            "detail": (
                "Rate limit exceeded. You can generate 5 resumes per hour. "
                "Please wait before submitting another request."
            )
        },
    )


__all__ = ["limiter", "rate_limit_exceeded_handler"]
