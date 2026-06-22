"""
middleware/security_headers.py
──────────────────────────────
Security headers middleware.

Adds a fixed set of HTTP security headers to every API response.
HSTS is intentionally omitted — Render.com handles it at the
infrastructure / CDN layer.

Register this as the SECOND middleware in main.py (after RequestIDMiddleware,
before CORSMiddleware) so security headers are applied even on CORS preflight
responses.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds OWASP-recommended security headers to every HTTP response.

    Headers applied:
      X-Content-Type-Options     — prevents MIME sniffing
      X-Frame-Options            — prevents clickjacking
      X-XSS-Protection           — legacy XSS filter for older browsers
      Referrer-Policy            — limits referrer leakage
      Cache-Control              — prevents caching of sensitive API responses
      Permissions-Policy         — restricts browser feature access
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        response: Response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )

        return response


__all__ = ["SecurityHeadersMiddleware"]
