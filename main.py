"""
main.py
───────
Job Seeker's Concierge — FastAPI application entry point.

Phase 5 additions:
  - RequestIDMiddleware:    first middleware — every request gets a UUID.
  - SecurityHeadersMiddleware: second — wraps all responses with security
    headers.
  - CORSMiddleware:         third.
  - Centralised error handlers (HTTPException, RequestValidationError,
    Exception catch-all) replace the previous bare SlowAPI handler.
  - Startup env-var validation: crashes immediately with a clear message
    if GEMINI_API_KEY or FIREBASE_SERVICE_ACCOUNT_JSON are missing.
  - ENVIRONMENT variable drives log context (production / development).
"""
from dotenv import load_dotenv
load_dotenv()
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

import firebase_admin
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIASGIMiddleware

from core.error_handlers import (
    generic_exception_handler,
    http_exception_handler,
    validation_error_handler,
)
from core.logger import logger
from core.rate_limiter import limiter, rate_limit_exceeded_handler
from middleware.firebase_auth import _initialize_firebase
from middleware.request_id import RequestIDMiddleware
from middleware.security_headers import SecurityHeadersMiddleware
from routers import whatsapp


# ── Startup env-var guard ─────────────────────────────────────────────────────
def _assert_env_vars() -> None:
    """
    Crash immediately with a readable message if required env vars are missing.
    This surfaces configuration errors at startup rather than at runtime.
    """
    missing: list[str] = []
    if not os.environ.get("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")
    # Firebase is bypassed for local testing
    if missing:
        print(
            f"\n[FATAL] Missing required environment variables: {', '.join(missing)}\n"
            "Set them in your .env file or Render.com environment settings.\n",
            file=sys.stderr,
        )
        sys.exit(1)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup / shutdown lifecycle handler."""
    environment = os.environ.get("ENVIRONMENT", "development")
    logger.info(
        "Job Seeker's Concierge API v2.1.0 starting up — environment: {}",
        environment,
    )
    _assert_env_vars()
    try:
        _initialize_firebase()
        logger.info("Firebase Admin SDK initialized.")
    except Exception as e:
        logger.warning(f"Firebase not initialized locally: {e}")
    yield
    logger.info("Job Seeker's Concierge API shut down.")


app = FastAPI(
    title="Job Seeker's Concierge API",
    description=(
        "Secure AI-powered backend for resume tailoring, cover letter generation, "
        "and job application assistance. Auth: Firebase. AI: Gemini (ADK)."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

# ── Rate limiter (must come before middleware stack) ──────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)  # type: ignore[arg-type]
app.add_middleware(SlowAPIASGIMiddleware)

# ── Middleware stack (order is CRITICAL — last added = outermost) ─────────────
# Execution order for a request:   RequestID → SecurityHeaders → CORS → handler
# Execution order for a response:  handler → CORS → SecurityHeaders → RequestID
#
# add_middleware() inserts at the FRONT of the stack, so we add in REVERSE order.
# Add CORSMiddleware first (innermost), then SecurityHeaders, then RequestID
# (outermost — runs first on request, last on response).

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down in Phase 6
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestIDMiddleware)

# ── Exception handlers ────────────────────────────────────────────────────────
app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore[arg-type]
app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(Exception, generic_exception_handler)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(whatsapp.router, prefix="/api/v1")


# ── Basic health check ────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health_check() -> dict:
    """Liveness probe — confirms the server is running."""
    return {"status": "online", "project": "Job Seeker's Concierge API"}


# ── Deep health check ─────────────────────────────────────────────────────────
@app.get("/health/deep", tags=["Health"])
async def deep_health_check(request: Request, response: Response) -> JSONResponse:
    """
    Readiness probe — verifies Firebase Admin SDK and Gemini API connectivity.

    Returns:
      200  { "status": "healthy",   "firebase": "connected",
             "gemini": "connected", "timestamp": "<ISO>",
             "request_id": "<uuid>" }
      503  { "status": "degraded",  ... "error": "<detail>" }
    """
    request_id: str = getattr(request.state, "request_id", "unknown")
    timestamp: str = datetime.now(timezone.utc).isoformat()
    firebase_status: str = "unknown"
    gemini_status: str = "unknown"
    error_detail: str = ""

    # ── Check 1: Firebase ─────────────────────────────────────────────────────
    try:
        if firebase_admin._apps:
            firebase_status = "connected"
        else:
            firebase_status = "not_initialized"
            error_detail = "Firebase Admin SDK is not initialized."
    except Exception as exc:
        firebase_status = "error"
        error_detail = f"Firebase check failed: {exc!r}"
        logger.error(
            "[{}] Deep health: Firebase check error — {}", request_id, repr(exc)
        )

    # ── Check 2: Gemini ───────────────────────────────────────────────────────
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
        gemini_response = llm.invoke("respond with OK")
        if gemini_response and gemini_response.content:
            gemini_status = "connected"
        else:
            gemini_status = "empty_response"
            error_detail = error_detail or "Gemini returned an empty response."
    except Exception as exc:
        gemini_status = "error"
        error_detail = error_detail or f"Gemini check failed: {exc!r}"
        logger.error(
            "[{}] Deep health: Gemini check error — {}", request_id, repr(exc)
        )

    # ── Build response ────────────────────────────────────────────────────────
    all_healthy: bool = (
        firebase_status == "connected" and gemini_status == "connected"
    )
    payload: dict = {
        "status": "healthy" if all_healthy else "degraded",
        "firebase": firebase_status,
        "gemini": gemini_status,
        "timestamp": timestamp,
        "request_id": request_id,
    }
    if error_detail:
        payload["error"] = error_detail

    http_status = 200 if all_healthy else 503
    logger.info(
        "[{}] Deep health: firebase={} gemini={} → HTTP {}",
        request_id,
        firebase_status,
        gemini_status,
        http_status,
    )
    return JSONResponse(content=payload, status_code=http_status)
