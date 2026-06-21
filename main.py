"""
main.py
───────
Job Seeker's Concierge — FastAPI application entry point.

Phase 4 additions:
  - Loguru structured logging configured at import time.
  - SlowAPI rate limiter wired to app.state and exception handler.
  - Deep health check endpoint: GET /health/deep
    Verifies Firebase Admin SDK initialization and Gemini API connectivity.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

import firebase_admin
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIASGIMiddleware

from core.logger import logger
from core.rate_limiter import limiter, rate_limit_exceeded_handler
from middleware.firebase_auth import _initialize_firebase
from routers import generate


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup / shutdown lifecycle handler."""
    logger.info("Job Seeker's Concierge API starting up...")
    _initialize_firebase()
    logger.info("Firebase Admin SDK initialized.")
    yield
    logger.info("Job Seeker's Concierge API shut down.")


app = FastAPI(
    title="Job Seeker's Concierge API",
    description=(
        "Secure AI-powered backend for resume tailoring, cover letter generation, "
        "and job application assistance. Auth: Firebase. AI: Gemini (ADK)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ── Rate limiter ──────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
app.add_middleware(SlowAPIASGIMiddleware)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down in Phase 6
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(generate.router, prefix="/api/v1")


# ── Basic health check ────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health_check() -> dict:
    """Liveness probe — confirms the server is running."""
    return {"status": "online", "project": "Job Seeker's Concierge API"}


# ── Deep health check ─────────────────────────────────────────────────────────
@app.get("/health/deep", tags=["Health"])
async def deep_health_check(response: Response) -> JSONResponse:
    """
    Readiness probe — verifies both Firebase and Gemini connectivity.

    Checks:
      1. Firebase Admin SDK has an initialized app.
      2. Gemini API responds to a minimal generate_content call.

    Returns:
      200  { "status": "healthy",   "firebase": "connected",
             "gemini": "connected", "timestamp": "<ISO>" }
      503  { "status": "degraded",  "firebase": "<status>",
             "gemini": "<status>",  "timestamp": "<ISO>",
             "error": "<detail>" }
    """
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
        logger.error("Deep health: Firebase check error — {}", repr(exc))

    # ── Check 2: Gemini ───────────────────────────────────────────────────────
    try:
        import google.genai as genai
        from core.config import settings

        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        gemini_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="respond with OK",
        )
        # Accept any non-empty response as a sign the API is reachable.
        if gemini_response and gemini_response.text:
            gemini_status = "connected"
        else:
            gemini_status = "empty_response"
            error_detail = error_detail or "Gemini returned an empty response."
    except Exception as exc:
        gemini_status = "error"
        error_detail = error_detail or f"Gemini check failed: {exc!r}"
        logger.error("Deep health: Gemini check error — {}", repr(exc))

    # ── Build response ────────────────────────────────────────────────────────
    all_healthy: bool = (
        firebase_status == "connected" and gemini_status == "connected"
    )

    payload: dict = {
        "status": "healthy" if all_healthy else "degraded",
        "firebase": firebase_status,
        "gemini": gemini_status,
        "timestamp": timestamp,
    }
    if error_detail:
        payload["error"] = error_detail

    http_status: int = 200 if all_healthy else 503
    logger.info(
        "Deep health check: firebase={} gemini={} → HTTP {}",
        firebase_status,
        gemini_status,
        http_status,
    )
    return JSONResponse(content=payload, status_code=http_status)
