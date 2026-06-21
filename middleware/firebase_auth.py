"""
middleware/firebase_auth.py
───────────────────────────
Firebase ID token verification dependency for FastAPI.

Changes from Phase 2:
  - Attaches decoded UID to request.state.user_uid so the rate limiter
    (slowapi) can use it as a rate-limit key without re-reading the token.
  - Adds loguru INFO / WARNING logs on auth success / failure.
  - Core verification logic (firebase_admin.auth.verify_id_token) is
    unchanged.

Note: verify_firebase_token now requires `request: Request` as a parameter
so it can write to request.state. FastAPI resolves this automatically.
"""

import json

import firebase_admin
from firebase_admin import auth, credentials
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core.config import settings
from core.logger import logger

_bearer_scheme = HTTPBearer(auto_error=False)


def _initialize_firebase() -> None:
    """Initialize Firebase Admin SDK once per process."""
    if not firebase_admin._apps:
        service_account_info: dict = json.loads(settings.FIREBASE_SERVICE_ACCOUNT_JSON)
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)


def verify_firebase_token(
    request: Request,
    credentials_header: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict:
    """
    FastAPI dependency: verify the Bearer token and return the decoded dict.

    Side-effect: sets request.state.user_uid so downstream components
    (rate limiter, orchestrator) can read the UID without re-parsing the
    token.

    Args:
        request:             The current FastAPI Request (injected by FastAPI).
        credentials_header:  HTTP Bearer credentials parsed from the
                             Authorization header.

    Returns:
        Decoded Firebase token dict (contains 'uid', 'email', etc.).

    Raises:
        HTTPException 401: Missing or invalid/expired token.
    """
    if credentials_header is None:
        logger.warning("Auth failed: Authorization header missing.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing.",
        )

    token: str = credentials_header.credentials

    try:
        decoded_token: dict = auth.verify_id_token(token)
    except Exception as exc:
        logger.warning("Auth failed: token verification error — {}", str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        ) from exc

    uid: str = decoded_token.get("uid", "unknown")

    # ── Attach UID to request state for the rate limiter ─────────────────────
    request.state.user_uid = uid

    logger.info("Auth verified for UID: {}", uid)
    return decoded_token
