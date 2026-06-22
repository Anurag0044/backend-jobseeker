"""
middleware/firebase_auth.py
───────────────────────────
Firebase ID token verification dependency for FastAPI.

Phase 5 change:
  - All log lines now include [request_id] prefix for log correlation.
  - request_id is read from request.state.request_id (guaranteed to exist
    because RequestIDMiddleware runs before auth is evaluated).
  - Core verification logic (firebase_admin.auth.verify_id_token) is unchanged.
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

    Side-effects:
      - Sets request.state.user_uid for the rate limiter key function.
      - Logs auth success/failure with [request_id] prefix.

    Args:
        request:             The current FastAPI Request.
        credentials_header:  HTTP Bearer credentials from Authorization header.

    Returns:
        Decoded Firebase token dict (contains 'uid', 'email', etc.).

    Raises:
        HTTPException 401: Missing or invalid/expired token.
    """
    request_id: str = getattr(request.state, "request_id", "unknown")

    if credentials_header is None:
        logger.warning("[{}] Auth failed — Authorization header missing.", request_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing.",
        )

    token: str = credentials_header.credentials

    try:
        decoded_token: dict = auth.verify_id_token(token)
    except Exception as exc:
        logger.warning(
            "[{}] Auth failed — token verification error: {}",
            request_id,
            str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        ) from exc

    uid: str = decoded_token.get("uid", "unknown")
    request.state.user_uid = uid

    logger.info("[{}] Auth verified — UID: {}", request_id, uid)
    return decoded_token
