import json

import firebase_admin
from firebase_admin import auth, credentials
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core.config import settings

_bearer_scheme = HTTPBearer(auto_error=False)


def _initialize_firebase() -> None:
    if not firebase_admin._apps:
        service_account_info: dict = json.loads(settings.FIREBASE_SERVICE_ACCOUNT_JSON)
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)


def verify_firebase_token(
    credentials_header: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict:
    if credentials_header is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing.",
        )

    token: str = credentials_header.credentials

    try:
        decoded_token: dict = auth.verify_id_token(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )

    return decoded_token
