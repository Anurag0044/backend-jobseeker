"""
tests/conftest.py
─────────────────
Shared pytest fixtures for the Job Seeker's Concierge test suite.

All fixtures use mocks — no real Firebase or Gemini API calls are made.

Fixtures (Phase 4 + Phase 5 additions):
  client              — AsyncClient pointed at the FastAPI app.
  mock_firebase_token — Patches firebase_admin.auth.verify_id_token.
  sample_resume_txt   — bytes of a minimal plain-text resume.
  sample_resume_pdf   — bytes of a structurally valid PDF.
  mock_request_id     — Returns a fixed UUID string for deterministic tests.
  mock_bleach         — Patches bleach.clean() to return input unchanged.
  mock_tenacity_retry — Disables tenacity wait so retries run instantly.
"""

import io
import os
import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ── Stub env vars before any app module is imported ───────────────────────────
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key-placeholder")
os.environ.setdefault(
    "FIREBASE_SERVICE_ACCOUNT_JSON",
    '{"type":"service_account","project_id":"test","private_key_id":"k",'
    '"private_key":"-----BEGIN RSA PRIVATE KEY-----\\nMIIEowIBAAKCAQEA2a\\n'
    "-----END RSA PRIVATE KEY-----\\n\","
    '"client_email":"test@test.iam.gserviceaccount.com","client_id":"1",'
    '"auth_uri":"https://accounts.google.com/o/oauth2/auth",'
    '"token_uri":"https://oauth2.googleapis.com/token"}',
)
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("ENVIRONMENT", "test")

from main import app  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
FAKE_UID = "test-user-uid-abc123"
FAKE_DECODED_TOKEN: dict = {
    "uid": FAKE_UID,
    "email": "test@example.com",
    "email_verified": True,
}
FIXED_REQUEST_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


# ── Phase 4 fixtures ──────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client():
    """AsyncClient pointed at the FastAPI app via ASGI transport."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def mock_firebase_token(monkeypatch):
    """
    Patch firebase_admin.auth.verify_id_token to return FAKE_DECODED_TOKEN
    and simulate an initialized Firebase app.
    """
    import firebase_admin
    import firebase_admin.auth as fb_auth

    monkeypatch.setattr(firebase_admin, "_apps", {"[DEFAULT]": object()})
    monkeypatch.setattr(
        fb_auth,
        "verify_id_token",
        lambda token, **kwargs: FAKE_DECODED_TOKEN,
    )
    return FAKE_DECODED_TOKEN


@pytest.fixture
def sample_resume_txt() -> bytes:
    """Minimal plain-text resume (>100 chars) for upload tests."""
    resume = (
        "Jane Doe\n"
        "jane.doe@example.com | linkedin.com/in/janedoe\n\n"
        "EXPERIENCE\n"
        "Software Engineer - Acme Corp (2021-Present)\n"
        "Built REST APIs with Python and FastAPI serving 50k daily users.\n"
        "Reduced API latency by 35% through Redis caching.\n"
        "Led migration of 3 services to Docker on AWS ECS.\n\n"
        "Junior Developer - StartupXYZ (2019-2021)\n"
        "Developed React front-end components.\n"
        "Wrote pytest integration tests achieving 85% coverage.\n\n"
        "EDUCATION\n"
        "B.Sc. Computer Science - State University (2019)\n\n"
        "SKILLS\n"
        "Python, FastAPI, React, Docker, AWS, PostgreSQL, Redis, pytest\n"
    )
    return resume.encode("utf-8")


@pytest.fixture
def sample_resume_pdf(sample_resume_txt) -> bytes:
    """Structurally valid in-memory PDF for upload tests."""
    from PyPDF2 import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    return buffer.read()


# ── Phase 5 fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_request_id(monkeypatch):
    """
    Return a fixed UUID string for deterministic test output.

    Sets request.state.request_id to FIXED_REQUEST_ID on any Request object
    by patching uuid.uuid4 used in RequestIDMiddleware to return the fixed value.
    """
    monkeypatch.setattr(
        "middleware.request_id.uuid.uuid4",
        lambda: uuid.UUID(FIXED_REQUEST_ID),
    )
    return FIXED_REQUEST_ID


@pytest.fixture
def mock_bleach(monkeypatch):
    """
    Patch bleach.clean() to return the input text unchanged.
    Avoids bleach processing overhead and dependency on bleach internals
    in unit tests focused on other logic.
    """
    monkeypatch.setattr(
        "agents.scraper_agent.bleach.clean",
        lambda text, **kwargs: text,
    )


@pytest.fixture
def mock_tenacity_retry(monkeypatch):
    """
    Disable tenacity retry waits so retries run instantly in the test suite.
    Patches the wait_exponential used in all three agents to return 0 seconds.
    """
    import tenacity

    monkeypatch.setattr(
        tenacity,
        "wait_exponential",
        lambda *args, **kwargs: tenacity.wait_none(),
    )
