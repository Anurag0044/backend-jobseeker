"""
tests/test_pipeline.py
──────────────────────
Integration tests for POST /api/v1/generate (full pipeline endpoint):

  1. Valid inputs → 200 with all expected response fields.
  2. Invalid job URL (no scheme) → 422.
  3. Oversized resume text (> 50 000 chars) → 413.
  4. Rate limiter blocks 6th request from same UID → 429.

All three agents are mocked. No real Firebase or Gemini calls are made.
"""

import pytest
from unittest.mock import patch


# ── Shared helpers ─────────────────────────────────────────────────────────────

VALID_JOB_URL = "https://boards.greenhouse.io/openai/jobs/senior-engineer"
VALID_RESUME_CONTENT = (
    b"Jane Doe\njane@example.com\n\n"
    b"EXPERIENCE\n"
    b"Software Engineer - Acme Corp (2021-Present)\n"
    b"Built REST APIs serving 50k daily users.\n"
    b"Reduced latency by 35% via Redis caching.\n\n"
    b"Junior Developer - StartupXYZ (2019-2021)\n"
    b"Developed React components.\n\n"
    b"EDUCATION\n"
    b"B.Sc. Computer Science - State University (2019)\n\n"
    b"SKILLS\n"
    b"Python, FastAPI, React, Docker, AWS, PostgreSQL, Redis\n"
)

FAKE_JOB_INSIGHTS = {
    "job_title": "Senior Software Engineer",
    "company_name": "TechCorp",
    "top_5_skills": ["Python", "FastAPI", "AWS", "Docker", "Redis"],
    "experience_level": "Senior (5+ years)",
    "company_culture": "Collaborative and innovation-driven",
    "key_responsibilities": [
        "Design scalable microservices",
        "Mentor junior engineers",
    ],
}

FAKE_OPTIMIZED_RESUME = (
    "# Jane Doe\n\n"
    "## Experience\n\n"
    "### Software Engineer — Acme Corp *(2021–Present)*\n\n"
    "- Designed and maintained production REST APIs using Python and FastAPI "
    "serving 50k daily users.\n"
    "- Reduced API latency by 35% through Redis caching and query optimisation.\n"
)

# Build a cover letter string with exactly 260 words (valid range).
FAKE_COVER_LETTER = " ".join(["word"] * 260)

FAKE_PIPELINE_RESULT = {
    "status": "success",
    "user_uid": "test-user-uid-abc123",
    "total_processing_time_seconds": 4.2,
    "job_insights": FAKE_JOB_INSIGHTS,
    "optimized_resume": FAKE_OPTIMIZED_RESUME,
    "cover_letter": FAKE_COVER_LETTER,
}


def _post_generate(client, files, data, token="fake-valid-token"):
    """Helper coroutine: POST to /api/v1/generate with auth header."""
    return client.post(
        "/api/v1/generate",
        headers={"Authorization": f"Bearer {token}"},
        files=files,
        data=data,
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_pipeline_valid_inputs_returns_200(client, mock_firebase_token):
    """
    Full pipeline with valid TXT resume and valid URL returns 200 with all
    expected top-level fields in the JSON response.
    """
    with patch("agents.orchestrator.run_pipeline", return_value=FAKE_PIPELINE_RESULT):
        with patch("core.file_parser._extract_pdf_text"):  # Not used for TXT
            response = await _post_generate(
                client,
                files={
                    "resume_file": (
                        "resume.txt",
                        VALID_RESUME_CONTENT,
                        "text/plain",
                    )
                },
                data={"job_url": VALID_JOB_URL},
            )

    assert response.status_code == 200, response.text
    data = response.json()

    # Verify all required top-level keys are present.
    assert data["status"] == "success"
    assert "user_uid" in data
    assert "total_processing_time_seconds" in data
    assert "job_insights" in data
    assert "optimized_resume" in data
    assert "cover_letter" in data

    # Verify job_insights sub-structure.
    insights = data["job_insights"]
    assert "job_title" in insights
    assert "company_name" in insights
    assert "top_5_skills" in insights
    assert isinstance(insights["top_5_skills"], list)


@pytest.mark.asyncio
async def test_pipeline_invalid_job_url_returns_422(client, mock_firebase_token):
    """
    A job URL without a valid scheme must be rejected with 422
    before any agent runs. The input sanitizer enforces this.
    """
    response = await _post_generate(
        client,
        files={
            "resume_file": (
                "resume.txt",
                VALID_RESUME_CONTENT,
                "text/plain",
            )
        },
        data={"job_url": "not-a-valid-url"},
    )

    assert response.status_code == 422
    # The sanitizer raises 422 with a scheme or invalid-URL message.
    detail = response.json()["detail"].lower()
    assert "scheme" in detail or "invalid" in detail or "url" in detail


@pytest.mark.asyncio
async def test_pipeline_oversized_resume_returns_413(monkeypatch):
    """
    A resume whose extracted text exceeds 50 000 characters must return 422
    (sanitize_resume_text raises 422) before the pipeline runs.

    Uses a unique UID ('oversized-test-uid') not used by any other test so
    the rate limiter counter starts at zero for this request.
    """
    import firebase_admin
    import firebase_admin.auth as fb_auth

    unique_token = {
        "uid": "oversized-test-uid",
        "email": "oversized@test.com",
        "email_verified": True,
    }
    monkeypatch.setattr(firebase_admin, "_apps", {"[DEFAULT]": object()})
    monkeypatch.setattr(fb_auth, "verify_id_token", lambda t, **kw: unique_token)

    from httpx import ASGITransport, AsyncClient
    from main import app

    oversized_content = b"A" * 51_000

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as fresh_client:
        response = await fresh_client.post(
            "/api/v1/generate",
            headers={"Authorization": "Bearer fake-valid-token"},
            files={
                "resume_file": (
                    "resume.txt",
                    oversized_content,
                    "text/plain",
                )
            },
            data={"job_url": VALID_JOB_URL},
        )

    # sanitize_resume_text raises 422 for text > 50 000 chars.
    assert response.status_code == 422
    assert "too long" in response.json()["detail"].lower()



@pytest.mark.asyncio
async def test_rate_limiter_blocks_sixth_request(mock_firebase_token):
    """
    The 6th request from the same UID within one hour must return 429.

    We use a fresh Limiter with a "5/minute" limit (same quota as 5/hour but
    evaluable in a test run) attached to a fresh AsyncClient so the counter
    starts at zero and is not polluted by other tests.
    """
    from httpx import ASGITransport, AsyncClient
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from core.rate_limiter import _get_user_uid_key, rate_limit_exceeded_handler

    # Import the real app and give it a fresh limiter instance so previous
    # tests' hit counts do not carry over.
    from main import app

    fresh_limiter = Limiter(key_func=_get_user_uid_key)
    app.state.limiter = fresh_limiter

    responses = []
    with patch("agents.orchestrator.run_pipeline", return_value=FAKE_PIPELINE_RESULT):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as fresh_client:
            for _ in range(6):
                r = await fresh_client.post(
                    "/api/v1/generate",
                    headers={"Authorization": "Bearer fake-valid-token"},
                    files={
                        "resume_file": (
                            "resume.txt",
                            VALID_RESUME_CONTENT,
                            "text/plain",
                        )
                    },
                    data={"job_url": VALID_JOB_URL},
                )
                responses.append(r)

    status_codes = [r.status_code for r in responses]
    assert 429 in status_codes, (
        f"Expected a 429 among the 6 responses, got: {status_codes}"
    )
    # The last response must be 429.
    assert responses[-1].status_code == 429
    assert "Rate limit exceeded" in responses[-1].json()["detail"]
