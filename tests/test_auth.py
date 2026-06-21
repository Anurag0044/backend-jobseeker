"""
tests/test_auth.py
──────────────────
Tests for Firebase authentication on POST /api/v1/generate:
  1. No Authorization header → 401.
  2. Invalid / expired token → 401.
  3. Valid mocked token → passes auth (not 401).
"""

import pytest
from unittest.mock import patch


@pytest.mark.asyncio
async def test_generate_no_auth_header_returns_401(client):
    """
    POST /api/v1/generate with no Authorization header must return 401.
    Sends a minimal multipart form with a TXT file so the request is
    otherwise well-formed — only the token is missing.
    """
    response = await client.post(
        "/api/v1/generate",
        files={"resume_file": ("resume.txt", b"some text content " * 20, "text/plain")},
        data={"job_url": "https://example.com/jobs/123"},
    )
    assert response.status_code == 401
    assert "Authorization header missing" in response.json()["detail"]


@pytest.mark.asyncio
async def test_generate_invalid_token_returns_401(client):
    """
    POST /api/v1/generate with a syntactically valid but unverifiable Bearer
    token must return 401.
    """
    import firebase_admin
    import firebase_admin.auth as fb_auth

    with patch.object(firebase_admin, "_apps", {"[DEFAULT]": object()}):
        with patch.object(
            fb_auth,
            "verify_id_token",
            side_effect=Exception("Token expired"),
        ):
            response = await client.post(
                "/api/v1/generate",
                headers={"Authorization": "Bearer this.is.invalid"},
                files={
                    "resume_file": (
                        "resume.txt",
                        b"some text content " * 20,
                        "text/plain",
                    )
                },
                data={"job_url": "https://example.com/jobs/123"},
            )

    assert response.status_code == 401
    assert "Invalid or expired token" in response.json()["detail"]


@pytest.mark.asyncio
async def test_generate_valid_token_passes_auth(client, mock_firebase_token):
    """
    POST /api/v1/generate with a valid mocked token must pass authentication
    (i.e. NOT return 401). The pipeline agents are mocked so no Gemini call
    is made; we only verify auth passes through.
    """
    fake_pipeline_result = {
        "status": "success",
        "user_uid": "test-user-uid-abc123",
        "total_processing_time_seconds": 0.1,
        "job_insights": {
            "job_title": "Engineer",
            "company_name": "Acme",
            "top_5_skills": ["Python", "FastAPI", "AWS", "Docker", "Redis"],
            "experience_level": "Mid",
            "company_culture": "Fast-paced",
            "key_responsibilities": ["Build APIs"],
        },
        "optimized_resume": "# Optimized Resume\n\nContent here.",
        "cover_letter": " ".join(["word"] * 260),  # 260 words — valid length
    }

    with patch(
        "agents.orchestrator.run_pipeline",
        return_value=fake_pipeline_result,
    ):
        # Also patch the scraper URL validation so a fake URL is accepted.
        with patch("agents.scraper_agent._validate_url", return_value=None):
            response = await client.post(
                "/api/v1/generate",
                headers={"Authorization": "Bearer fake-valid-token"},
                files={
                    "resume_file": (
                        "resume.txt",
                        b"Jane Doe\nSoftware Engineer\n" + b"Skills: Python\n" * 20,
                        "text/plain",
                    )
                },
                data={"job_url": "https://example.com/jobs/valid"},
            )

    # Must not be 401 — auth passed.
    assert response.status_code != 401
    assert response.status_code == 200
