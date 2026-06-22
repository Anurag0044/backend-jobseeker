"""
tests/test_error_handlers.py
─────────────────────────────
Tests for centralised error handler behaviour via the live FastAPI app.

Tests:
  1. 401 response contains all ErrorResponse schema fields.
  2. 422 response contains all ErrorResponse schema fields.
  3. 500 response never exposes stack trace to client.
  4. Every error response contains a non-empty request_id field.
  5. Every error response contains a non-empty timestamp field.
"""

import pytest
from unittest.mock import patch


@pytest.mark.asyncio
async def test_401_response_has_error_schema_fields(client):
    """
    A 401 from a missing auth header must return the ErrorResponse schema:
    status, code, detail, request_id, timestamp.
    """
    response = await client.post(
        "/api/v1/generate",
        files={"resume_file": ("r.txt", b"x" * 200, "text/plain")},
        data={"job_url": "https://jobs.example.org/1"},
    )
    assert response.status_code == 401
    data = response.json()
    assert data["status"] == "error"
    assert data["code"] == 401
    assert "detail" in data
    assert "request_id" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_422_response_has_error_schema_fields(client, mock_firebase_token):
    """
    A 422 from an invalid job URL must return the ErrorResponse schema.
    """
    response = await client.post(
        "/api/v1/generate",
        headers={"Authorization": "Bearer fake-token"},
        files={"resume_file": ("r.txt", b"Jane Doe " * 30, "text/plain")},
        data={"job_url": "not-a-url"},
    )
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["code"] == 422
    assert "detail" in data
    assert "request_id" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_500_response_never_exposes_stack_trace(client, mock_firebase_token):
    """
    A 500 HTTPException raised by the pipeline must return the ErrorResponse
    schema and must not expose any stack trace, internal class names, or
    raw exception details to the client.
    """
    from fastapi import HTTPException as FE

    with patch(
        "agents.orchestrator.run_pipeline",
        side_effect=FE(
            status_code=500,
            detail="Resume optimization produced invalid output. Please retry.",
        ),
    ):
        response = await client.post(
            "/api/v1/generate",
            headers={"Authorization": "Bearer fake-token"},
            files={
                "resume_file": (
                    "resume.txt",
                    b"Jane Doe\nSoftware Engineer - Acme Corp (2021)\n" * 5,
                    "text/plain",
                )
            },
            data={"job_url": "https://boards.greenhouse.io/job/1"},
        )

    assert response.status_code == 500
    data = response.json()

    # Must follow ErrorResponse schema.
    assert data["status"] == "error"
    assert data["code"] == 500
    assert "request_id" in data
    assert "timestamp" in data

    body = response.text
    # Must NOT contain internal Python artifacts.
    assert "Traceback" not in body
    assert "File \"" not in body
    assert "orchestrator" not in body


@pytest.mark.asyncio
async def test_every_error_response_has_request_id(client):
    """
    Every error response must contain a non-empty request_id field
    (set by RequestIDMiddleware before any handler runs).
    """
    response = await client.post(
        "/api/v1/generate",
        # No auth header → 401
        files={"resume_file": ("r.txt", b"x" * 200, "text/plain")},
        data={"job_url": "https://jobs.company.com/1"},
    )
    data = response.json()
    assert "request_id" in data
    request_id = data["request_id"]
    assert request_id and request_id != "unknown"
    # Must look like a UUID4 (36 chars with hyphens).
    assert len(request_id) == 36
    assert request_id.count("-") == 4


@pytest.mark.asyncio
async def test_every_error_response_has_timestamp(client):
    """
    Every error response must contain a non-empty ISO timestamp field.
    """
    response = await client.post(
        "/api/v1/generate",
        # No auth → 401
        files={"resume_file": ("r.txt", b"x" * 200, "text/plain")},
        data={"job_url": "https://jobs.company.com/1"},
    )
    data = response.json()
    assert "timestamp" in data
    ts = data["timestamp"]
    assert ts and "T" in ts  # ISO-8601 format contains 'T' separator
