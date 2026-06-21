"""
tests/test_health.py
────────────────────
Tests for health check endpoints:
  - GET /          → basic liveness probe
  - GET /health/deep → deep readiness probe (Firebase + Gemini)
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
async def test_root_returns_200(client):
    """GET / returns 200 with the expected JSON body."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "project" in data


@pytest.mark.asyncio
async def test_deep_health_returns_200_when_both_services_ok(client):
    """
    GET /health/deep returns 200 when Firebase is initialized and Gemini
    responds to a minimal API call (both mocked).

    The deep_health_check endpoint does `import google.genai as genai` locally.
    We patch 'google.genai.Client' at the module level so the local import
    picks up the mock.
    """
    import firebase_admin

    mock_genai_response = MagicMock()
    mock_genai_response.text = "OK"

    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = mock_genai_response

    with patch.object(firebase_admin, "_apps", {"[DEFAULT]": object()}):
        with patch("google.genai.Client", return_value=mock_client_instance):
            response = await client.get("/health/deep")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["firebase"] == "connected"
    assert data["gemini"] == "connected"
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_deep_health_returns_503_when_gemini_fails(client):
    """
    GET /health/deep returns 503 when the Gemini API call raises an exception.
    """
    import firebase_admin

    with patch.object(firebase_admin, "_apps", {"[DEFAULT]": object()}):
        with patch(
            "google.genai.Client",
            side_effect=Exception("Gemini API unreachable"),
        ):
            response = await client.get("/health/deep")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["firebase"] == "connected"
    assert data["gemini"] == "error"
    assert "error" in data
