"""
tests/test_sanitizer.py
───────────────────────
Unit tests for core/input_sanitizer.py

Tests:
  1.  Valid HTTPS URL passes sanitization.
  2.  HTTP URL passes sanitization.
  3.  URL without scheme raises 422.
  4.  URL with localhost raises 422.
  5.  URL over 500 characters raises 422.
  6.  example.com raises 422.
  7.  Valid resume text passes sanitization.
  8.  Resume under 100 characters raises 422.
  9.  Resume over 50 000 characters raises 422.
  10. Null bytes are stripped from resume text.
  11. Excessive blank lines are normalized.
"""

import pytest

from core.input_sanitizer import sanitize_job_url, sanitize_resume_text
from fastapi import HTTPException


# ── URL sanitizer tests ───────────────────────────────────────────────────────

def test_valid_https_url_passes():
    """Valid HTTPS URL is returned clean (whitespace stripped)."""
    url = "  https://boards.greenhouse.io/openai/jobs/123  "
    result = sanitize_job_url(url)
    assert result == "https://boards.greenhouse.io/openai/jobs/123"


def test_valid_http_url_passes():
    """HTTP (non-HTTPS) job URLs are also valid."""
    url = "http://careers.example.org/job/456"
    result = sanitize_job_url(url)
    assert result == url


def test_url_without_scheme_raises_422():
    """URL missing the scheme (e.g. just 'careers.company.com') raises 422."""
    with pytest.raises(HTTPException) as exc_info:
        sanitize_job_url("careers.company.com/job/789")
    assert exc_info.value.status_code == 422
    assert "scheme" in exc_info.value.detail.lower() or "invalid" in exc_info.value.detail.lower()


def test_localhost_url_raises_422():
    """localhost is blocked and raises 422."""
    with pytest.raises(HTTPException) as exc_info:
        sanitize_job_url("http://localhost:8000/job/1")
    assert exc_info.value.status_code == 422
    assert "not allowed" in exc_info.value.detail


def test_url_over_500_chars_raises_422():
    """URL longer than 500 characters raises 422."""
    long_url = "https://example.org/" + "a" * 490
    assert len(long_url) > 500
    with pytest.raises(HTTPException) as exc_info:
        sanitize_job_url(long_url)
    assert exc_info.value.status_code == 422
    assert "too long" in exc_info.value.detail.lower()


def test_example_com_raises_422():
    """example.com is in the block-list and must raise 422."""
    with pytest.raises(HTTPException) as exc_info:
        sanitize_job_url("https://example.com/jobs/123")
    assert exc_info.value.status_code == 422
    assert "not allowed" in exc_info.value.detail


# ── Resume text sanitizer tests ───────────────────────────────────────────────

_VALID_RESUME = (
    "Jane Doe\n"
    "Software Engineer - Acme Corp (2021-Present)\n"
    "Built REST APIs serving 50k users daily using Python and FastAPI.\n"
    "Reduced latency by 35% with Redis caching.\n"
    "Junior Developer - StartupXYZ (2019-2021)\n"
    "Developed React components and wrote pytest tests.\n"
    "B.Sc. Computer Science - State University (2019)\n"
    "Skills: Python, FastAPI, Docker, AWS, Redis, PostgreSQL\n"
)


def test_valid_resume_text_passes():
    """Valid resume text is returned clean with leading/trailing whitespace stripped."""
    padded = "  " + _VALID_RESUME + "  "
    result = sanitize_resume_text(padded)
    assert isinstance(result, str)
    assert len(result) >= 100
    assert result == result.strip()


def test_resume_under_100_chars_raises_422():
    """Resume text under 100 characters raises 422."""
    short = "Jane Doe\nSoftware Engineer\n"
    assert len(short) < 100
    with pytest.raises(HTTPException) as exc_info:
        sanitize_resume_text(short)
    assert exc_info.value.status_code == 422
    assert "too short" in exc_info.value.detail.lower()


def test_resume_over_50000_chars_raises_422():
    """Resume text over 50 000 characters raises 422."""
    long_resume = _VALID_RESUME + ("A " * 30_000)
    assert len(long_resume) > 50_000
    with pytest.raises(HTTPException) as exc_info:
        sanitize_resume_text(long_resume)
    assert exc_info.value.status_code == 422
    assert "too long" in exc_info.value.detail.lower()


def test_null_bytes_stripped():
    """Null bytes embedded in resume text are removed silently."""
    resume_with_nulls = _VALID_RESUME[:50] + "\x00\x00" + _VALID_RESUME[50:]
    result = sanitize_resume_text(resume_with_nulls)
    assert "\x00" not in result
    # Text content should still be intact.
    assert "Jane Doe" in result


def test_excessive_blank_lines_normalized():
    """4+ consecutive newlines are collapsed to 2."""
    resume_with_gaps = _VALID_RESUME.replace("\n\n", "\n\n\n\n\n")
    result = sanitize_resume_text(resume_with_gaps)
    # After normalization there must be no run of 3+ newlines.
    assert "\n\n\n" not in result
