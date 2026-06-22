"""
tests/test_hallucination_guard.py
──────────────────────────────────
Unit tests for the validate_no_hallucination() function in resume_agent.py
and the length-guard / retry logic in run_resume_agent().

Tests:
  1. Resume with same years as original passes validation.
  2. Resume with a fabricated year not in original raises 500.
  3. Optimized resume shorter than 50% of original triggers retry path.
  4. Empty optimized resume raises 500.
"""

import pytest
from unittest.mock import patch

from fastapi import HTTPException

from agents.resume_agent import run_resume_agent, validate_no_hallucination


# ── validate_no_hallucination unit tests ──────────────────────────────────────

def test_same_dates_pass_validation():
    """
    Optimized resume that contains only years present in the original
    must pass without raising.
    """
    original = (
        "Jane Doe\n"
        "Software Engineer - Acme Corp (2021-Present)\n"
        "Junior Developer - StartupXYZ (2019-2021)\n"
        "B.Sc. Computer Science - State University (2019)\n"
    )
    optimized = (
        "# Jane Doe\n\n"
        "## Experience\n"
        "**Software Engineer** - Acme Corp (2021-Present)\n"
        "- Built production APIs\n\n"
        "**Junior Developer** - StartupXYZ (2019-2021)\n"
        "- Developed React components\n\n"
        "## Education\n"
        "B.Sc. Computer Science - State University (2019)\n"
    )
    # Should not raise.
    result = validate_no_hallucination(original, optimized, request_id="test-001")
    assert result == optimized


def test_fabricated_date_raises_500():
    """
    Optimized resume containing a 4-digit year that does NOT appear in the
    original resume must raise HTTPException 500.
    """
    original = (
        "Jane Doe\n"
        "Software Engineer - Acme Corp (2021-Present)\n"
        "B.Sc. Computer Science - State University (2019)\n"
    )
    # Introduces fabricated year 2015 which is not in original.
    optimized = (
        "# Jane Doe\n\n"
        "Software Engineer - Acme Corp (2021-Present)\n"
        "Previously at Acme Corp (2015-2019)\n"  # hallucinated
        "B.Sc. Computer Science - State University (2019)\n"
    )
    with pytest.raises(HTTPException) as exc_info:
        validate_no_hallucination(original, optimized, request_id="test-002")
    assert exc_info.value.status_code == 500
    assert "inconsistent" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_short_output_triggers_retry():
    """
    If the first Gemini call returns text shorter than 50% of the original,
    run_resume_agent() must attempt a retry call.
    The retry call returns valid output; the final result is the retry output.
    """
    original_resume = "A" * 400   # 400-char original
    short_output = "B" * 50      # 50 chars < 50% of 400 (200 min)
    good_output = "C" * 300      # 300 chars > 50% threshold

    job_insights = {
        "job_title": "Engineer",
        "company_name": "Acme",
        "top_5_skills": [],
        "experience_level": "Mid",
        "company_culture": "Fast-paced",
        "key_responsibilities": [],
    }

    call_count = {"n": 0}

    async def _mock_run_once(prompt: str) -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return short_output
        return good_output

    with patch("agents.resume_agent._run_agent_once", side_effect=_mock_run_once):
        result = await run_resume_agent(
            resume_text=original_resume,
            job_insights=job_insights,
            request_id="test-003",
        )

    # Should have called Gemini twice (first call short, retry call good).
    assert call_count["n"] == 2
    assert result == good_output


@pytest.mark.asyncio
async def test_empty_optimized_resume_raises_500():
    """
    run_resume_agent() must raise HTTPException 500 if Gemini returns
    an empty string.
    """
    job_insights = {
        "job_title": "Engineer",
        "company_name": "Acme",
        "top_5_skills": [],
        "experience_level": "Mid",
        "company_culture": "Fast-paced",
        "key_responsibilities": [],
    }

    with patch("agents.resume_agent._run_agent_once", return_value=""):
        with pytest.raises(HTTPException) as exc_info:
            await run_resume_agent(
                resume_text="A" * 200,
                job_insights=job_insights,
                request_id="test-004",
            )

    assert exc_info.value.status_code == 500
    assert "invalid output" in exc_info.value.detail.lower()
