"""
agents/resume_agent.py
──────────────────────
Agent 2 — The Resume Optimizer.

Phase 5 hardening:
  1. tenacity retry (3 attempts, exponential backoff) on Gemini transient errors.
  2. validate_no_hallucination() — date-level and proper-noun-level checks:
     - Any 4-digit year in optimized output not present in original → raises 500.
     - > 3 new capitalised proper nouns → logs WARNING (does not raise).
  3. Output length guard: if < 50% of original length, retry once before raising.
"""

import re
import uuid

from fastapi import HTTPException
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.logger import logger

try:
    from google.api_core.exceptions import (
        DeadlineExceeded,
        ResourceExhausted,
        ServiceUnavailable,
    )
    _RETRYABLE = (ServiceUnavailable, DeadlineExceeded, ResourceExhausted)
except ImportError:
    _RETRYABLE = (Exception,)  # type: ignore[assignment]

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are a professional resume optimization specialist and ATS expert. "
    "You will be given a candidate's base resume and structured insights "
    "from a target job posting.\n\n"
    "Your task:\n"
    "1. Rewrite the resume bullet points to naturally incorporate the "
    "top 5 skills and key responsibilities from the job insights.\n"
    "2. Preserve the EXACT structure, timeline, job titles, company names, "
    "and dates from the original resume.\n"
    "3. NEVER invent, add, or hallucinate any new roles, skills, dates, "
    "certifications, or experiences that are not already present in "
    "the original resume.\n"
    "4. Return the complete optimized resume in clean Markdown format.\n\n"
    "HARD RULE: If a skill from the job posting does not exist in any "
    "form in the original resume, do NOT add it. Only rephrase and "
    "emphasize what is genuinely there."
)

_APP_NAME = "resume_optimizer_agent"


def _build_agent() -> LlmAgent:
    return LlmAgent(
        name="resume_optimizer",
        model="gemini-2.5-flash",
        instruction=_SYSTEM_PROMPT,
    )


def _log_retry(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Agent 2 — Gemini call failed (attempt {}), retrying… exc={}",
        retry_state.attempt_number,
        repr(exc),
    )


def validate_no_hallucination(
    original: str,
    optimized: str,
    request_id: str = "",
) -> str:
    """
    Post-processing hallucination guard for the optimized resume.

    Checks:
      1. Any 4-digit year (e.g. 2019, 2023) present in `optimized` but NOT
         in `original` is a fabricated date → raises HTTPException 500.
      2. More than 3 new capitalised proper nouns (4+ chars) in `optimized`
         that are not in `original` → logs WARNING (does not raise, as some
         rephrasing is expected).

    Args:
        original:   The candidate's original resume text.
        optimized:  The Gemini-generated optimized resume text.
        request_id: Trace ID for log correlation.

    Returns:
        The validated optimized resume string (unchanged if it passes).

    Raises:
        HTTPException 500: If fabricated dates are detected.
    """
    # ── Check 1: Date hallucination ───────────────────────────────────────────
    original_years: set[str] = set(re.findall(r"\b(?:19|20)\d{2}\b", original))
    optimized_years: set[str] = set(re.findall(r"\b(?:19|20)\d{2}\b", optimized))
    new_years: set[str] = optimized_years - original_years

    if new_years:
        logger.error(
            "[{}] Hallucination detected — new dates found: {}",
            request_id,
            sorted(new_years),
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "Resume optimization produced inconsistent output. Please retry."
            ),
        )

    # ── Check 2: Proper noun proliferation ────────────────────────────────────
    def _proper_nouns(text: str) -> set[str]:
        return {
            w for w in re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", text)
        }

    original_nouns = _proper_nouns(original)
    optimized_nouns = _proper_nouns(optimized)
    new_nouns = optimized_nouns - original_nouns

    if len(new_nouns) > 3:
        logger.warning(
            "[{}] Agent 2 — {} new proper nouns in optimized resume: {}",
            request_id,
            len(new_nouns),
            sorted(new_nouns)[:10],
        )

    return optimized


async def _run_agent_once(user_prompt: str) -> str:
    """Single ADK agent invocation — used by both first call and length retry."""
    session_service = InMemorySessionService()
    agent = _build_agent()
    runner = Runner(
        agent=agent,
        app_name=_APP_NAME,
        session_service=session_service,
    )
    user_id = "resume_service"
    session_id = f"resume-{uuid.uuid4().hex}"

    await session_service.create_session(
        app_name=_APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    user_message = types.Content(
        role="user",
        parts=[types.Part(text=user_prompt)],
    )

    result: str = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_message,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                result = event.content.parts[0].text or ""
            break
    return result


async def run_resume_agent(
    resume_text: str,
    job_insights: dict,
    request_id: str = "",
) -> str:
    """
    Optimize the candidate's resume for the target job.

    Args:
        resume_text:  Candidate's original resume (sanitized plain text).
        job_insights: Structured dict from Agent 1.
        request_id:   Trace ID for log correlation.

    Returns:
        Optimized resume as a Markdown string.

    Raises:
        HTTPException 500: On empty output, suspicious truncation, or
                           hallucinated dates after all retries.
    """
    logger.info(
        "[{}] Agent 2 started — resume length: {} characters",
        request_id,
        len(resume_text),
    )

    job_insights_formatted = "\n".join(
        f"- {key}: {value}" for key, value in job_insights.items()
    )

    user_prompt = (
        "## Candidate's Original Resume\n\n"
        f"{resume_text}\n\n"
        "---\n\n"
        "## Target Job Insights\n\n"
        f"{job_insights_formatted}\n\n"
        "---\n\n"
        "Please produce the ATS-optimized resume in clean Markdown format. "
        "Remember: only rephrase and emphasize what is genuinely in the "
        "original resume. Do NOT add anything that isn't already there."
    )

    logger.info("[{}] Agent 2 — sending to Gemini for optimization", request_id)

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=_log_retry,
        reraise=True,
    )
    async def _gemini_with_retry() -> str:
        return await _run_agent_once(user_prompt)

    try:
        optimized_resume: str = await _gemini_with_retry()
    except Exception as exc:
        logger.error("[{}] Agent 2 — Gemini runner error: {}", request_id, repr(exc))
        raise HTTPException(
            status_code=500,
            detail="Resume optimization produced invalid output. Please retry.",
        ) from exc

    # ── Check 1: non-empty ────────────────────────────────────────────────────
    if not optimized_resume.strip():
        logger.error("[{}] Agent 2 — Gemini returned empty resume output.", request_id)
        raise HTTPException(
            status_code=500,
            detail="Resume optimization produced invalid output. Please retry.",
        )

    # ── Check 2: length >= 50% of input (truncation guard with one retry) ─────
    min_expected = len(resume_text) * 0.50
    if len(optimized_resume) < min_expected:
        logger.warning(
            "[{}] Resume output suspiciously short ({} chars vs {} input) "
            "— retrying once",
            request_id,
            len(optimized_resume),
            len(resume_text),
        )
        try:
            optimized_resume = await _run_agent_once(user_prompt)
        except Exception as exc:
            logger.error(
                "[{}] Agent 2 — retry attempt also failed: {}", request_id, repr(exc)
            )

        if not optimized_resume.strip() or len(optimized_resume) < min_expected:
            logger.error(
                "[{}] Agent 2 — output still too short after retry ({} chars).",
                request_id,
                len(optimized_resume),
            )
            raise HTTPException(
                status_code=500,
                detail="Resume optimization produced invalid output. Please retry.",
            )

    # ── Check 3: hallucination guard ──────────────────────────────────────────
    optimized_resume = validate_no_hallucination(
        original=resume_text,
        optimized=optimized_resume,
        request_id=request_id,
    )

    logger.info(
        "[{}] Agent 2 complete — optimized resume length: {} characters",
        request_id,
        len(optimized_resume),
    )
    return optimized_resume
