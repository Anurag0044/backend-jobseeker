"""
agents/cover_letter_agent.py
────────────────────────────
Agent 3 — The Cover Letter Writer.

Phase 5 hardening:
  1. tenacity retry (3 attempts, exponential backoff) on Gemini transient errors.
  2. validate_cover_letter(): word-count validation with one retry on short
     output before raising 500.  Over 600 words → truncate at sentence boundary.
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
    "You are an expert career coach and professional writer. Using the "
    "candidate's resume and the structured job insights provided, write "
    "a compelling, personalized cover letter.\n\n"
    "Structure (3 paragraphs):\n"
    "Paragraph 1 — Hook: Open with a confident, specific statement about "
    "why this candidate is an excellent fit for the role at this company. "
    "Reference the company name and job title directly.\n\n"
    "Paragraph 2 — Evidence: Highlight 2-3 specific, quantifiable "
    "achievements from the resume that directly map to the top skills "
    "or responsibilities of the job.\n\n"
    "Paragraph 3 — Close: Express genuine enthusiasm for the company "
    "culture and mission (use the company_culture field). End with a "
    "clear call to action requesting an interview.\n\n"
    "Tone: Professional, confident, human. Not generic. Not robotic.\n"
    "Length: 250-350 words.\n"
    "Format: Return in clean Markdown."
)

_APP_NAME = "cover_letter_agent"
_MIN_WORDS = 200
_MAX_WORDS = 600


def _log_retry(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Agent 3 — Gemini call failed (attempt {}), retrying… exc={}",
        retry_state.attempt_number,
        repr(exc),
    )


def _count_words(text: str) -> int:
    return len(text.split())


def _truncate_at_sentence_boundary(text: str, max_words: int) -> str:
    """
    Truncate `text` to at most `max_words` words, ending at the last
    sentence-ending punctuation within that word limit.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    candidate = " ".join(words[:max_words])
    match = re.search(r"[.!?][^.!?]*$", candidate)
    if match:
        return candidate[: match.start() + 1]
    return candidate


def _build_agent() -> LlmAgent:
    return LlmAgent(
        name="cover_letter_writer",
        model="gemini-2.5-flash",
        instruction=_SYSTEM_PROMPT,
    )


async def _run_agent_once(user_prompt: str) -> str:
    """Single ADK agent invocation."""
    session_service = InMemorySessionService()
    agent = _build_agent()
    runner = Runner(
        agent=agent,
        app_name=_APP_NAME,
        session_service=session_service,
    )
    user_id = "cover_letter_service"
    session_id = f"cover-{uuid.uuid4().hex}"

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


async def validate_cover_letter(
    text: str,
    user_prompt: str,
    request_id: str = "",
) -> str:
    """
    Validate word count; retry once if under 200 words.
    Truncate at sentence boundary if over 600 words.

    Args:
        text:        Cover letter text from Gemini.
        user_prompt: Original prompt — used for the retry call.
        request_id:  Trace ID.

    Returns:
        Validated (and possibly truncated) cover letter string.

    Raises:
        HTTPException 500: If under 200 words even after one retry.
    """
    word_count = _count_words(text)

    if word_count < _MIN_WORDS:
        logger.warning(
            "[{}] Agent 3 — cover letter too short ({} words) — retrying once.",
            request_id,
            word_count,
        )
        try:
            text = await _run_agent_once(user_prompt)
            word_count = _count_words(text)
        except Exception as exc:
            logger.error(
                "[{}] Agent 3 — retry failed: {}", request_id, repr(exc)
            )

        if word_count < _MIN_WORDS:
            logger.error(
                "[{}] Agent 3 — cover letter still too short ({} words) after retry.",
                request_id,
                word_count,
            )
            raise HTTPException(
                status_code=500,
                detail="Cover letter generation failed. Please retry.",
            )

    if word_count > _MAX_WORDS:
        logger.warning(
            "[{}] Agent 3 — cover letter {} words, truncating to {}.",
            request_id,
            word_count,
            _MAX_WORDS,
        )
        text = _truncate_at_sentence_boundary(text, _MAX_WORDS)
        word_count = _count_words(text)

    logger.info(
        "[{}] Agent 3 — cover letter validated: {} words.", request_id, word_count
    )
    return text


async def run_cover_letter_agent(
    resume_text: str,
    job_insights: dict,
    request_id: str = "",
) -> str:
    """
    Write a personalised cover letter for the target job.

    Args:
        resume_text:  Candidate's original resume (sanitized plain text).
        job_insights: Structured dict from Agent 1.
        request_id:   Trace ID for log correlation.

    Returns:
        Cover letter as a Markdown string (200–600 words).

    Raises:
        HTTPException 500: On persistent failure or word-count violations.
    """
    logger.info("[{}] Agent 3 started — generating cover letter", request_id)

    job_insights_formatted = "\n".join(
        f"- {key}: {value}" for key, value in job_insights.items()
    )

    user_prompt = (
        "## Candidate's Resume\n\n"
        f"{resume_text}\n\n"
        "---\n\n"
        "## Target Job Insights\n\n"
        f"{job_insights_formatted}\n\n"
        "---\n\n"
        "Please write the cover letter now, following the 3-paragraph "
        "structure from your instructions. Use the company name and job "
        "title directly. Keep it between 250 and 350 words. Return clean "
        "Markdown only."
    )

    logger.info("[{}] Agent 3 — sending to Gemini", request_id)

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
        cover_letter: str = await _gemini_with_retry()
    except Exception as exc:
        logger.error("[{}] Agent 3 — Gemini runner error: {}", request_id, repr(exc))
        raise HTTPException(
            status_code=500,
            detail="Cover letter generation failed unexpectedly. Please retry.",
        ) from exc

    cover_letter = await validate_cover_letter(cover_letter, user_prompt, request_id)

    logger.info(
        "[{}] Agent 3 complete — cover letter length: {} characters",
        request_id,
        len(cover_letter),
    )
    return cover_letter
