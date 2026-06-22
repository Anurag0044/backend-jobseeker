"""
agents/scraper_agent.py
───────────────────────
Agent 1 — The Job Scraper.

Phase 5 hardening:
  1. 15-second HTTP timeout → 504 on timeout.
  2. bleach.clean() sanitizes raw HTML before BeautifulSoup (prompt injection
     prevention).
  3. tenacity retry (3 attempts, exponential backoff) on Gemini transient
     errors.
  4. Graceful degradation: JSON parse failure returns default job_insights dict
     instead of raising — pipeline continues to produce resume + cover letter.
"""

import json
import uuid
from urllib.parse import urlparse

import bleach
import httpx
from bs4 import BeautifulSoup
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

# ── Retry-eligible Gemini exception types ─────────────────────────────────────
# Import lazily to avoid hard crash if google-api-core is not installed.
try:
    from google.api_core.exceptions import (
        DeadlineExceeded,
        ResourceExhausted,
        ServiceUnavailable,
    )
    _RETRYABLE_GEMINI_EXCEPTIONS = (ServiceUnavailable, DeadlineExceeded, ResourceExhausted)
except ImportError:
    # Fallback: retry on generic Exception subclasses only.
    _RETRYABLE_GEMINI_EXCEPTIONS = (Exception,)  # type: ignore[assignment]

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are a precise job analysis agent. Given raw job posting text, "
    "extract and return ONLY a valid JSON object with this exact schema:\n"
    "{\n"
    '  "job_title": str,\n'
    '  "company_name": str,\n'
    '  "top_5_skills": [str, str, str, str, str],\n'
    '  "experience_level": str,\n'
    '  "company_culture": str,\n'
    '  "key_responsibilities": [str]\n'
    "}\n"
    "Return only the JSON. No explanation. No markdown fences."
)

_APP_NAME = "job_scraper_agent"

_DEFAULT_JOB_INSIGHTS: dict = {
    "job_title": "Unknown",
    "company_name": "Unknown",
    "top_5_skills": [],
    "experience_level": "Unknown",
    "company_culture": "Not available",
    "key_responsibilities": [],
}


def _build_agent() -> LlmAgent:
    return LlmAgent(
        name="job_scraper",
        model="gemini-2.5-flash",
        instruction=_SYSTEM_PROMPT,
    )


def _validate_url(job_url: str) -> None:
    try:
        parsed = urlparse(job_url)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="Invalid job URL. Please provide a full URL including https://",
        ) from exc
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(
            status_code=422,
            detail="Invalid job URL. Please provide a full URL including https://",
        )


def _log_retry_attempt(retry_state: RetryCallState) -> None:
    """tenacity before_sleep callback — logs each retry at WARNING."""
    attempt = retry_state.attempt_number
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Agent 1 — Gemini call failed (attempt {}), retrying… exc={}",
        attempt,
        repr(exc),
    )


async def _call_gemini_agent(visible_text: str, request_id: str) -> str:
    """
    Run the ADK LlmAgent against Gemini and return the raw text output.
    Retried up to 3 times on transient Gemini errors.
    """
    session_service = InMemorySessionService()
    agent = _build_agent()
    runner = Runner(
        agent=agent,
        app_name=_APP_NAME,
        session_service=session_service,
    )
    user_id = "scraper_service"
    session_id = f"scraper-{uuid.uuid4().hex}"

    await session_service.create_session(
        app_name=_APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    user_message = types.Content(
        role="user",
        parts=[
            types.Part(
                text=(
                    "Here is the raw text from a job posting page. "
                    "Extract and return the structured JSON:\n\n"
                    f"{visible_text}"
                )
            )
        ],
    )

    raw_output: str = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_message,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                raw_output = event.content.parts[0].text or ""
            break

    return raw_output


async def run_scraper_agent(job_url: str, request_id: str = "") -> dict:
    """
    Validate URL, fetch job page, sanitize HTML, extract visible text,
    send to Gemini, parse JSON.  Returns default insights on parse failure
    (graceful degradation — pipeline continues).

    Args:
        job_url:    Publicly accessible job posting URL.
        request_id: Trace ID for log correlation.

    Returns:
        Structured job insights dict.

    Raises:
        HTTPException 422: Invalid URL.
        HTTPException 502: HTTP fetch failed.
        HTTPException 504: HTTP fetch timed out.
    """
    logger.info("[{}] Agent 1 started — fetching URL: {}", request_id, job_url)
    _validate_url(job_url)

    # ── Step 1: Fetch raw HTML (15-second timeout) ────────────────────────────
    try:
        async with httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; JobConciergeBot/2.0; "
                    "+https://jobconcierge.app)"
                )
            },
        ) as client:
            response = await client.get(job_url)
            response.raise_for_status()
            raw_html = response.text
    except httpx.TimeoutException as exc:
        logger.error("[{}] Agent 1 — timeout fetching URL: {}", request_id, job_url)
        raise HTTPException(
            status_code=504,
            detail="Job page took too long to respond. Please try again.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        logger.error(
            "[{}] Agent 1 — HTTP {} fetching URL: {}",
            request_id,
            exc.response.status_code,
            job_url,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                f"Failed to fetch job posting: HTTP {exc.response.status_code} "
                f"from {job_url}."
            ),
        ) from exc
    except httpx.RequestError as exc:
        logger.error(
            "[{}] Agent 1 — network error fetching URL: {} — {}",
            request_id,
            job_url,
            repr(exc),
        )
        raise HTTPException(
            status_code=502,
            detail=f"Network error while fetching job posting: {exc!r}",
        ) from exc

    logger.info(
        "[{}] Agent 1 — raw HTML fetched ({} characters)", request_id, len(raw_html)
    )

    # ── Step 2: Sanitize HTML with bleach (prompt injection prevention) ───────
    # bleach.clean with no allowed tags strips ALL HTML tags, leaving only text.
    sanitized_html: str = bleach.clean(
        raw_html,
        tags=[],          # allow zero tags — strip everything
        attributes={},
        strip=True,
    )

    # ── Step 3: Further extract with BeautifulSoup ────────────────────────────
    soup = BeautifulSoup(sanitized_html, "lxml")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    visible_text = soup.get_text(separator="\n", strip=True)
    visible_text = visible_text[:12_000]

    logger.info(
        "[{}] Agent 1 — sending to Gemini for extraction ({} chars of visible text)",
        request_id,
        len(visible_text),
    )

    # ── Step 4: Call Gemini with retry logic ──────────────────────────────────
    @retry(
        retry=retry_if_exception_type(_RETRYABLE_GEMINI_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=_log_retry_attempt,
        reraise=True,
    )
    async def _gemini_with_retry() -> str:
        return await _call_gemini_agent(visible_text, request_id)

    try:
        raw_output: str = await _gemini_with_retry()
    except Exception as exc:
        # Gemini completely unavailable — use defaults and continue pipeline.
        logger.warning(
            "[{}] Agent 1 — Gemini call exhausted all retries ({}). "
            "Using default job insights.",
            request_id,
            repr(exc),
        )
        return dict(_DEFAULT_JOB_INSIGHTS)

    # ── Step 5: Parse JSON ────────────────────────────────────────────────────
    clean_output = raw_output.strip()
    if clean_output.startswith("```"):
        lines = clean_output.splitlines()
        clean_output = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()

    try:
        job_insights: dict = json.loads(clean_output)
    except json.JSONDecodeError:
        # Graceful degradation: parse failure → default insights.
        logger.warning(
            "[{}] Agent 1 JSON parse failed — using defaults. raw={!r}",
            request_id,
            raw_output[:200],
        )
        return dict(_DEFAULT_JOB_INSIGHTS)

    # Fill defaults for any missing keys.
    job_insights.setdefault("job_title", "Unknown")
    job_insights.setdefault("company_name", "Unknown")
    job_insights.setdefault("top_5_skills", [])
    job_insights.setdefault("experience_level", "Unknown")
    job_insights.setdefault("company_culture", "Not specified")
    job_insights.setdefault("key_responsibilities", [])

    logger.info(
        "[{}] Agent 1 complete — job title: {} at {}",
        request_id,
        job_insights.get("job_title"),
        job_insights.get("company_name"),
    )
    return job_insights
