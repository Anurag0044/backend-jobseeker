"""
agents/scraper_agent.py
───────────────────────
Agent 1 — The Job Scraper.

Responsibility:
  1. Validate the job URL scheme.
  2. Fetch the raw HTML of the job posting URL with httpx.
  3. Extract visible text using BeautifulSoup.
  4. Send that text to a Gemini-powered Google ADK agent with a precise
     extraction system prompt.
  5. Parse and return the structured JSON dict.

Raises:
  HTTPException 422  — invalid URL scheme.
  HTTPException 502  — URL fetch failed (network / DNS / 4xx-5xx).
  HTTPException 422  — Gemini response could not be parsed as valid JSON.
"""

import json
import uuid
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import HTTPException
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from core.logger import logger

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


def _build_agent() -> LlmAgent:
    """Construct a fresh LlmAgent for job scraping."""
    return LlmAgent(
        name="job_scraper",
        model="gemini-2.5-flash",
        instruction=_SYSTEM_PROMPT,
    )


def _validate_url(job_url: str) -> None:
    """
    Validate that job_url has a proper http/https scheme and a netloc.

    Raises:
        HTTPException 422: If the URL is malformed or uses an invalid scheme.
    """
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


async def run_scraper_agent(job_url: str) -> dict:
    """
    Validate URL, fetch a job posting page, extract its text, and return
    structured job insights as a Python dict.

    Args:
        job_url: Publicly accessible job posting URL.

    Returns:
        Structured dict matching the schema defined in the system prompt.

    Raises:
        HTTPException 422: If the URL is invalid or JSON parsing fails.
        HTTPException 502: If the HTTP fetch fails.
    """
    # ── Step 0: Validate URL ──────────────────────────────────────────────────
    logger.info("Agent 1 started — fetching URL: {}", job_url)
    _validate_url(job_url)

    # ── Step 1: Fetch raw HTML ────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(
            timeout=30.0,
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
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Agent 1 — HTTP error fetching URL: {} — status {}",
            job_url,
            exc.response.status_code,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                f"Failed to fetch job posting: HTTP {exc.response.status_code} "
                f"from {job_url}."
            ),
        ) from exc
    except httpx.RequestError as exc:
        logger.error("Agent 1 — network error fetching URL: {} — {}", job_url, repr(exc))
        raise HTTPException(
            status_code=502,
            detail=f"Network error while fetching job posting: {exc!r}",
        ) from exc

    logger.info(
        "Agent 1 — raw HTML fetched ({} characters)", len(raw_html)
    )

    # ── Step 2: Extract visible text with BeautifulSoup ──────────────────────
    soup = BeautifulSoup(raw_html, "lxml")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    visible_text = soup.get_text(separator="\n", strip=True)
    # Truncate to ~12 000 chars to stay within context limits.
    visible_text = visible_text[:12_000]

    logger.info(
        "Agent 1 — sending to Gemini for extraction ({} chars of visible text)",
        len(visible_text),
    )

    # ── Step 3: Run ADK agent ─────────────────────────────────────────────────
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

    # ── Step 4: Parse and validate JSON ──────────────────────────────────────
    clean_output = raw_output.strip()
    if clean_output.startswith("```"):
        lines = clean_output.splitlines()
        clean_output = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()

    try:
        job_insights: dict = json.loads(clean_output)
    except json.JSONDecodeError as exc:
        logger.error(
            "Agent 1 — JSON parse failed. raw_output={!r} error={}",
            raw_output[:200],
            exc,
        )
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Scraper Agent failed to return valid JSON.",
                "json_error": str(exc),
                "raw_output": raw_output,
            },
        ) from exc

    # Fill defaults for any missing keys.
    job_insights.setdefault("job_title", "Unknown")
    job_insights.setdefault("company_name", "Unknown")
    job_insights.setdefault("top_5_skills", [])
    job_insights.setdefault("experience_level", "Unknown")
    job_insights.setdefault("company_culture", "Not specified")
    job_insights.setdefault("key_responsibilities", [])

    logger.info(
        "Agent 1 complete — job title: {} at {}",
        job_insights.get("job_title"),
        job_insights.get("company_name"),
    )
    return job_insights
