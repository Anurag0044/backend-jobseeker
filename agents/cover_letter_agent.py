"""
agents/cover_letter_agent.py
────────────────────────────
Agent 3 — The Cover Letter Writer.

Responsibility:
  Accept the candidate's resume text and structured job insights from Agent 1,
  then produce a compelling, personalised cover letter in Markdown.

Post-processing validation:
  - Under 200 words → raises HTTPException 500.
  - Over 600 words  → truncated at the last sentence boundary before the
                       600-word mark and a WARNING is logged.
"""

import re
import uuid

from fastapi import HTTPException
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from core.logger import logger

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


def _count_words(text: str) -> int:
    """Return the approximate word count of a string."""
    return len(text.split())


def _truncate_at_word_boundary(text: str, max_words: int) -> str:
    """
    Truncate `text` to at most `max_words` words, ending at the last
    sentence boundary (period, exclamation mark, or question mark)
    within that word limit.

    If no sentence boundary is found, truncate at the word boundary.
    """
    words = text.split()
    if len(words) <= max_words:
        return text

    # Join the first max_words words, then find the last sentence end.
    candidate = " ".join(words[:max_words])
    # Find the last sentence-ending punctuation.
    match = re.search(r"[.!?][^.!?]*$", candidate)
    if match:
        # Keep up to and including the punctuation mark.
        return candidate[: match.start() + 1]
    return candidate


def _build_agent() -> LlmAgent:
    """Construct a fresh LlmAgent for cover letter writing."""
    return LlmAgent(
        name="cover_letter_writer",
        model="gemini-2.5-flash",
        instruction=_SYSTEM_PROMPT,
    )


async def run_cover_letter_agent(resume_text: str, job_insights: dict) -> str:
    """
    Write a personalised cover letter for the target job.

    Args:
        resume_text:  The candidate's original resume as plain text.
        job_insights: Structured dict from the Scraper Agent.

    Returns:
        Cover letter as a Markdown string (200–600 words; truncated if over).

    Raises:
        HTTPException 500: If the cover letter is under 200 words or empty.
    """
    logger.info("Agent 3 started — generating cover letter")

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

    # ── Run ADK agent ─────────────────────────────────────────────────────────
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

    logger.info("Agent 3 — sending to Gemini")

    cover_letter: str = ""
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_message,
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    cover_letter = event.content.parts[0].text or ""
                break
    except Exception as exc:
        logger.error("Agent 3 — Gemini runner error: {}", repr(exc))
        raise HTTPException(
            status_code=500,
            detail="Cover letter generation failed unexpectedly. Please retry.",
        ) from exc

    # ── Post-processing validation ────────────────────────────────────────────
    word_count = _count_words(cover_letter)

    if word_count < _MIN_WORDS:
        logger.error(
            "Agent 3 — cover letter too short ({} words, minimum {}).",
            word_count,
            _MIN_WORDS,
        )
        raise HTTPException(
            status_code=500,
            detail="Cover letter too short. Please retry.",
        )

    if word_count > _MAX_WORDS:
        logger.warning(
            "Agent 3 — cover letter exceeds {} words ({} words). "
            "Truncating at sentence boundary.",
            _MAX_WORDS,
            word_count,
        )
        cover_letter = _truncate_at_word_boundary(cover_letter, _MAX_WORDS)

    logger.info(
        "Agent 3 complete — cover letter length: {} characters",
        len(cover_letter),
    )
    return cover_letter
