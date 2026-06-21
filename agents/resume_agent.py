"""
agents/resume_agent.py
──────────────────────
Agent 2 — The Resume Optimizer.

Responsibility:
  Accept the candidate's original resume text and the structured job insights
  dict produced by Agent 1, then produce an ATS-optimized resume in Markdown.

Anti-hallucination guarantees (enforced at TWO levels):
  1. System prompt — an explicit HARD RULE instructs the model never to
     invent roles, skills, dates, or experiences not in the original resume.
  2. Post-processing — the returned Markdown is verified for:
     a. Non-emptiness.
     b. Length >= 50% of the input (guards against truncated responses).
     c. Reference-line ratio check (80% of capitalized phrases preserved).
"""

import uuid

from fastapi import HTTPException
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from core.logger import logger

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
    """Construct a fresh LlmAgent for resume optimization."""
    return LlmAgent(
        name="resume_optimizer",
        model="gemini-2.5-flash",
        instruction=_SYSTEM_PROMPT,
    )


def _extract_company_names(resume_text: str) -> list[str]:
    """
    Heuristic: collect capitalised multi-word lines from the original resume
    that are likely company or organisation names.
    """
    candidates: list[str] = []
    for line in resume_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("-", "*", "#", ">")):
            continue
        words = stripped.split()
        cap_words = [w for w in words if w and w[0].isupper()]
        if len(cap_words) >= 2:
            candidates.append(stripped)
    return candidates[:20]


def _validate_no_hallucination(
    original_resume: str, optimized_resume: str
) -> bool:
    """
    Post-processing anti-hallucination guard.

    Returns True if >= 80% of reference lines from the original are still
    present in the optimized output.
    """
    reference_lines = _extract_company_names(original_resume)
    if not reference_lines:
        return True

    hits = sum(
        1 for line in reference_lines if line.lower() in optimized_resume.lower()
    )
    ratio = hits / len(reference_lines)
    logger.debug(
        "Resume hallucination check: {}/{} reference lines found ({:.0f}%).",
        hits,
        len(reference_lines),
        ratio * 100,
    )
    return ratio >= 0.80


async def run_resume_agent(resume_text: str, job_insights: dict) -> str:
    """
    Optimize the candidate's resume for the target job.

    Args:
        resume_text:  The candidate's original resume as plain text.
        job_insights: Structured dict from the Scraper Agent.

    Returns:
        Optimized resume as a Markdown string.

    Raises:
        HTTPException 500: If the ADK runner returns an invalid response.
    """
    logger.info(
        "Agent 2 started — resume length: {} characters", len(resume_text)
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

    # ── Run ADK agent ─────────────────────────────────────────────────────────
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

    logger.info("Agent 2 — sending to Gemini for optimization")

    optimized_resume: str = ""
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_message,
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    optimized_resume = event.content.parts[0].text or ""
                break
    except Exception as exc:
        logger.error("Agent 2 — Gemini runner error: {}", repr(exc))
        raise HTTPException(
            status_code=500,
            detail="Resume optimization produced invalid output. Please retry.",
        ) from exc

    # ── Post-processing validation ────────────────────────────────────────────
    # Check 1: non-empty
    if not optimized_resume.strip():
        logger.error("Agent 2 — Gemini returned empty resume output.")
        raise HTTPException(
            status_code=500,
            detail="Resume optimization produced invalid output. Please retry.",
        )

    # Check 2: output length >= 50% of input length (truncation guard)
    min_expected_length = len(resume_text) * 0.50
    if len(optimized_resume) < min_expected_length:
        logger.error(
            "Agent 2 — output too short ({} chars vs {} input chars). "
            "Possible truncation.",
            len(optimized_resume),
            len(resume_text),
        )
        raise HTTPException(
            status_code=500,
            detail="Resume optimization produced invalid output. Please retry.",
        )

    # Check 3: anti-hallucination reference-line guard
    # HARD RULE (code-level enforcement): if the model substantially rewrote
    # the candidate's identity, fall back to the original resume.
    if not _validate_no_hallucination(resume_text, optimized_resume):
        logger.warning(
            "Agent 2 — hallucination guard triggered. "
            "Returning original resume as safe fallback."
        )
        optimized_resume = (
            "<!-- WARNING: Optimization guard triggered. "
            "Original resume returned. -->\n\n" + resume_text
        )

    logger.info(
        "Agent 2 complete — optimized resume length: {} characters",
        len(optimized_resume),
    )
    return optimized_resume
