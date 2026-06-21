"""
agents/orchestrator.py
──────────────────────
Master Orchestrator — run_pipeline()

Drives the three-agent pipeline in strict sequential order:
  Agent 1 (Scraper)      → job_insights
  Agent 2 (Resume)       → optimized_resume
  Agent 3 (Cover Letter) → cover_letter

Returns a unified result dict (including pipeline timing) to the router.
"""

import time

from fastapi import HTTPException

from agents.cover_letter_agent import run_cover_letter_agent
from agents.resume_agent import run_resume_agent
from agents.scraper_agent import run_scraper_agent
from core.logger import logger


async def run_pipeline(
    resume_text: str,
    job_url: str,
    user_uid: str,
) -> dict:
    """
    Execute the full 3-agent pipeline and return the consolidated result.

    Args:
        resume_text: Candidate's base resume as plain text.
        job_url:     Publicly accessible job posting URL.
        user_uid:    Firebase UID of the authenticated user.

    Returns:
        {
            "status":                       "success",
            "user_uid":                     str,
            "total_processing_time_seconds": float,
            "job_insights":                 dict,
            "optimized_resume":             str  (Markdown),
            "cover_letter":                 str  (Markdown),
        }

    Raises:
        HTTPException 500: On any unhandled pipeline error not already
                           wrapped by an individual agent.
    """
    logger.info("Pipeline started for UID: {}", user_uid)
    pipeline_start: float = time.perf_counter()

    try:
        # ── Agent 1: Job Scraper ──────────────────────────────────────────────
        job_insights: dict = await run_scraper_agent(job_url)
        logger.info(
            "Agent 1 complete. job_title={!r}  company={!r}",
            job_insights.get("job_title"),
            job_insights.get("company_name"),
        )

        # ── Agent 2: Resume Optimizer ─────────────────────────────────────────
        optimized_resume: str = await run_resume_agent(
            resume_text=resume_text,
            job_insights=job_insights,
        )
        logger.info(
            "Agent 2 complete. optimized_resume length={} chars",
            len(optimized_resume),
        )

        # ── Agent 3: Cover Letter Writer ──────────────────────────────────────
        cover_letter: str = await run_cover_letter_agent(
            resume_text=resume_text,
            job_insights=job_insights,
        )
        logger.info(
            "Agent 3 complete. cover_letter length={} chars",
            len(cover_letter),
        )

    except HTTPException:
        # Re-raise agent HTTPExceptions verbatim — they carry the correct
        # status code and user-facing detail already.
        raise
    except Exception as exc:
        logger.error(
            "Pipeline FAILED for UID: {} — unhandled error: {}",
            user_uid,
            repr(exc),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline encountered an unexpected error: {exc!r}",
        ) from exc

    elapsed: float = round(time.perf_counter() - pipeline_start, 2)
    logger.info(
        "Pipeline complete for UID: {} — total time: {}s",
        user_uid,
        elapsed,
    )

    return {
        "status": "success",
        "user_uid": user_uid,
        "total_processing_time_seconds": elapsed,
        "job_insights": job_insights,
        "optimized_resume": optimized_resume,
        "cover_letter": cover_letter,
    }
