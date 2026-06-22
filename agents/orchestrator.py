"""
agents/orchestrator.py
──────────────────────
Master Orchestrator — run_pipeline()

Phase 5 upgrades:
  1. Accepts request_id parameter; threads it through every log line and
     every agent call so the full request can be traced in logs.
  2. Per-agent try/except:
     - Agent 1 failure → uses default job_insights, pipeline continues.
     - Agent 2 failure → raises HTTPException 500 with agent context.
     - Agent 3 failure → raises HTTPException 500 with agent context.
"""

import time

from fastapi import HTTPException

from agents.cover_letter_agent import run_cover_letter_agent
from agents.resume_agent import run_resume_agent
from agents.scraper_agent import _DEFAULT_JOB_INSIGHTS, run_scraper_agent
from core.logger import logger


async def run_pipeline(
    resume_text: str,
    job_url: str,
    user_uid: str,
    request_id: str = "",
) -> dict:
    """
    Execute the full 3-agent pipeline and return the consolidated result.

    Agent 1 failure → graceful degradation (default insights, pipeline
    continues).  Agent 2 or 3 failure → HTTPException 500 with context.

    Args:
        resume_text: Candidate's base resume (already sanitized).
        job_url:     Publicly accessible job posting URL.
        user_uid:    Firebase UID of the authenticated user.
        request_id:  Trace ID for log correlation.

    Returns:
        {
            "status":                        "success",
            "user_uid":                      str,
            "request_id":                    str,
            "total_processing_time_seconds": float,
            "job_insights":                  dict,
            "optimized_resume":              str  (Markdown),
            "cover_letter":                  str  (Markdown),
        }
    """
    logger.info("[{}] Pipeline started for UID: {}", request_id, user_uid)
    pipeline_start: float = time.perf_counter()

    # ── Agent 1: Job Scraper ──────────────────────────────────────────────────
    try:
        job_insights: dict = await run_scraper_agent(
            job_url=job_url,
            request_id=request_id,
        )
        logger.info(
            "[{}] Agent 1 complete. job_title={!r}  company={!r}",
            request_id,
            job_insights.get("job_title"),
            job_insights.get("company_name"),
        )
    except HTTPException as exc:
        # Network / URL fetch errors (502, 504, 422) should propagate.
        if exc.status_code in (502, 504, 422):
            raise
        # Any other Agent 1 HTTP error → degrade gracefully.
        logger.warning(
            "[{}] Agent 1 raised HTTP {} — falling back to default job insights.",
            request_id,
            exc.status_code,
        )
        job_insights = dict(_DEFAULT_JOB_INSIGHTS)
    except Exception as exc:
        logger.warning(
            "[{}] Agent 1 unexpected error: {} — using default job insights.",
            request_id,
            repr(exc),
        )
        job_insights = dict(_DEFAULT_JOB_INSIGHTS)

    # ── Agent 2: Resume Optimizer ─────────────────────────────────────────────
    try:
        optimized_resume: str = await run_resume_agent(
            resume_text=resume_text,
            job_insights=job_insights,
            request_id=request_id,
        )
        logger.info(
            "[{}] Agent 2 complete. optimized_resume length={} chars",
            request_id,
            len(optimized_resume),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "[{}] Agent 2 unexpected error: {}", request_id, repr(exc)
        )
        raise HTTPException(
            status_code=500,
            detail=(
                f"Resume optimization failed unexpectedly "
                f"(request_id={request_id}). Please retry."
            ),
        ) from exc

    # ── Agent 3: Cover Letter Writer ──────────────────────────────────────────
    try:
        cover_letter: str = await run_cover_letter_agent(
            resume_text=resume_text,
            job_insights=job_insights,
            request_id=request_id,
        )
        logger.info(
            "[{}] Agent 3 complete. cover_letter length={} chars",
            request_id,
            len(cover_letter),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "[{}] Agent 3 unexpected error: {}", request_id, repr(exc)
        )
        raise HTTPException(
            status_code=500,
            detail=(
                f"Cover letter generation failed unexpectedly "
                f"(request_id={request_id}). Please retry."
            ),
        ) from exc

    elapsed: float = round(time.perf_counter() - pipeline_start, 2)
    logger.info(
        "[{}] Pipeline complete for UID: {} — total time: {}s",
        request_id,
        user_uid,
        elapsed,
    )

    return {
        "status": "success",
        "user_uid": user_uid,
        "request_id": request_id,
        "total_processing_time_seconds": elapsed,
        "job_insights": job_insights,
        "optimized_resume": optimized_resume,
        "cover_letter": cover_letter,
    }
