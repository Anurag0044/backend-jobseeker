"""
routers/generate.py
───────────────────
POST /api/v1/generate

Phase 5 upgrades:
  - Calls sanitize_job_url() and sanitize_resume_text() on all inputs.
  - Extracts request_id from request.state and passes it to the pipeline.
  - Logs entry and success with request_id and elapsed time.
"""

import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from agents import orchestrator
from core.file_parser import parse_resume_file
from core.input_sanitizer import sanitize_job_url, sanitize_resume_text
from core.logger import logger
from core.rate_limiter import limiter
from middleware.firebase_auth import verify_firebase_token

router = APIRouter()

_MAX_FILE_BYTES: int = 5 * 1024 * 1024   # 5 MB


@router.post("/generate")
@limiter.limit("5/hour")
async def generate(
    request: Request,
    resume_file: UploadFile = File(
        ...,
        description="Resume file (PDF or TXT, maximum 5 MB).",
    ),
    job_url: str = Form(
        ...,
        description="Full URL of the job posting (must start with https://).",
    ),
    decoded_token: dict = Depends(verify_firebase_token),
) -> dict:
    """
    Run the full 3-agent pipeline:
      1. Parse the uploaded resume file to plain text.
      2. Sanitize URL and resume text inputs.
      3. Scrape and analyse the job posting (Agent 1).
      4. Optimize the resume for ATS (Agent 2).
      5. Write a personalised cover letter (Agent 3).

    Returns structured JSON with job_insights, optimized_resume,
    cover_letter, request_id, and total_processing_time_seconds.
    """
    uid: str = decoded_token["uid"]
    filename: str = resume_file.filename or "unknown"
    request_id: str = getattr(request.state, "request_id", "unknown")
    start: float = time.perf_counter()

    # ── Input validation: file size ───────────────────────────────────────────
    raw_bytes: bytes = await resume_file.read()
    file_size_kb = round(len(raw_bytes) / 1024, 1)
    if len(raw_bytes) > _MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Resume file too large. Maximum size is 5MB.",
        )
    await resume_file.seek(0)

    logger.info(
        "[{}] Generate hit — UID: {} file: {} size: {}kb url: {}",
        request_id,
        uid,
        filename,
        file_size_kb,
        job_url,
    )

    # ── Sanitize URL first (fast fail before file parsing) ───────────────────
    clean_url: str = sanitize_job_url(job_url)

    # ── Parse resume file ─────────────────────────────────────────────────────
    resume_text: str = await parse_resume_file(resume_file)

    # ── Sanitize resume text ──────────────────────────────────────────────────
    resume_text = sanitize_resume_text(resume_text)

    # ── Run the 3-agent pipeline ──────────────────────────────────────────────
    result: dict = await orchestrator.run_pipeline(
        resume_text=resume_text,
        job_url=clean_url,
        user_uid=uid,
        request_id=request_id,
    )

    elapsed: float = round(time.perf_counter() - start, 2)
    logger.info(
        "[{}] Generate complete — UID: {} time: {}s",
        request_id,
        uid,
        elapsed,
    )
    return result
