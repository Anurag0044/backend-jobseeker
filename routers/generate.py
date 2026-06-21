"""
routers/generate.py
───────────────────
POST /api/v1/generate

Phase 4 upgrade:
  - Accepts multipart/form-data (resume_file + job_url form field)
    instead of a JSON body with raw resume_text.
  - Validates file size (≤ 5 MB), URL length (≤ 500 chars), and
    extracted text length (≤ 50 000 chars).
  - Applies per-user rate limiting (5 requests / hour) via slowapi.
  - Delegates to parse_resume_file() then orchestrator.run_pipeline().
"""

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from agents import orchestrator
from core.file_parser import parse_resume_file
from core.logger import logger
from core.rate_limiter import limiter
from middleware.firebase_auth import verify_firebase_token

router = APIRouter()

# ── Size limits ───────────────────────────────────────────────────────────────
_MAX_FILE_BYTES: int = 5 * 1024 * 1024   # 5 MB
_MAX_URL_CHARS: int = 500
_MAX_RESUME_CHARS: int = 50_000


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
      2. Scrape and analyse the job posting (Agent 1).
      3. Optimize the resume for ATS (Agent 2).
      4. Write a personalised cover letter (Agent 3).

    Returns structured JSON with job_insights, optimized_resume,
    cover_letter, and total_processing_time_seconds.
    """
    uid: str = decoded_token["uid"]
    filename: str = resume_file.filename or "unknown"

    logger.info(
        "Generate endpoint hit — UID: {}, file: {}, url: {}",
        uid,
        filename,
        job_url,
    )

    # ── Input validation: job URL length ──────────────────────────────────────
    if len(job_url) > _MAX_URL_CHARS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Job URL too long (max {_MAX_URL_CHARS} characters). "
                "Please provide a direct link to the job posting."
            ),
        )

    # ── Input validation: file size ───────────────────────────────────────────
    # Read the file once to check size, then seek back to the start so
    # parse_resume_file can read it fresh.
    raw_bytes: bytes = await resume_file.read()
    if len(raw_bytes) > _MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Resume file too large. Maximum size is 5MB.",
        )
    # Seek back so the parser can read from the start.
    await resume_file.seek(0)

    # ── Parse resume file ─────────────────────────────────────────────────────
    resume_text: str = await parse_resume_file(resume_file)

    # ── Input validation: extracted text length ───────────────────────────────
    if len(resume_text) > _MAX_RESUME_CHARS:
        raise HTTPException(
            status_code=413,
            detail=(
                "Resume text too long. Please upload a condensed version."
            ),
        )

    # ── Run the 3-agent pipeline ──────────────────────────────────────────────
    result: dict = await orchestrator.run_pipeline(
        resume_text=resume_text,
        job_url=job_url,
        user_uid=uid,
    )
    return result
