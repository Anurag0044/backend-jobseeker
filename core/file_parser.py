"""
core/file_parser.py
───────────────────
Utility for parsing uploaded resume files (PDF or TXT).

Exports:
  parse_resume_file(file: UploadFile) -> str

Supported types:
  - application/pdf  →  PyPDF2.PdfReader text extraction
  - text/plain       →  UTF-8 decode

Raises:
  HTTPException 415  — unsupported MIME type
  HTTPException 422  — file is empty or too short to be a valid resume
"""

import asyncio
import io
import re

import PyPDF2
from fastapi import HTTPException, UploadFile

from core.logger import logger

# Minimum character count for a resume to be considered valid.
_MIN_RESUME_CHARS: int = 100


def _extract_pdf_text(file_bytes: bytes) -> str:
    """
    Synchronous helper: extract all text from PDF bytes using PyPDF2.
    Runs in a thread pool via asyncio.to_thread to avoid blocking the event
    loop on CPU-bound PDF parsing.

    Args:
        file_bytes: Raw PDF file bytes.

    Returns:
        All page text joined by newlines.

    Raises:
        ValueError: If PyPDF2 cannot read the file.
    """
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)
    return "\n".join(pages)


async def parse_resume_file(file: UploadFile) -> str:
    """
    Parse an uploaded resume file (PDF or TXT) and return clean text.

    Args:
        file: FastAPI UploadFile from the multipart form.

    Returns:
        Extracted and cleaned resume text string.

    Raises:
        HTTPException 415: Unsupported file MIME type.
        HTTPException 422: File is empty, unreadable, or under 100 chars.
    """
    content_type: str = file.content_type or ""
    filename: str = file.filename or "unknown"

    logger.info(
        "File parser: received file '{}' with content_type='{}'",
        filename,
        content_type,
    )

    # ── Read raw bytes ────────────────────────────────────────────────────────
    raw_bytes: bytes = await file.read()

    # ── Route by content type ─────────────────────────────────────────────────
    if content_type == "application/pdf":
        logger.info("File parser: extracting text from PDF ({} bytes).", len(raw_bytes))
        try:
            # Offload CPU-bound PDF parsing to a thread pool.
            extracted_text: str = await asyncio.to_thread(_extract_pdf_text, raw_bytes)
        except Exception as exc:
            logger.error("File parser: PyPDF2 failed to read PDF — {}", exc)
            raise HTTPException(
                status_code=422,
                detail=(
                    "Resume file appears to be empty or unreadable. "
                    "Please upload a valid resume."
                ),
            ) from exc

    elif content_type == "text/plain":
        logger.info("File parser: decoding plain-text file ({} bytes).", len(raw_bytes))
        try:
            extracted_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            logger.error("File parser: UTF-8 decode failed — {}", exc)
            raise HTTPException(
                status_code=422,
                detail=(
                    "Could not read the text file. "
                    "Please ensure it is UTF-8 encoded."
                ),
            ) from exc

    else:
        logger.warning(
            "File parser: rejected unsupported content_type='{}'.", content_type
        )
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Please upload a PDF or TXT file.",
        )

    # ── Clean extracted text ──────────────────────────────────────────────────
    # Collapse runs of 3+ blank lines into a single blank line.
    clean_text: str = re.sub(r"\n{3,}", "\n\n", extracted_text)
    # Collapse runs of horizontal whitespace (but not newlines).
    clean_text = re.sub(r"[ \t]{2,}", " ", clean_text)
    clean_text = clean_text.strip()

    # ── Validate minimum length ───────────────────────────────────────────────
    if len(clean_text) < _MIN_RESUME_CHARS:
        logger.warning(
            "File parser: extracted text too short ({} chars) — rejecting.",
            len(clean_text),
        )
        raise HTTPException(
            status_code=422,
            detail=(
                "Resume file appears to be empty or unreadable. "
                "Please upload a valid resume."
            ),
        )

    logger.info(
        "File parser: successfully extracted {} characters from '{}'.",
        len(clean_text),
        filename,
    )
    return clean_text


__all__ = ["parse_resume_file"]
