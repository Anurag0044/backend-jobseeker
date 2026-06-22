"""
core/input_sanitizer.py
───────────────────────
Input sanitization utilities for job URLs and resume text.

Functions:
    sanitize_job_url(url: str) -> str
    sanitize_resume_text(text: str) -> str

Both functions raise HTTPException 422 on invalid input with a descriptive
detail message so the client knows exactly what to fix.
"""

import re
import unicodedata
from urllib.parse import urlparse

from fastapi import HTTPException

# ── Domain block-list — known non-job-page domains that waste API quota ───────
_BLOCKED_DOMAINS: frozenset[str] = frozenset(
    [
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "example.com",
    ]
)

_MAX_URL_CHARS: int = 500
_MIN_RESUME_CHARS: int = 100
_MAX_RESUME_CHARS: int = 50_000


def sanitize_job_url(url: str) -> str:
    """
    Sanitize and validate a job posting URL.

    Steps:
      1. Strip leading/trailing whitespace.
      2. Verify scheme is http or https.
      3. Verify netloc is not empty.
      4. Verify total length is under 500 characters.
      5. Block known non-job-page domains.

    Args:
        url: Raw job URL string from the form field.

    Returns:
        Clean URL string (whitespace stripped).

    Raises:
        HTTPException 422: On any validation failure with a descriptive detail.
    """
    url = url.strip()

    if not url:
        raise HTTPException(
            status_code=422,
            detail="Job URL is required.",
        )

    if len(url) > _MAX_URL_CHARS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Job URL is too long ({len(url)} characters). "
                f"Maximum allowed is {_MAX_URL_CHARS} characters."
            ),
        )

    try:
        parsed = urlparse(url)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Job URL could not be parsed. Please provide a valid URL.",
        )

    if parsed.scheme not in ("http", "https"):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid URL scheme '{parsed.scheme}'. "
                "Please provide a full URL starting with https://"
            ),
        )

    if not parsed.netloc:
        raise HTTPException(
            status_code=422,
            detail="Invalid job URL. Please provide a full URL including https://",
        )

    # Extract bare hostname without port for block-list check.
    hostname = parsed.hostname or ""
    if hostname in _BLOCKED_DOMAINS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"The domain '{hostname}' is not allowed. "
                "Please provide a public job posting URL."
            ),
        )

    return url


def sanitize_resume_text(text: str) -> str:
    """
    Sanitize extracted resume text before passing it to agents.

    Steps:
      1. Strip leading/trailing whitespace.
      2. Remove null bytes and non-printable control characters
         (except standard whitespace: space, tab, newline, carriage return).
      3. Normalize multiple consecutive blank lines to maximum 2.
      4. Verify length is between 100 and 50 000 characters.

    Args:
        text: Extracted resume text from the uploaded file.

    Returns:
        Cleaned resume text string.

    Raises:
        HTTPException 422: If the text is too short or too long after cleaning.
    """
    text = text.strip()

    # Remove null bytes and C0/C1 control characters except standard whitespace.
    # unicodedata.category 'Cc' is "control character".
    allowed_whitespace = {"\n", "\r", "\t", " "}

    def _is_printable(ch: str) -> bool:
        if ch in allowed_whitespace:
            return True
        cat = unicodedata.category(ch)
        # Reject all control characters (Cc) and separators that aren't spaces.
        return cat not in ("Cc", "Cs")

    text = "".join(ch for ch in text if _is_printable(ch))

    # Normalize runs of 3+ consecutive blank lines to exactly 2.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Final strip after normalisation.
    text = text.strip()

    if len(text) < _MIN_RESUME_CHARS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Resume text is too short ({len(text)} characters after cleaning). "
                f"Minimum required is {_MIN_RESUME_CHARS} characters. "
                "Please upload a complete resume."
            ),
        )

    if len(text) > _MAX_RESUME_CHARS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Resume text is too long ({len(text)} characters). "
                f"Maximum allowed is {_MAX_RESUME_CHARS} characters. "
                "Please upload a condensed version of your resume."
            ),
        )

    return text


__all__ = ["sanitize_job_url", "sanitize_resume_text"]
