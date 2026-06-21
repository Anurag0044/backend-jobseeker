"""
tests/test_file_parser.py
─────────────────────────
Unit tests for core/file_parser.py:
  1. Valid TXT file → extracted string returned.
  2. Valid PDF file → extracted string returned (PyPDF2 mocked).
  3. Unsupported MIME type (.docx) → 415.
  4. Empty PDF (0 bytes readable) → 422.
  5. PDF with under 100 characters → 422.
"""

import io
import pytest
from unittest.mock import patch

from starlette.datastructures import Headers
from fastapi import UploadFile


def _make_upload_file(
    content: bytes,
    filename: str,
    content_type: str,
) -> UploadFile:
    """
    Helper: construct an in-memory UploadFile with the correct content-type
    header. Starlette's UploadFile derives content_type from its headers dict,
    so we pass the Content-Type as a header instead of setting the property.
    """
    headers = Headers({"content-type": content_type})
    file_like = io.BytesIO(content)
    upload = UploadFile(filename=filename, file=file_like, headers=headers)
    return upload


@pytest.mark.asyncio
async def test_valid_txt_file_returns_text(sample_resume_txt):
    """Valid plain-text file returns the extracted string."""
    from core.file_parser import parse_resume_file

    upload = _make_upload_file(sample_resume_txt, "resume.txt", "text/plain")
    result = await parse_resume_file(upload)

    assert isinstance(result, str)
    assert len(result) >= 100
    assert "Jane Doe" in result


@pytest.mark.asyncio
async def test_valid_pdf_file_returns_text(sample_resume_txt):
    """
    Valid PDF file returns extracted text.
    PyPDF2 text extraction is mocked to return the sample resume text so the
    test is not dependent on PyPDF2's ability to read the minimal test PDF.
    """
    from core.file_parser import parse_resume_file

    upload = _make_upload_file(b"%PDF-1.4 fake", "resume.pdf", "application/pdf")
    sample_text = sample_resume_txt.decode("utf-8")

    with patch(
        "core.file_parser._extract_pdf_text",
        return_value=sample_text,
    ):
        result = await parse_resume_file(upload)

    assert isinstance(result, str)
    assert len(result) >= 100


@pytest.mark.asyncio
async def test_unsupported_file_type_returns_415():
    """DOCX (or any non-PDF/TXT) file returns HTTP 415."""
    from core.file_parser import parse_resume_file
    from fastapi import HTTPException

    upload = _make_upload_file(
        b"PK fake docx bytes",
        "resume.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    with pytest.raises(HTTPException) as exc_info:
        await parse_resume_file(upload)

    assert exc_info.value.status_code == 415
    assert "Unsupported file type" in exc_info.value.detail


@pytest.mark.asyncio
async def test_empty_pdf_returns_422():
    """
    PDF that produces no extractable text at all (empty string from PyPDF2)
    returns HTTP 422.
    """
    from core.file_parser import parse_resume_file
    from fastapi import HTTPException

    upload = _make_upload_file(b"%PDF-1.4 empty", "resume.pdf", "application/pdf")

    with patch("core.file_parser._extract_pdf_text", return_value=""):
        with pytest.raises(HTTPException) as exc_info:
            await parse_resume_file(upload)

    assert exc_info.value.status_code == 422
    assert "empty or unreadable" in exc_info.value.detail


@pytest.mark.asyncio
async def test_pdf_under_100_chars_returns_422():
    """
    PDF whose extracted text is valid but under 100 characters returns
    HTTP 422 (too short to be a real resume).
    """
    from core.file_parser import parse_resume_file
    from fastapi import HTTPException

    upload = _make_upload_file(b"%PDF-1.4 short", "resume.pdf", "application/pdf")
    short_text = "Name: J"  # 7 characters — well under threshold

    with patch("core.file_parser._extract_pdf_text", return_value=short_text):
        with pytest.raises(HTTPException) as exc_info:
            await parse_resume_file(upload)

    assert exc_info.value.status_code == 422
    assert "empty or unreadable" in exc_info.value.detail
