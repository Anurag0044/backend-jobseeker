"""
core/logger.py
──────────────
Structured logging using loguru.

Configuration:
  - Format: timestamp | level | module | message
  - Level:  read from LOG_LEVEL env variable (default: INFO)
  - Output: stdout — Render.com captures stdout natively
  - Single shared `logger` instance imported across all modules

Usage:
    from core.logger import logger
    logger.info("Something happened")
    logger.error("Something broke: {detail}", detail=str(exc))
"""

import os
import sys

from loguru import logger

# ── Remove loguru's default stderr handler ────────────────────────────────────
logger.remove()

# ── Read log level from environment (default INFO) ────────────────────────────
_LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()

# ── Add stdout handler with structured format ─────────────────────────────────
logger.add(
    sys.stdout,
    level=_LOG_LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    colorize=False,   # Render.com log viewer handles colour codes poorly
    backtrace=True,   # Show full traceback context on errors
    diagnose=False,   # Disable variable values in tracebacks (security)
    enqueue=False,    # Synchronous; we're already in an async event loop
)

__all__ = ["logger"]
