"""
core/gemini_client.py
─────────────────────
Singleton initializer for the Google GenAI client (new unified SDK).

configure() is called exactly once per process via the module-level guard
_CONFIGURED. Every agent imports get_model() to obtain a ready-to-use
GenerativeModel instance without risk of double-initialization.

Note: Uses google.genai (the new unified SDK bundled with google-adk),
not the deprecated google.generativeai package.
"""

import google.genai as genai

from core.config import settings

# ── Module-level singleton guard ─────────────────────────────────────────────
_CLIENT: genai.Client | None = None


def _get_client() -> genai.Client:
    """Return a configured genai.Client instance (singleton)."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _CLIENT


def get_model() -> str:
    """
    Return the canonical model identifier used across all agents.

    The ADK LlmAgent accepts a model string directly; this function
    centralises the model selection so it can be changed in one place.

    Returns:
        Gemini model identifier string.
    """
    # Ensure the client is initialised (side-effect: validates the API key).
    _get_client()
    return "gemini-2.5-flash"
