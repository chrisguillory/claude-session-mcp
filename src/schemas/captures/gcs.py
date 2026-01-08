"""
GCS (Google Cloud Storage) capture classes.

This module contains captures for storage.googleapis.com traffic,
specifically the version check endpoint used by Claude Code.
"""

from __future__ import annotations

from typing import Literal

from src.schemas.captures.base import RequestCapture, ResponseCapture
from src.schemas.cc_internal_api import EmptyBody
from src.schemas.cc_internal_api.base import StrictModel

# ==============================================================================
# GCS Version Check
# ==============================================================================


class RawTextBody(StrictModel):
    """
    Body container for text responses that couldn't be parsed as JSON.

    Created by preprocessing when body type is 'text' and JSON parsing fails.
    The raw_text field contains the original text content.
    """

    raw_text: str


class GCSVersionRequestCapture(RequestCapture):
    """Captured GET /claude-code-dist-.../latest request (version check)."""

    host: Literal['storage.googleapis.com']
    method: Literal['GET']
    body: EmptyBody


class GCSVersionResponseCapture(ResponseCapture):
    """Captured GET /claude-code-dist-.../latest response (version string)."""

    host: Literal['storage.googleapis.com']
    body: RawTextBody
