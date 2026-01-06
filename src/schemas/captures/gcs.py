"""
GCS and fallback capture classes.

This module contains:
- GCS version check captures (storage.googleapis.com)
- Unknown/fallback captures for unmapped endpoints
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import Field

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


# ==============================================================================
# Fallback for unknown endpoints
# ==============================================================================


class UnknownRequestCapture(RequestCapture):
    """
    Fallback capture for unmapped request endpoints.

    Allows the system to gracefully handle new APIs without breaking validation.
    """

    # Body can be dict OR list (Datadog sends list of log entries).
    # Fallback for unmodeled endpoints; should shrink as coverage increases.
    body: Mapping[str, Any] | Sequence[Any] = Field(default_factory=dict)  # noqa: loose-typing


class UnknownResponseCapture(ResponseCapture):
    """
    Fallback capture for unmapped response endpoints.

    Allows the system to gracefully handle new APIs without breaking validation.
    """

    # Body can be dict OR list.
    # Fallback for unmodeled endpoints; should shrink as coverage increases.
    body: Mapping[str, Any] | Sequence[Any] = Field(default_factory=dict)  # noqa: loose-typing
    # Fallback for unmodeled SSE events.
    events: Sequence[Mapping[str, Any]] = Field(default_factory=list)  # noqa: loose-typing
