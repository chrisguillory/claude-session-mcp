"""
Fallback capture classes for unmapped endpoints.

This module contains application-layer fallback captures for HTTP traffic
we've observed but haven't created specific typed models for yet. These
represent gaps in our domain knowledge, not infrastructure concerns.

As endpoints are typed, their captures should migrate from these fallback
types to service-specific modules (anthropic.py, segment.py, etc.).

These are "True Unknown" captures - entire endpoints are unmapped, not just
individual fields. The body uses `Mapping[str, Any] | Sequence[Any]` because
the structure is completely unknown (could be dict OR list).

See README.md for the distinction between:
- Typed + Fallback pattern (PermissiveModel subclasses) - no directive needed
- True Unknown captures (this file) - directive required, goal is to shrink coverage
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pydantic

from src.schemas.captures.base import RequestCapture, ResponseCapture

# ==============================================================================
# Fallback for unknown endpoints
# ==============================================================================


class UnknownRequestCapture(RequestCapture):
    """
    Fallback capture for unmapped request endpoints.

    Used when we observe traffic but haven't created typed models
    for that endpoint yet. The body accepts any JSON-serializable
    structure since we don't know the schema.

    Migration path:
        When you type an endpoint, create a service-specific module
        and typed request/response models. Update capture logic to
        use the typed models instead of this fallback.
    """

    # Body can be dict OR list (Datadog sends list of log entries).
    # Fallback for unmodeled endpoints; should shrink as coverage increases.
    body: Mapping[str, Any] | Sequence[Any] = pydantic.Field(
        default_factory=dict
    )  # check_schema_typing.py: loose-typing


class UnknownResponseCapture(ResponseCapture):
    """
    Fallback capture for unmapped response endpoints.

    Paired with UnknownRequestCapture for unmapped endpoints.
    """

    # Body can be dict OR list.
    # Fallback for unmodeled endpoints; should shrink as coverage increases.
    body: Mapping[str, Any] | Sequence[Any] = pydantic.Field(
        default_factory=dict
    )  # check_schema_typing.py: loose-typing
    # Fallback for unmodeled SSE events.
    events: Sequence[Mapping[str, Any]] = pydantic.Field(default_factory=list)  # check_schema_typing.py: loose-typing
