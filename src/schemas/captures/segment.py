"""
Segment Analytics capture classes.

This module contains capture wrappers for api.segment.io endpoints.
Segment events are discriminated by the `type` field.

Official event types: track, identify, page, screen, group, alias
See: https://segment.com/docs/connections/sources/catalog/libraries/server/http-api/
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from src.schemas.captures.base import RequestCapture, ResponseCapture
from src.schemas.cc_internal_api.base import StrictModel
from src.schemas.types import PermissiveModel

# ==============================================================================
# Context and Metadata
# ==============================================================================


class SegmentLibraryContext(StrictModel):
    """Library info in Segment context object."""

    name: str  # e.g., "@segment/analytics-node"
    version: str  # e.g., "1.3.0"


class SegmentContext(StrictModel):
    """Common context object for Segment events."""

    library: SegmentLibraryContext | None = None
    # Context can have many optional SDK-specific fields
    model_config = {'extra': 'allow'}


class SegmentMetadata(StrictModel):
    """SDK metadata added by Segment libraries."""

    nodeVersion: str | None = None
    jsRuntime: str | None = None
    # Allow other runtime-specific fields
    model_config = {'extra': 'allow'}


# ==============================================================================
# Event Base
# ==============================================================================


class SegmentEventBase(StrictModel):
    """Common fields for all Segment events.

    All Segment events share these base fields. The API requires either
    userId or anonymousId (at least one must be present).
    """

    timestamp: str  # ISO 8601 timestamp
    messageId: str  # Unique message ID for deduplication
    # One of userId or anonymousId is required by Segment API
    userId: str | None = None
    anonymousId: str | None = None
    # Optional common fields
    context: SegmentContext | None = None
    integrations: Mapping[str, Any] | None = None  # check_schema_typing.py: loose-typing
    # SDK metadata (uses underscore prefix in JSON)
    segment_metadata: SegmentMetadata | None = None

    model_config = {
        'extra': 'forbid',
        'populate_by_name': True,
        'alias_generator': lambda s: '_metadata' if s == 'segment_metadata' else s,
    }


# ==============================================================================
# Claude Code-Specific Traits
# ==============================================================================


class ClaudeCodeIdentifyTraits(StrictModel):
    """Traits for Claude Code identify events.

    Claude Code sends these specific traits during user identification.
    """

    email: str
    account_uuid: str
    organization_uuid: str


class UnknownSegmentTraits(PermissiveModel):
    """Fallback for unknown Segment trait structures.

    Uses PermissiveModel to accept any fields while remaining a proper type.
    Detection: isinstance(x, UnknownSegmentTraits) or isinstance(x, PermissiveModel)
    """

    pass


class UnknownSegmentProperties(PermissiveModel):
    """Fallback for unknown Segment event properties.

    Uses PermissiveModel to accept any fields while remaining a proper type.
    Detection: isinstance(x, UnknownSegmentProperties) or isinstance(x, PermissiveModel)
    """

    pass


# ==============================================================================
# Event Types (Discriminated by `type` field)
# ==============================================================================


class SegmentIdentifyEvent(SegmentEventBase):
    """Identify event - associates traits with a user.

    Used to associate user traits (email, name, etc.) with a user ID.
    Claude Code uses this to identify the logged-in user.
    """

    type: Literal['identify']
    # Claude Code uses strict traits; unknown traits use PermissiveModel fallback
    traits: ClaudeCodeIdentifyTraits | UnknownSegmentTraits | None = None


class SegmentTrackEvent(SegmentEventBase):
    """Track event - records a user action.

    Used to record custom events like button clicks, form submissions, etc.
    """

    type: Literal['track']
    event: str  # Event name (required for track)
    properties: UnknownSegmentProperties | None = None


class SegmentPageEvent(SegmentEventBase):
    """Page event - records a page view (web).

    Used to track page views in web applications.
    """

    type: Literal['page']
    name: str | None = None  # Page name
    category: str | None = None  # Page category
    properties: UnknownSegmentProperties | None = None


class SegmentScreenEvent(SegmentEventBase):
    """Screen event - records a screen view (mobile).

    Used to track screen views in mobile applications.
    """

    type: Literal['screen']
    name: str | None = None  # Screen name
    category: str | None = None  # Screen category
    properties: UnknownSegmentProperties | None = None


class SegmentGroupEvent(SegmentEventBase):
    """Group event - associates user with a group/account.

    Used to associate a user with a company, organization, or team.
    """

    type: Literal['group']
    groupId: str  # Group ID (required)
    traits: UnknownSegmentTraits | None = None


class SegmentAliasEvent(SegmentEventBase):
    """Alias event - merges user identities.

    Used to merge an anonymous user with an identified user.
    """

    type: Literal['alias']
    previousId: str  # Previous user ID to merge


# Discriminated union of all Segment event types
SegmentEvent = (
    SegmentIdentifyEvent
    | SegmentTrackEvent
    | SegmentPageEvent
    | SegmentScreenEvent
    | SegmentGroupEvent
    | SegmentAliasEvent
)


# ==============================================================================
# Batch Request/Response
# ==============================================================================


class SegmentBatchRequest(StrictModel):
    """Segment batch request body.

    The batch endpoint accepts an array of events, each discriminated by `type`.
    """

    batch: Sequence[SegmentEvent]  # Strictly typed discriminated union
    sentAt: str  # ISO timestamp


class SegmentBatchResponse(StrictModel):
    """Segment batch response body."""

    success: bool


# ==============================================================================
# Capture Classes
# ==============================================================================


class SegmentRequestCapture(RequestCapture):
    """Base for api.segment.io requests."""

    host: Literal['api.segment.io']


class SegmentResponseCapture(ResponseCapture):
    """Base for api.segment.io responses."""

    host: Literal['api.segment.io']


class SegmentBatchRequestCapture(SegmentRequestCapture):
    """Captured POST /v1/batch request (analytics)."""

    method: Literal['POST']
    body: SegmentBatchRequest


class SegmentBatchResponseCapture(SegmentResponseCapture):
    """Captured POST /v1/batch response (analytics)."""

    body: SegmentBatchResponse
