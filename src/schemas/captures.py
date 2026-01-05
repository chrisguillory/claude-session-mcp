"""
HTTP traffic capture modeling with type-safe validation.

This module provides Pydantic v2 models for capturing, validating, and
analyzing HTTP traffic from the Claude Code observability platform.

Architecture:
- RequestCapture / ResponseCapture: Separate bases for requests vs responses
- Service-level inheritance: AnthropicRequestCapture, StatsigResponseCapture, etc.
- Endpoint-specific wrappers: MessagesRequestCapture, etc.
- Discriminated union: CapturedTraffic with callable discriminator
- Registry-based dispatch: get_capture_type()

Design Decisions:

    1. Decomposed Request/Response Bases (2026-01-05)

    HTTP requests and responses are fundamentally different entities. Using a
    shared base class with `status_code: int | None` loses type precision and
    prevents fail-fast validation. Decomposed bases provide:
    - Type precision: ResponseCapture.status_code is `int`, not `int | None`
    - Fail-fast: Missing status_code on response fails validation immediately
    - Clean semantics: No defensive coding for impossible None cases

    The ~12 duplicated fields between bases are acceptable because:
    - Each base is self-contained and readable
    - No hidden inheritance to trace
    - PermissiveModel.extra='ignore' handles JSON fields not in model

    2. Callable Discriminator for Type Dispatch (2026-01-05)

    We use a callable discriminator (`get_capture_type()`) rather than a simple
    field-based discriminator. This is a deliberate architectural choice:

    WHY NOT a simple field discriminator?
    - Type dispatch depends on MULTIPLE fields: host, path, direction
    - For messages responses, we also inspect body.type (SSE vs JSON)
    - Pydantic's simple Discriminator('field') only supports single-field lookup

    WHY NOT inject a `capture_type` field in intercept_traffic.py?
    - Separation of concerns: intercept_traffic.py is a "dumb" memorializer
    - It captures raw HTTP faithfully without semantic interpretation
    - All type/schema logic belongs here in the validation layer
    - Adding new capture types should only require changes to this file
    - Keeps the capture format stable while interpretation can evolve

    The callable discriminator is the RIGHT place for this logic - it maintains
    clean separation between Layer 1 (capture) and Layer 2 (validation).

Usage:
    from src.schemas.captures import load_capture, CapturedTraffic

    capture = load_capture(Path("captures/req_001_api_anthropic_com_v1_messages.json"))

    if isinstance(capture, MessagesRequestCapture):
        print(f"Model: {capture.body.model}")
        print(f"Messages: {len(capture.body.messages)}")
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Discriminator, Field, Tag, TypeAdapter, ValidationError

# Import existing body schemas - NO DUPLICATION
from src.schemas.cc_internal_api import (
    MessagesRequest,
    SSEEvent,
    StatsigInitializeEmptyBody,
    StatsigInitializeFullBody,
    StatsigInitializeRequest,
    StatsigRegisterRequest,
    StatsigRegisterResponse,
    TelemetryBatchRequest,
    TelemetryBatchResponse,
)
from src.schemas.cc_internal_api.base import PermissiveModel

# ==============================================================================
# LEVEL 1: Decomposed base classes for requests vs responses
# ==============================================================================


class RequestCapture(PermissiveModel):
    """
    Base for all HTTP request captures.

    Contains fields that are meaningful for requests. Response-specific fields
    (status_code, reason, duration_seconds) are NOT included - they exist in
    the JSON as null but are ignored by PermissiveModel.extra='ignore'.

    This enables fail-fast validation: if a capture is missing a required
    request field, validation fails immediately.
    """

    # --- Identity (common to requests and responses) ---
    flow_id: str = Field(description='mitmproxy flow ID for correlation')
    sequence: int = Field(description='Sequence number in capture session')
    timestamp: float = Field(description='Unix timestamp')
    timestamp_iso: str = Field(description='ISO 8601 timestamp')

    # --- Direction (discriminator field, no default - fail-fast) ---
    direction: Literal['request'] = Field(description='HTTP direction')

    # --- HTTP request context ---
    host: str = Field(description='Request host')
    path: str = Field(description='Request path')
    method: str = Field(description='HTTP method')
    scheme: str = Field(description='URL scheme')
    port: int = Field(description='Port number')
    query: Mapping[str, str] = Field(description='Query parameters')
    headers: Mapping[str, str] = Field(description='HTTP headers')
    http_version: str = Field(description='HTTP version')

    # NOTE: status_code, reason, duration_seconds exist in JSON as null
    # but are NOT declared here - PermissiveModel ignores them


class ResponseCapture(PermissiveModel):
    """
    Base for all HTTP response captures.

    Contains fields that are meaningful for responses. Response-specific fields
    (status_code, reason, duration_seconds) are REQUIRED - no None allowed.

    This enables fail-fast validation: if a response capture is missing
    status_code, validation fails immediately with a clear error.
    """

    # --- Identity (common to requests and responses) ---
    flow_id: str = Field(description='mitmproxy flow ID for correlation')
    sequence: int = Field(description='Sequence number in capture session')
    timestamp: float = Field(description='Unix timestamp')
    timestamp_iso: str = Field(description='ISO 8601 timestamp')

    # --- Direction (discriminator field, no default - fail-fast) ---
    direction: Literal['response'] = Field(description='HTTP direction')

    # --- Correlation fields (copied from request for correlation) ---
    host: str = Field(description='Request host (for correlation)')
    path: str = Field(description='Request path (for correlation)')
    method: str = Field(description='HTTP method (echoed from request)')

    # --- HTTP response context ---
    status_code: int = Field(description='HTTP status code')
    reason: str = Field(description='HTTP status reason')
    headers: Mapping[str, str] = Field(description='HTTP headers')
    http_version: str = Field(description='HTTP version')
    duration_seconds: float = Field(description='Request duration')

    # --- Echoed from request (present in JSON, included for completeness) ---
    scheme: str = Field(description='URL scheme (echoed from request)')
    port: int = Field(description='Port number (echoed from request)')
    query: Mapping[str, str] = Field(description='Query parameters (echoed)')


# ==============================================================================
# LEVEL 2: Service-specific base classes
# ==============================================================================


class AnthropicRequestCapture(RequestCapture):
    """Base for all Anthropic API request captures."""

    host: Literal['api.anthropic.com'] = 'api.anthropic.com'


class AnthropicResponseCapture(ResponseCapture):
    """Base for all Anthropic API response captures."""

    host: Literal['api.anthropic.com'] = 'api.anthropic.com'


class StatsigRequestCapture(RequestCapture):
    """Base for all Statsig request captures."""

    host: Literal['statsig.anthropic.com'] = 'statsig.anthropic.com'


class StatsigResponseCapture(ResponseCapture):
    """Base for all Statsig response captures."""

    host: Literal['statsig.anthropic.com'] = 'statsig.anthropic.com'


# ==============================================================================
# LEVEL 3: Endpoint-specific wrappers - Anthropic API
# ==============================================================================


class MessagesRequestCapture(AnthropicRequestCapture):
    """Captured POST /v1/messages request."""

    method: Literal['POST'] = 'POST'
    body: MessagesRequest


class MessagesStreamResponseCapture(AnthropicResponseCapture):
    """
    Captured POST /v1/messages streaming response (SSE).

    Used when request has stream: true (the common case for Claude Code).
    The events field contains parsed SSE events from the stream.
    """

    events: Sequence[SSEEvent] = Field(description='Parsed SSE events')


class MessagesJsonResponseCapture(AnthropicResponseCapture):
    """
    Captured POST /v1/messages non-streaming response (JSON).

    Used when request has stream: false, OR for error responses (4xx/5xx).
    The body contains either a complete Message or an error object.
    """

    body: Mapping[str, Any] = Field(description='Complete message or error response')


class TelemetryRequestCapture(AnthropicRequestCapture):
    """Captured POST /api/event_logging/batch request."""

    method: Literal['POST'] = 'POST'
    body: TelemetryBatchRequest


class TelemetryResponseCapture(AnthropicResponseCapture):
    """Captured POST /api/event_logging/batch response."""

    body: TelemetryBatchResponse


# ==============================================================================
# LEVEL 3: Endpoint-specific wrappers - Statsig
# ==============================================================================


class StatsigRegisterRequestCapture(StatsigRequestCapture):
    """Captured POST /v1/rgstr request (event logging)."""

    method: Literal['POST'] = 'POST'
    body: StatsigRegisterRequest


class StatsigRegisterResponseCapture(StatsigResponseCapture):
    """Captured POST /v1/rgstr response (202 Accepted)."""

    body: StatsigRegisterResponse


class StatsigInitializeRequestCapture(StatsigRequestCapture):
    """Captured POST /v1/initialize request (feature flag init)."""

    method: Literal['POST'] = 'POST'
    body: StatsigInitializeRequest


class StatsigInitializeResponseCapture(StatsigResponseCapture):
    """
    Captured POST /v1/initialize response.

    Two response types:
    - 200 OK: Full feature flags (StatsigInitializeFullBody)
    - 204 No Content: Empty response when no updates (StatsigInitializeEmptyBody)
    """

    body: StatsigInitializeEmptyBody | StatsigInitializeFullBody


# ==============================================================================
# Fallback for unknown endpoints
# ==============================================================================


class UnknownRequestCapture(RequestCapture):
    """
    Fallback capture for unmapped request endpoints.

    Allows the system to gracefully handle new APIs without breaking validation.
    """

    # Body can be dict OR list (Datadog sends list of log entries)
    body: Mapping[str, Any] | Sequence[Any] = Field(default_factory=dict)


class UnknownResponseCapture(ResponseCapture):
    """
    Fallback capture for unmapped response endpoints.

    Allows the system to gracefully handle new APIs without breaking validation.
    """

    # Body can be dict OR list
    body: Mapping[str, Any] | Sequence[Any] = Field(default_factory=dict)
    events: Sequence[Mapping[str, Any]] = Field(default_factory=list)


# ==============================================================================
# Registry and discriminator
# ==============================================================================

# Mapping from (host, path_pattern, direction) to discriminator tag
# Path patterns are normalized (no query strings, SDK variants normalized)
# Note: Messages responses require body.type inspection, handled in get_capture_type()
CAPTURE_REGISTRY: dict[tuple[str, str, str], str] = {
    # Anthropic API - Messages (request only; response needs body.type inspection)
    ('api.anthropic.com', '/v1/messages', 'request'): 'messages_request',
    # Anthropic API - Telemetry
    ('api.anthropic.com', '/api/event_logging/batch', 'request'): 'telemetry_request',
    ('api.anthropic.com', '/api/event_logging/batch', 'response'): 'telemetry_response',
    # Statsig - Register
    ('statsig.anthropic.com', '/v1/rgstr', 'request'): 'statsig_register_request',
    ('statsig.anthropic.com', '/v1/rgstr', 'response'): 'statsig_register_response',
    # Statsig - Initialize
    ('statsig.anthropic.com', '/v1/initialize', 'request'): 'statsig_initialize_request',
    ('statsig.anthropic.com', '/v1/initialize', 'response'): 'statsig_initialize_response',
}


def _normalize_path(path: str) -> str:
    """
    Normalize path for registry lookup.

    Handles:
    - Query string removal: /path?query=value → /path
    - SDK variant normalization: /api/eval/sdk-ABC123 → /api/eval/sdk
    """
    # Remove query string
    path = path.split('?')[0]

    # Normalize dynamic SDK paths
    path = re.sub(r'/api/eval/sdk-[a-zA-Z0-9]+', '/api/eval/sdk', path)

    return path


def _extract_endpoint_from_filename(filename: str) -> tuple[str, str]:
    """
    Extract host and path hint from capture filename.

    Filenames are like: req_001_api_anthropic_com_v1_messages_beta_true.json

    Returns (host_hint, path_hint) where:
    - host_hint: 'api.anthropic.com' or 'statsig.anthropic.com' or ''
    - path_hint: '/v1/messages' or '/api/event_logging' or ''
    """
    # Remove prefix and extension
    stem = filename.replace('.json', '')
    parts = stem.split('_', 2)  # ['req', '001', 'api_anthropic_com_...']

    if len(parts) < 3:
        return '', ''

    endpoint_info = parts[2]  # 'api_anthropic_com_v1_messages_beta_true'

    # Detect host
    host = ''
    if endpoint_info.startswith('api_anthropic_com'):
        host = 'api.anthropic.com'
    elif endpoint_info.startswith('statsig_anthropic_com'):
        host = 'statsig.anthropic.com'

    # Try to detect path pattern
    path = ''
    if 'v1_messages' in endpoint_info and 'count_tokens' not in endpoint_info:
        path = '/v1/messages'
    elif 'event_logging' in endpoint_info:
        path = '/api/event_logging/batch'
    elif 'v1_rgstr' in endpoint_info:
        path = '/v1/rgstr'

    return host, path


def get_capture_type(v: Any) -> str:
    """
    Callable discriminator for CapturedTraffic union.

    Reads HTTP context (host, path, direction) from input data
    and returns discriminator tag for union dispatch.

    Special handling:
    - Messages responses: inspects body.type to distinguish SSE vs JSON
    - Other endpoints: uses registry lookup

    Must handle both dict (deserialization) and model instances
    (serialization/re-validation).
    """
    # Extract fields from either dict or model instance
    if isinstance(v, dict):
        host = v.get('host', '')
        raw_path = v.get('path', '')
        direction = v.get('direction', '')
        events = v.get('events')
    else:
        # During serialization, input is model instance
        host = getattr(v, 'host', '')
        raw_path = getattr(v, 'path', '')
        direction = getattr(v, 'direction', '')
        events = getattr(v, 'events', None)

    # Normalize path for lookup
    path = _normalize_path(raw_path)

    # Special handling for Messages API responses - need to distinguish SSE vs JSON
    # After _preprocess_capture():
    #   - SSE responses: body deleted, events populated from body.events
    #   - JSON responses: body.data extracted to body field
    # Discriminate based on presence of events field (set by preprocessing)
    if host == 'api.anthropic.com' and path == '/v1/messages' and direction == 'response':
        if events is not None:
            return 'messages_stream_response'
        return 'messages_json_response'

    # Registry lookup: (host, path, direction)
    key = (host, path, direction)
    if key in CAPTURE_REGISTRY:
        return CAPTURE_REGISTRY[key]

    # Fallback for unknown endpoints - dispatch based on direction
    if direction == 'request':
        return 'unknown_request'
    elif direction == 'response':
        return 'unknown_response'

    # Ultimate fallback
    return 'unknown_request'


# ==============================================================================
# Discriminated union
# ==============================================================================

CapturedTraffic = Annotated[
    # Anthropic API - Messages
    Annotated[MessagesRequestCapture, Tag('messages_request')]
    | Annotated[MessagesStreamResponseCapture, Tag('messages_stream_response')]
    | Annotated[MessagesJsonResponseCapture, Tag('messages_json_response')]
    # Anthropic API - Telemetry
    | Annotated[TelemetryRequestCapture, Tag('telemetry_request')]
    | Annotated[TelemetryResponseCapture, Tag('telemetry_response')]
    # Statsig - Register
    | Annotated[StatsigRegisterRequestCapture, Tag('statsig_register_request')]
    | Annotated[StatsigRegisterResponseCapture, Tag('statsig_register_response')]
    # Statsig - Initialize
    | Annotated[StatsigInitializeRequestCapture, Tag('statsig_initialize_request')]
    | Annotated[StatsigInitializeResponseCapture, Tag('statsig_initialize_response')]
    # Fallback
    | Annotated[UnknownRequestCapture, Tag('unknown_request')]
    | Annotated[UnknownResponseCapture, Tag('unknown_response')],
    Discriminator(get_capture_type),
]


# ==============================================================================
# Preprocessing and loading
# ==============================================================================

# Cached adapter for performance
_CAPTURE_ADAPTER: TypeAdapter[CapturedTraffic] = TypeAdapter(CapturedTraffic)


def _preprocess_capture(data: dict[str, Any], filepath: Path | None = None) -> dict[str, Any]:
    """
    Preprocess capture data before Pydantic validation.

    Transforms:
    - JSON bodies: extract from {"type": "json", "data": {...}}
    - SSE responses: extract events and convert to list
    - Text bodies: attempt JSON parsing (for Statsig)
    - Missing host/path: extract from filename for responses

    Args:
        data: Raw capture JSON
        filepath: Optional filepath for extracting host/path from filename

    Returns:
        Preprocessed data ready for Pydantic validation
    """
    # Handle missing host/path for response captures
    if data.get('direction') == 'response' and not data.get('host'):
        if filepath:
            host, path = _extract_endpoint_from_filename(filepath.name)
            if host:
                data['host'] = host
            if path:
                data['path'] = path

    # Process body wrapper
    body_wrapper = data.get('body', {})

    if isinstance(body_wrapper, dict):
        body_type = body_wrapper.get('type')

        if body_type == 'json':
            # JSON body: extract data field
            data['body'] = body_wrapper.get('data', {})

        elif body_type == 'sse':
            # SSE response: extract parsed_data from each event
            events = body_wrapper.get('events', [])
            data['events'] = [event.get('parsed_data', {}) for event in events if event.get('parsed_data')]
            # Remove body wrapper since we've extracted events
            if 'body' in data:
                del data['body']

        elif body_type == 'text':
            # Text body: try to parse as JSON (Statsig sends JSON as text)
            raw = body_wrapper.get('data', '')
            if raw:
                try:
                    data['body'] = json.loads(raw)
                except json.JSONDecodeError:
                    # Keep as raw text in a dict
                    data['body'] = {'_raw_text': raw}

        elif body_type == 'empty' or not body_wrapper:
            # Empty body
            data['body'] = {}

    return data


def load_capture(filepath: Path) -> CapturedTraffic:
    """
    Load and validate a capture file.

    Returns a CapturedTraffic instance, which will be one of:
    - MessagesRequestCapture for /v1/messages requests
    - MessagesStreamResponseCapture for /v1/messages SSE responses (stream: true)
    - MessagesJsonResponseCapture for /v1/messages JSON responses (stream: false)
    - TelemetryRequestCapture for /api/event_logging/batch
    - StatsigRegisterRequestCapture for Statsig /v1/rgstr
    - UnknownRequestCapture, UnknownResponseCapture for unmapped endpoints

    The discriminator automatically routes to the correct type based on
    HTTP context (host, path, direction, body format).

    Args:
        filepath: Path to capture JSON file

    Returns:
        Validated capture object with full type information

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file isn't valid JSON
        ValidationError: If data doesn't match capture schema
    """
    with open(filepath) as f:
        raw_data = json.load(f)

    clean_data = _preprocess_capture(raw_data, filepath)
    return _CAPTURE_ADAPTER.validate_python(clean_data)


def load_captures_batch(
    directory: Path, pattern: str = '*.json'
) -> tuple[list[CapturedTraffic], dict[Path, Exception]]:
    """
    Load and validate multiple captures.

    Args:
        directory: Directory containing capture files
        pattern: Glob pattern for filenames

    Returns:
        Tuple of (validated captures, errors dict with file → error mapping)
    """
    captures: list[CapturedTraffic] = []
    errors: dict[Path, Exception] = {}

    for filepath in sorted(directory.glob(pattern)):
        try:
            captures.append(load_capture(filepath))
        except (ValidationError, json.JSONDecodeError, OSError) as e:
            errors[filepath] = e

    return captures, errors
