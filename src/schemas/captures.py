"""
HTTP traffic capture modeling with type-safe validation.

This module provides Pydantic v2 models for capturing, validating, and
analyzing HTTP traffic from the Claude Code observability platform.

Architecture:
- CaptureBase: Common HTTP context for all captures
- Service-level inheritance: AnthropicCapture, StatsigCapture
- Endpoint-specific wrappers: MessagesRequestCapture, etc.
- Discriminated union: CapturedTraffic with callable discriminator
- Registry-based dispatch: get_capture_type()

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
    StatsigRegisterRequest,
    TelemetryBatchRequest,
)
from src.schemas.cc_internal_api.base import PermissiveModel

# ==============================================================================
# LEVEL 1: Base class with HTTP context
# ==============================================================================


class CaptureBase(PermissiveModel):
    """
    Common HTTP context for all captures.

    This base class captures the HTTP metadata that is orthogonal to
    the API contract. All captures have flow_id, direction, timestamp, etc.

    Note: Response captures may not have host/path directly - these are
    extracted from the filename or correlated with requests.
    """

    # Identification
    flow_id: str = Field(description='mitmproxy flow ID for correlation')
    sequence: int = Field(description='Sequence number in capture session')
    direction: Literal['request', 'response'] = Field(description='HTTP direction')

    # URL components (may be extracted from filename for responses)
    host: str = Field(default='', description='Request host')
    path: str = Field(default='', description='Request path')
    method: str = Field(default='', description='HTTP method')

    # Timing
    timestamp: float = Field(description='Unix timestamp')
    timestamp_iso: str = Field(description='ISO 8601 timestamp')

    # HTTP metadata
    http_version: str = Field(default='HTTP/1.1', description='HTTP version')
    headers: Mapping[str, str] = Field(default_factory=dict, description='HTTP headers')

    # Request-specific (optional for responses)
    scheme: str = Field(default='https', description='URL scheme')
    port: int = Field(default=443, description='Port number')
    query: Mapping[str, str] = Field(default_factory=dict, description='Query parameters')

    # Response-specific (optional for requests)
    status_code: int | None = Field(default=None, description='HTTP status code')
    reason: str | None = Field(default=None, description='HTTP status reason')
    duration_seconds: float | None = Field(default=None, description='Request duration')


# ==============================================================================
# LEVEL 2: Service-specific base classes
# ==============================================================================


class AnthropicCapture(CaptureBase):
    """Base for all Anthropic API captures."""

    host: Literal['api.anthropic.com'] = 'api.anthropic.com'


class StatsigCapture(CaptureBase):
    """Base for all Statsig captures."""

    host: Literal['statsig.anthropic.com'] = 'statsig.anthropic.com'


# ==============================================================================
# LEVEL 3: Endpoint-specific wrappers - Anthropic API
# ==============================================================================


class MessagesRequestCapture(AnthropicCapture):
    """Captured POST /v1/messages request."""

    direction: Literal['request'] = 'request'
    method: Literal['POST'] = 'POST'
    body: MessagesRequest


class MessagesResponseCapture(AnthropicCapture):
    """Captured POST /v1/messages response (SSE stream)."""

    direction: Literal['response'] = 'response'
    status_code: int = 200
    events: Sequence[SSEEvent] = Field(default_factory=list)


class TelemetryRequestCapture(AnthropicCapture):
    """Captured POST /api/event_logging/batch request."""

    direction: Literal['request'] = 'request'
    method: Literal['POST'] = 'POST'
    body: TelemetryBatchRequest


# ==============================================================================
# LEVEL 3: Endpoint-specific wrappers - Statsig
# ==============================================================================


class StatsigRegisterCapture(StatsigCapture):
    """Captured POST /v1/rgstr request (event logging)."""

    direction: Literal['request'] = 'request'
    method: Literal['POST'] = 'POST'
    body: StatsigRegisterRequest


# ==============================================================================
# Fallback for unknown endpoints
# ==============================================================================


class UnknownCapture(CaptureBase):
    """
    Fallback capture for unmapped endpoints.

    This allows the system to gracefully handle new APIs without
    breaking validation. When a new endpoint is added, captures will
    validate against this model until a specific wrapper is created.
    """

    # Body can be dict OR list (Datadog sends list of log entries)
    body: Mapping[str, Any] | Sequence[Any] = Field(default_factory=dict)
    events: Sequence[Mapping[str, Any]] = Field(default_factory=list)


# ==============================================================================
# Registry and discriminator
# ==============================================================================

# Mapping from (host, path_pattern, direction) to discriminator tag
# Path patterns are normalized (no query strings, SDK variants normalized)
CAPTURE_REGISTRY: dict[tuple[str, str, str], str] = {
    # Anthropic API - Messages
    ('api.anthropic.com', '/v1/messages', 'request'): 'messages_request',
    ('api.anthropic.com', '/v1/messages', 'response'): 'messages_response',
    # Anthropic API - Telemetry
    ('api.anthropic.com', '/api/event_logging/batch', 'request'): 'telemetry_request',
    # Statsig - Register
    ('statsig.anthropic.com', '/v1/rgstr', 'request'): 'statsig_register',
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

    Must handle both dict (deserialization) and model instances
    (serialization/re-validation).
    """
    # Extract fields from either dict or model instance
    if isinstance(v, dict):
        host = v.get('host', '')
        raw_path = v.get('path', '')
        direction = v.get('direction', '')
    else:
        # During serialization, input is model instance
        host = getattr(v, 'host', '')
        raw_path = getattr(v, 'path', '')
        direction = getattr(v, 'direction', '')

    # Normalize path for lookup
    path = _normalize_path(raw_path)

    # Registry lookup: (host, path, direction)
    key = (host, path, direction)
    if key in CAPTURE_REGISTRY:
        return CAPTURE_REGISTRY[key]

    # Partial match - try with just host and path (for responses that
    # might have direction but different path due to normalization)
    for (reg_host, reg_path, reg_dir), tag in CAPTURE_REGISTRY.items():
        if host == reg_host and path == reg_path:
            # Found matching host/path, use correct direction
            if direction == reg_dir:
                return tag

    # Fallback for unknown endpoints
    return 'unknown'


# ==============================================================================
# Discriminated union
# ==============================================================================

CapturedTraffic = Annotated[
    Annotated[MessagesRequestCapture, Tag('messages_request')]
    | Annotated[MessagesResponseCapture, Tag('messages_response')]
    | Annotated[TelemetryRequestCapture, Tag('telemetry_request')]
    | Annotated[StatsigRegisterCapture, Tag('statsig_register')]
    | Annotated[UnknownCapture, Tag('unknown')],
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
    - MessagesRequestCapture, MessagesResponseCapture for /v1/messages
    - TelemetryRequestCapture for /api/event_logging/batch
    - StatsigRegisterCapture for Statsig /v1/rgstr
    - UnknownCapture for unmapped endpoints

    The discriminator automatically routes to the correct type based on
    HTTP context (host, path, direction).

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
