"""
Capture loading and preprocessing utilities.

This module provides:
- CapturedTraffic: The discriminated union of all capture types
- load_capture(): Load and validate a single capture file
- load_captures_batch(): Load multiple captures with error handling
- _preprocess_capture(): Transform raw JSON for Pydantic validation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import Discriminator, Tag, TypeAdapter, ValidationError

# Import all capture types for the discriminated union
from src.schemas.captures.anthropic import (
    ClientDataRequestCapture,
    ClientDataResponseCapture,
    CountTokensRequestCapture,
    CountTokensResponseCapture,
    EvalRequestCapture,
    EvalResponseCapture,
    HelloRequestCapture,
    HelloResponseCapture,
    MessagesJsonResponseCapture,
    MessagesRequestCapture,
    MessagesStreamResponseCapture,
    MetricsEnabledRequestCapture,
    MetricsEnabledResponseCapture,
    MetricsRequestCapture,
    MetricsResponseCapture,
    TelemetryRequestCapture,
    TelemetryResponseCapture,
)
from src.schemas.captures.datadog import DatadogRequestCapture, DatadogResponseCapture
from src.schemas.captures.gcs import (
    GCSVersionRequestCapture,
    GCSVersionResponseCapture,
    UnknownRequestCapture,
    UnknownResponseCapture,
)
from src.schemas.captures.registry import extract_endpoint_from_filename, get_capture_type
from src.schemas.captures.statsig import (
    StatsigInitializeRequestCapture,
    StatsigInitializeResponseCapture,
    StatsigRegisterRequestCapture,
    StatsigRegisterResponseCapture,
)

# ==============================================================================
# Discriminated union of all capture types
# ==============================================================================

CapturedTraffic = Annotated[
    # Anthropic API - Messages
    Annotated[MessagesRequestCapture, Tag('messages_request')]
    | Annotated[MessagesStreamResponseCapture, Tag('messages_stream_response')]
    | Annotated[MessagesJsonResponseCapture, Tag('messages_json_response')]
    # Anthropic API - Telemetry
    | Annotated[TelemetryRequestCapture, Tag('telemetry_request')]
    | Annotated[TelemetryResponseCapture, Tag('telemetry_response')]
    # Anthropic API - Count Tokens
    | Annotated[CountTokensRequestCapture, Tag('count_tokens_request')]
    | Annotated[CountTokensResponseCapture, Tag('count_tokens_response')]
    # Anthropic API - Feature Flags (Eval)
    | Annotated[EvalRequestCapture, Tag('eval_request')]
    | Annotated[EvalResponseCapture, Tag('eval_response')]
    # Anthropic API - Metrics
    | Annotated[MetricsRequestCapture, Tag('metrics_request')]
    | Annotated[MetricsResponseCapture, Tag('metrics_response')]
    | Annotated[MetricsEnabledResponseCapture, Tag('metrics_enabled_response')]
    # Anthropic API - Health/OAuth (GET)
    | Annotated[HelloRequestCapture, Tag('hello_request')]
    | Annotated[HelloResponseCapture, Tag('hello_response')]
    | Annotated[ClientDataRequestCapture, Tag('client_data_request')]
    | Annotated[ClientDataResponseCapture, Tag('client_data_response')]
    | Annotated[MetricsEnabledRequestCapture, Tag('metrics_enabled_request')]
    # Statsig - Register
    | Annotated[StatsigRegisterRequestCapture, Tag('statsig_register_request')]
    | Annotated[StatsigRegisterResponseCapture, Tag('statsig_register_response')]
    # Statsig - Initialize
    | Annotated[StatsigInitializeRequestCapture, Tag('statsig_initialize_request')]
    | Annotated[StatsigInitializeResponseCapture, Tag('statsig_initialize_response')]
    # External - Datadog
    | Annotated[DatadogRequestCapture, Tag('datadog_request')]
    | Annotated[DatadogResponseCapture, Tag('datadog_response')]
    # External - GCS
    | Annotated[GCSVersionRequestCapture, Tag('gcs_version_request')]
    | Annotated[GCSVersionResponseCapture, Tag('gcs_version_response')]
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

    NOTE: This function MUTATES the input dict in place for efficiency.
    The return value is the same object, not a copy.

    Transforms:
    - JSON bodies: extract from {"type": "json", "data": {...}}
    - SSE responses: extract events and convert to list
    - Text bodies: attempt JSON parsing (for Statsig)
    - Missing host/path: extract from filename for responses

    Args:
        data: Raw capture JSON (will be mutated in place)
        filepath: Optional filepath for extracting host/path from filename

    Returns:
        The same dict, now preprocessed and ready for Pydantic validation
    """
    # Handle missing host/path for response captures
    if data.get('direction') == 'response' and not data.get('host'):
        if filepath:
            host, path = extract_endpoint_from_filename(filepath.name)
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
                    data['body'] = {'raw_text': raw}

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
        Tuple of (validated captures, errors dict with file -> error mapping)
    """
    captures: list[CapturedTraffic] = []
    errors: dict[Path, Exception] = {}

    for filepath in sorted(directory.glob(pattern)):
        try:
            captures.append(load_capture(filepath))
        except (ValidationError, json.JSONDecodeError, OSError) as e:
            errors[filepath] = e

    return captures, errors
