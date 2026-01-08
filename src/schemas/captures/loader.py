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

import pydantic

# Import all capture types for the discriminated union
from src.schemas.captures.anthropic import (
    AccountSettingsRequestCapture,
    AccountSettingsResponseCapture,
    ClientDataRequestCapture,
    ClientDataResponseCapture,
    CountTokensRequestCapture,
    CountTokensResponseCapture,
    CreateApiKeyRequestCapture,
    CreateApiKeyResponseCapture,
    EvalRequestCapture,
    EvalResponseCapture,
    GroveRequestCapture,
    GroveResponseCapture,
    HelloRequestCapture,
    HelloResponseCapture,
    MessagesJsonResponseCapture,
    MessagesRequestCapture,
    MessagesStreamResponseCapture,
    MetricsEnabledRequestCapture,
    MetricsEnabledResponseCapture,
    MetricsRequestCapture,
    MetricsResponseCapture,
    ModelAccessRequestCapture,
    ModelAccessResponseCapture,
    ProfileRequestCapture,
    ProfileResponseCapture,
    ReferralEligibilityRequestCapture,
    ReferralEligibilityResponseCapture,
    ReferralRedemptionsRequestCapture,
    ReferralRedemptionsResponseCapture,
    RolesRequestCapture,
    RolesResponseCapture,
    SettingsRequestCapture,
    SettingsResponseCapture,
    TelemetryRequestCapture,
    TelemetryResponseCapture,
)
from src.schemas.captures.datadog import DatadogRequestCapture, DatadogResponseCapture
from src.schemas.captures.external import (
    CodeClaudeComDocRequestCapture,
    CodeClaudeComDocResponseCapture,
    DomainInfoRequestCapture,
    DomainInfoResponseCapture,
    OAuthTokenRequestCapture,
    OAuthTokenResponseCapture,
    PlatformClaudeComDocRequestCapture,
    PlatformClaudeComDocResponseCapture,
)
from src.schemas.captures.gcs import (
    GCSVersionRequestCapture,
    GCSVersionResponseCapture,
)
from src.schemas.captures.proxy import ProxyErrorCapture
from src.schemas.captures.registry import extract_endpoint_from_filename, get_capture_type
from src.schemas.captures.segment import (
    SegmentBatchRequestCapture,
    SegmentBatchResponseCapture,
)
from src.schemas.captures.statsig import (
    StatsigInitializeRequestCapture,
    StatsigInitializeResponseCapture,
    StatsigRegisterRequestCapture,
    StatsigRegisterResponseCapture,
)
from src.schemas.captures.unknown import (
    UnknownRequestCapture,
    UnknownResponseCapture,
)

# ==============================================================================
# Discriminated union of all capture types
# ==============================================================================

CapturedTraffic = Annotated[
    # Anthropic API - Messages
    Annotated[MessagesRequestCapture, pydantic.Tag('messages_request')]
    | Annotated[MessagesStreamResponseCapture, pydantic.Tag('messages_stream_response')]
    | Annotated[MessagesJsonResponseCapture, pydantic.Tag('messages_json_response')]
    # Anthropic API - Telemetry
    | Annotated[TelemetryRequestCapture, pydantic.Tag('telemetry_request')]
    | Annotated[TelemetryResponseCapture, pydantic.Tag('telemetry_response')]
    # Anthropic API - Count Tokens
    | Annotated[CountTokensRequestCapture, pydantic.Tag('count_tokens_request')]
    | Annotated[CountTokensResponseCapture, pydantic.Tag('count_tokens_response')]
    # Anthropic API - Feature Flags (Eval)
    | Annotated[EvalRequestCapture, pydantic.Tag('eval_request')]
    | Annotated[EvalResponseCapture, pydantic.Tag('eval_response')]
    # Anthropic API - Metrics
    | Annotated[MetricsRequestCapture, pydantic.Tag('metrics_request')]
    | Annotated[MetricsResponseCapture, pydantic.Tag('metrics_response')]
    | Annotated[MetricsEnabledRequestCapture, pydantic.Tag('metrics_enabled_request')]
    | Annotated[MetricsEnabledResponseCapture, pydantic.Tag('metrics_enabled_response')]
    # Anthropic API - Health
    | Annotated[HelloRequestCapture, pydantic.Tag('hello_request')]
    | Annotated[HelloResponseCapture, pydantic.Tag('hello_response')]
    # Anthropic API - Grove
    | Annotated[GroveRequestCapture, pydantic.Tag('grove_request')]
    | Annotated[GroveResponseCapture, pydantic.Tag('grove_response')]
    # Anthropic API - Settings
    | Annotated[SettingsRequestCapture, pydantic.Tag('settings_request')]
    | Annotated[SettingsResponseCapture, pydantic.Tag('settings_response')]
    # Anthropic API - OAuth
    | Annotated[ClientDataRequestCapture, pydantic.Tag('client_data_request')]
    | Annotated[ClientDataResponseCapture, pydantic.Tag('client_data_response')]
    | Annotated[ProfileRequestCapture, pydantic.Tag('profile_request')]
    | Annotated[ProfileResponseCapture, pydantic.Tag('profile_response')]
    | Annotated[RolesRequestCapture, pydantic.Tag('roles_request')]
    | Annotated[RolesResponseCapture, pydantic.Tag('roles_response')]
    | Annotated[AccountSettingsRequestCapture, pydantic.Tag('account_settings_request')]
    | Annotated[AccountSettingsResponseCapture, pydantic.Tag('account_settings_response')]
    | Annotated[CreateApiKeyRequestCapture, pydantic.Tag('create_api_key_request')]
    | Annotated[CreateApiKeyResponseCapture, pydantic.Tag('create_api_key_response')]
    # Anthropic API - Referral
    | Annotated[ReferralEligibilityRequestCapture, pydantic.Tag('referral_eligibility_request')]
    | Annotated[ReferralEligibilityResponseCapture, pydantic.Tag('referral_eligibility_response')]
    | Annotated[ReferralRedemptionsRequestCapture, pydantic.Tag('referral_redemptions_request')]
    | Annotated[ReferralRedemptionsResponseCapture, pydantic.Tag('referral_redemptions_response')]
    # Anthropic API - Model Access
    | Annotated[ModelAccessRequestCapture, pydantic.Tag('model_access_request')]
    | Annotated[ModelAccessResponseCapture, pydantic.Tag('model_access_response')]
    # Statsig - Register
    | Annotated[StatsigRegisterRequestCapture, pydantic.Tag('statsig_register_request')]
    | Annotated[StatsigRegisterResponseCapture, pydantic.Tag('statsig_register_response')]
    # Statsig - Initialize
    | Annotated[StatsigInitializeRequestCapture, pydantic.Tag('statsig_initialize_request')]
    | Annotated[StatsigInitializeResponseCapture, pydantic.Tag('statsig_initialize_response')]
    # External - Datadog
    | Annotated[DatadogRequestCapture, pydantic.Tag('datadog_request')]
    | Annotated[DatadogResponseCapture, pydantic.Tag('datadog_response')]
    # External - GCS
    | Annotated[GCSVersionRequestCapture, pydantic.Tag('gcs_version_request')]
    | Annotated[GCSVersionResponseCapture, pydantic.Tag('gcs_version_response')]
    # External - Console OAuth
    | Annotated[OAuthTokenRequestCapture, pydantic.Tag('oauth_token_request')]
    | Annotated[OAuthTokenResponseCapture, pydantic.Tag('oauth_token_response')]
    # External - Segment
    | Annotated[SegmentBatchRequestCapture, pydantic.Tag('segment_batch_request')]
    | Annotated[SegmentBatchResponseCapture, pydantic.Tag('segment_batch_response')]
    # External - Claude.ai
    | Annotated[DomainInfoRequestCapture, pydantic.Tag('domain_info_request')]
    | Annotated[DomainInfoResponseCapture, pydantic.Tag('domain_info_response')]
    # External - Doc Fetches
    | Annotated[CodeClaudeComDocRequestCapture, pydantic.Tag('code_claude_doc_request')]
    | Annotated[CodeClaudeComDocResponseCapture, pydantic.Tag('code_claude_doc_response')]
    | Annotated[PlatformClaudeComDocRequestCapture, pydantic.Tag('platform_claude_doc_request')]
    | Annotated[PlatformClaudeComDocResponseCapture, pydantic.Tag('platform_claude_doc_response')]
    # Fallback
    | Annotated[UnknownRequestCapture, pydantic.Tag('unknown_request')]
    | Annotated[UnknownResponseCapture, pydantic.Tag('unknown_response')]
    # Proxy errors
    | Annotated[ProxyErrorCapture, pydantic.Tag('proxy_error')],
    pydantic.Discriminator(get_capture_type),
]


# ==============================================================================
# Preprocessing and loading
# ==============================================================================

# Cached adapter for performance
_CAPTURE_ADAPTER: pydantic.TypeAdapter[CapturedTraffic] = pydantic.TypeAdapter(CapturedTraffic)


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

    # Skip body processing for error captures (they don't have body field)
    if data.get('direction') == 'error':
        return data

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
        except (pydantic.ValidationError, json.JSONDecodeError, OSError) as e:
            errors[filepath] = e

    return captures, errors
