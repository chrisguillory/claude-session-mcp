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

    1. Decomposed Request/Response Bases

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

    2. Callable Discriminator for Type Dispatch

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
    ClientDataResponse,
    CountTokensRequest,
    CountTokensResponse,
    EmptyBody,
    EvalRequest,
    EvalResponse,
    HelloResponse,
    MessagesRequest,
    MessagesResponse,
    MetricsEnabledResponse,
    MetricsRequest,
    MetricsResponse,
    SSEEvent,
    StatsigInitializeFullBody,
    StatsigInitializeRequest,
    StatsigRegisterRequest,
    StatsigRegisterResponse,
    TelemetryBatchRequest,
    TelemetryBatchResponse,
)
from src.schemas.cc_internal_api.base import StrictModel

# ==============================================================================
# Connection metadata from mitmproxy
# ==============================================================================


class ConnectionTiming(StrictModel):
    """Timing information for connection establishment."""

    start: float = Field(description='Connection start timestamp')
    tcp_setup: float | None = None
    tls_setup: float | None = None


class ClientConnection(StrictModel):
    """Client connection metadata from mitmproxy."""

    id: str = Field(description='Connection UUID')
    address: Sequence[Any] = Field(description='Client address tuple')
    tls_version: str | None = None
    timing: ConnectionTiming


class ServerConnection(StrictModel):
    """Server connection metadata from mitmproxy."""

    id: str = Field(description='Connection UUID')
    address: Sequence[Any] = Field(description='Server address tuple')
    tls_established: bool | None = None
    tls_version: str | None = None
    alpn: str | None = None
    sni: str | None = None
    timing: ConnectionTiming


# ==============================================================================
# LEVEL 1: Decomposed base classes for requests vs responses
# ==============================================================================


class RequestCapture(StrictModel):
    """
    Base for all HTTP request captures.

    Contains fields that are meaningful for requests. Response-specific fields
    (status_code, reason, duration_seconds) are NOT included - they exist in
    the JSON as null but are ignored.

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

    # --- mitmproxy metadata ---
    url: str = Field(description='Full URL')
    cookies: Mapping[str, str] = Field(description='HTTP cookies')
    is_replay: bool | None = Field(description='mitmproxy replay flag')
    client_conn: ClientConnection = Field(description='Client connection info')

    # --- Claude session correlation (added by intercept script) ---
    session_id: str | None = Field(description='Claude Code session ID')


class ResponseCapture(StrictModel):
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

    # --- mitmproxy metadata ---
    url: str = Field(description='Full URL')
    cookies: Mapping[str, str] = Field(description='HTTP cookies')
    is_replay: bool | None = Field(description='mitmproxy replay flag')
    server_conn: ServerConnection = Field(description='Server connection info')

    # --- Claude session correlation (added by intercept script) ---
    session_id: str | None = Field(description='Claude Code session ID')


# ==============================================================================
# LEVEL 2: Service-specific base classes
# ==============================================================================


class AnthropicRequestCapture(RequestCapture):
    """Base for all Anthropic API request captures."""

    host: Literal['api.anthropic.com']


class AnthropicResponseCapture(ResponseCapture):
    """Base for all Anthropic API response captures."""

    host: Literal['api.anthropic.com']


class StatsigRequestCapture(RequestCapture):
    """Base for all Statsig request captures."""

    host: Literal['statsig.anthropic.com']


class StatsigResponseCapture(ResponseCapture):
    """Base for all Statsig response captures."""

    host: Literal['statsig.anthropic.com']


# ==============================================================================
# LEVEL 3: Endpoint-specific wrappers - Anthropic API
# ==============================================================================


class MessagesRequestCapture(AnthropicRequestCapture):
    """Captured POST /v1/messages request."""

    method: Literal['POST']
    body: MessagesRequest


class MessagesStreamResponseCapture(AnthropicResponseCapture):
    """
    Captured POST /v1/messages streaming response (SSE).

    Used when request has stream: true (the common case for Claude Code).
    The events field contains parsed SSE events from the stream.
    """

    events: Sequence[SSEEvent] = Field(description='Parsed SSE events')


class ApiErrorDetail(StrictModel):
    """Error detail in API error response."""

    type: str  # e.g., "invalid_request_error", "overloaded_error"
    message: str


class ApiError(StrictModel):
    """
    API error response structure.

    Returned on 4xx/5xx responses instead of a Message.
    """

    type: Literal['error']
    error: ApiErrorDetail


class MessagesJsonResponseCapture(AnthropicResponseCapture):
    """
    Captured POST /v1/messages non-streaming response (JSON).

    Used when request has stream: false, OR for error responses (4xx/5xx).
    The body contains either a complete Message or an error object.
    """

    body: MessagesResponse | ApiError


class TelemetryRequestCapture(AnthropicRequestCapture):
    """Captured POST /api/event_logging/batch request."""

    method: Literal['POST']
    body: TelemetryBatchRequest


class TelemetryResponseCapture(AnthropicResponseCapture):
    """Captured POST /api/event_logging/batch response."""

    body: TelemetryBatchResponse


# --- Count Tokens ---


class CountTokensRequestCapture(AnthropicRequestCapture):
    """Captured POST /v1/messages/count_tokens request."""

    method: Literal['POST']
    body: CountTokensRequest


class CountTokensResponseCapture(AnthropicResponseCapture):
    """Captured POST /v1/messages/count_tokens response."""

    body: CountTokensResponse


# --- Feature Flags (Eval) ---


class EvalRequestCapture(AnthropicRequestCapture):
    """Captured POST /api/eval/sdk-{code} request (feature flags)."""

    method: Literal['POST']
    body: EvalRequest


class EvalResponseCapture(AnthropicResponseCapture):
    """Captured POST /api/eval/sdk-{code} response (feature flags)."""

    body: EvalResponse


# --- Metrics ---


class MetricsRequestCapture(AnthropicRequestCapture):
    """Captured POST /api/claude_code/metrics request (OpenTelemetry metrics)."""

    method: Literal['POST']
    body: MetricsRequest


class MetricsResponseCapture(AnthropicResponseCapture):
    """Captured POST /api/claude_code/metrics response."""

    body: MetricsResponse


class MetricsEnabledResponseCapture(AnthropicResponseCapture):
    """Captured GET /api/claude_code/organizations/metrics_enabled response."""

    body: MetricsEnabledResponse


# --- Health/OAuth (GET requests) ---


class HelloRequestCapture(AnthropicRequestCapture):
    """Captured GET /api/hello request (health check)."""

    method: Literal['GET']
    body: EmptyBody


class HelloResponseCapture(AnthropicResponseCapture):
    """Captured GET /api/hello response (health check)."""

    body: HelloResponse


class ClientDataRequestCapture(AnthropicRequestCapture):
    """Captured GET /api/oauth/claude_cli/client_data request."""

    method: Literal['GET']
    body: EmptyBody


class ClientDataResponseCapture(AnthropicResponseCapture):
    """Captured GET /api/oauth/claude_cli/client_data response."""

    body: ClientDataResponse


class MetricsEnabledRequestCapture(AnthropicRequestCapture):
    """Captured GET /api/claude_code/organizations/metrics_enabled request."""

    method: Literal['GET']
    body: EmptyBody


# ==============================================================================
# LEVEL 3: Endpoint-specific wrappers - Statsig
# ==============================================================================


class StatsigRegisterRequestCapture(StatsigRequestCapture):
    """Captured POST /v1/rgstr request (event logging)."""

    method: Literal['POST']
    body: StatsigRegisterRequest


class StatsigRegisterResponseCapture(StatsigResponseCapture):
    """Captured POST /v1/rgstr response (202 Accepted)."""

    body: StatsigRegisterResponse


class StatsigInitializeRequestCapture(StatsigRequestCapture):
    """Captured POST /v1/initialize request (feature flag init)."""

    method: Literal['POST']
    body: StatsigInitializeRequest


class StatsigInitializeResponseCapture(StatsigResponseCapture):
    """
    Captured POST /v1/initialize response.

    Two response types:
    - 200 OK: Full feature flags (StatsigInitializeFullBody)
    - 204 No Content: Empty response when no updates (EmptyBody)
    """

    body: EmptyBody | StatsigInitializeFullBody


# ==============================================================================
# LEVEL 3: Endpoint-specific wrappers - External Services
# ==============================================================================


class DatadogBaseLogEntry(StrictModel):
    """
    Base log entry in Datadog /api/v2/logs request.

    Contains 53 fields always present in all log entries.
    Specialized subclasses add query and/or agent context fields.
    """

    # --- Datadog standard fields ---
    ddsource: str  # "nodejs"
    ddtags: str  # Comma-separated tags
    message: str  # Event name, e.g., "tengu_api_success"
    service: str  # "claude-code"
    hostname: str  # "claude-code"
    env: str  # "external"

    # --- Claude Code context ---
    model: str  # Model ID
    session_id: str  # Session UUID
    user_type: str  # "external" or "internal"
    betas: str  # Comma-separated beta features
    entrypoint: str  # "cli"
    is_interactive: str  # "true" or "false" (string)
    client_type: str  # "cli"

    # --- SWE Bench fields ---
    swe_bench_run_id: str
    swe_bench_instance_id: str
    swe_bench_task_id: str

    # --- Environment ---
    platform: str  # "darwin"
    arch: str  # "arm64"
    node_version: str  # "v24.3.0"
    terminal: str  # "pycharm"
    package_managers: str  # "npm"
    runtimes: str  # "bun,node"

    # --- Boolean flags ---
    is_running_with_bun: bool
    is_ci: bool
    is_claubbit: bool
    is_claude_code_remote: bool
    is_conductor: bool
    is_github_action: bool
    is_claude_code_action: bool
    is_claude_ai_auth: bool

    # --- Version info ---
    version: str  # "2.0.76"
    version_base: str  # "2.0.76"
    build_time: str  # ISO 8601
    deployment_environment: str  # "unknown-darwin"

    # --- Usage metrics ---
    message_count: int
    message_tokens: int
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    uncached_input_tokens: int

    # --- Timing ---
    duration_ms: int
    duration_ms_including_retries: int
    attempt: int
    ttft_ms: int
    build_age_mins: int

    # --- API context ---
    provider: str  # "firstParty"
    request_id: str  # "req_..."
    stop_reason: str  # "end_turn"
    cost_u_s_d: float  # Cost in USD

    # --- Session flags ---
    did_fall_back_to_non_streaming: bool
    is_non_interactive_session: bool
    print: bool
    is_t_t_y: bool

    # --- Query context (always present) ---
    query_source: str  # "prompt_suggestion"
    permission_mode: str  # "default"


class DatadogQueryLogEntry(DatadogBaseLogEntry):
    """Log entry with query chain context (most common)."""

    query_chain_id: str  # UUID
    query_depth: int


class DatadogAgentLogEntry(DatadogBaseLogEntry):
    """Log entry with agent/subagent context."""

    agent_id: str  # Short ID like "a7155f2"
    agent_type: str  # "subagent"


class DatadogAgentQueryLogEntry(DatadogQueryLogEntry):
    """Log entry with both query chain and agent context."""

    agent_id: str
    agent_type: str


# Union of all Datadog log entry types - Pydantic will try each in order
DatadogLogEntry = DatadogAgentQueryLogEntry | DatadogAgentLogEntry | DatadogQueryLogEntry | DatadogBaseLogEntry


class DatadogRequestCapture(RequestCapture):
    """Captured POST /api/v2/logs request (Datadog logging)."""

    host: Literal['http-intake.logs.us5.datadoghq.com']
    method: Literal['POST']
    body: Sequence[DatadogLogEntry]


class DatadogResponseCapture(ResponseCapture):
    """
    Captured POST /api/v2/logs response (202 Accepted).

    Datadog returns empty JSON on success. We don't use EmptyBody here because
    that's for empty HTTP bodies, while this is an empty JSON object.
    """

    host: Literal['http-intake.logs.us5.datadoghq.com']
    # Body is always {} on 202 success - nothing to type beyond empty mapping


class GCSVersionRequestCapture(RequestCapture):
    """Captured GET /claude-code-dist-.../latest request (version check)."""

    host: Literal['storage.googleapis.com']
    method: Literal['GET']
    body: EmptyBody


class RawTextBody(StrictModel):
    """
    Body container for text responses that couldn't be parsed as JSON.

    Created by preprocessing when body type is 'text' and JSON parsing fails.
    The _raw_text field contains the original text content.
    """

    _raw_text: str


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
    # Anthropic API - Count Tokens
    ('api.anthropic.com', '/v1/messages/count_tokens', 'request'): 'count_tokens_request',
    ('api.anthropic.com', '/v1/messages/count_tokens', 'response'): 'count_tokens_response',
    # Anthropic API - Feature Flags (Eval)
    ('api.anthropic.com', '/api/eval/sdk', 'request'): 'eval_request',
    ('api.anthropic.com', '/api/eval/sdk', 'response'): 'eval_response',
    # Anthropic API - Metrics
    ('api.anthropic.com', '/api/claude_code/metrics', 'request'): 'metrics_request',
    ('api.anthropic.com', '/api/claude_code/metrics', 'response'): 'metrics_response',
    ('api.anthropic.com', '/api/claude_code/organizations/metrics_enabled', 'response'): 'metrics_enabled_response',
    # Anthropic API - Health/OAuth (GET)
    ('api.anthropic.com', '/api/hello', 'request'): 'hello_request',
    ('api.anthropic.com', '/api/hello', 'response'): 'hello_response',
    ('api.anthropic.com', '/api/oauth/claude_cli/client_data', 'request'): 'client_data_request',
    ('api.anthropic.com', '/api/oauth/claude_cli/client_data', 'response'): 'client_data_response',
    ('api.anthropic.com', '/api/claude_code/organizations/metrics_enabled', 'request'): 'metrics_enabled_request',
    # Statsig - Register
    ('statsig.anthropic.com', '/v1/rgstr', 'request'): 'statsig_register_request',
    ('statsig.anthropic.com', '/v1/rgstr', 'response'): 'statsig_register_response',
    # Statsig - Initialize
    ('statsig.anthropic.com', '/v1/initialize', 'request'): 'statsig_initialize_request',
    ('statsig.anthropic.com', '/v1/initialize', 'response'): 'statsig_initialize_response',
    # External - Datadog
    ('http-intake.logs.us5.datadoghq.com', '/api/v2/logs', 'request'): 'datadog_request',
    ('http-intake.logs.us5.datadoghq.com', '/api/v2/logs', 'response'): 'datadog_response',
    # External - GCS (version check)
    ('storage.googleapis.com', '/claude-code-dist/latest', 'request'): 'gcs_version_request',
    ('storage.googleapis.com', '/claude-code-dist/latest', 'response'): 'gcs_version_response',
}


def _normalize_path(path: str) -> str:
    """
    Normalize path for registry lookup.

    Handles:
    - Query string removal: /path?query=value → /path
    - SDK variant normalization: /api/eval/sdk-ABC123 → /api/eval/sdk
    - GCS path normalization: /claude-code-dist-UUID/claude-code-releases/latest → /claude-code-dist/latest
    """
    # Remove query string
    path = path.split('?')[0]

    # Normalize dynamic SDK paths
    path = re.sub(r'/api/eval/sdk-[a-zA-Z0-9]+', '/api/eval/sdk', path)

    # Normalize GCS version check paths (strip UUID and intermediate dir)
    path = re.sub(
        r'/claude-code-dist-[a-f0-9-]+/claude-code-releases/latest',
        '/claude-code-dist/latest',
        path,
    )

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
