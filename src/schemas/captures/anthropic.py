"""
Anthropic API endpoint capture classes.

This module contains capture wrappers for all Anthropic API endpoints:
- /v1/messages - Main conversation API (streaming and non-streaming)
- /api/event_logging/batch - Telemetry
- /v1/messages/count_tokens - Token counting
- /api/eval/sdk-* - Feature flags
- /api/claude_code/metrics - Metrics reporting
- /api/hello, /api/oauth/* - Health and OAuth endpoints
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from src.schemas.captures.base import AnthropicRequestCapture, AnthropicResponseCapture
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
    TelemetryBatchRequest,
    TelemetryBatchResponse,
)
from src.schemas.cc_internal_api.base import StrictModel

# ==============================================================================
# Messages API (/v1/messages)
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


# ==============================================================================
# Telemetry (/api/event_logging/batch)
# ==============================================================================


class TelemetryRequestCapture(AnthropicRequestCapture):
    """Captured POST /api/event_logging/batch request."""

    method: Literal['POST']
    body: TelemetryBatchRequest


class TelemetryResponseCapture(AnthropicResponseCapture):
    """Captured POST /api/event_logging/batch response."""

    body: TelemetryBatchResponse


# ==============================================================================
# Count Tokens (/v1/messages/count_tokens)
# ==============================================================================


class CountTokensRequestCapture(AnthropicRequestCapture):
    """Captured POST /v1/messages/count_tokens request."""

    method: Literal['POST']
    body: CountTokensRequest


class CountTokensResponseCapture(AnthropicResponseCapture):
    """Captured POST /v1/messages/count_tokens response."""

    body: CountTokensResponse


# ==============================================================================
# Feature Flags (/api/eval/sdk-*)
# ==============================================================================


class EvalRequestCapture(AnthropicRequestCapture):
    """Captured POST /api/eval/sdk-{code} request (feature flags)."""

    method: Literal['POST']
    body: EvalRequest


class EvalResponseCapture(AnthropicResponseCapture):
    """Captured POST /api/eval/sdk-{code} response (feature flags)."""

    body: EvalResponse


# ==============================================================================
# Metrics (/api/claude_code/metrics, /api/claude_code/organizations/metrics_enabled)
# ==============================================================================


class MetricsRequestCapture(AnthropicRequestCapture):
    """Captured POST /api/claude_code/metrics request (OpenTelemetry metrics)."""

    method: Literal['POST']
    body: MetricsRequest


class MetricsResponseCapture(AnthropicResponseCapture):
    """Captured POST /api/claude_code/metrics response."""

    body: MetricsResponse


class MetricsEnabledRequestCapture(AnthropicRequestCapture):
    """Captured GET /api/claude_code/organizations/metrics_enabled request."""

    method: Literal['GET']
    body: EmptyBody


class MetricsEnabledResponseCapture(AnthropicResponseCapture):
    """Captured GET /api/claude_code/organizations/metrics_enabled response."""

    body: MetricsEnabledResponse


# ==============================================================================
# Health/OAuth (GET requests)
# ==============================================================================


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
