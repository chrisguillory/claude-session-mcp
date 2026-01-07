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

import pydantic

from src.schemas.captures.base import AnthropicRequestCapture, AnthropicResponseCapture
from src.schemas.cc_internal_api import (
    AccountSettingsResponse,
    ClientDataResponse,
    CliRolesResponse,
    CountTokensRequest,
    CountTokensResponse,
    CreateApiKeyResponse,
    EmptyBody,
    EvalRequest,
    EvalResponse,
    GroveResponse,
    HelloResponse,
    MessagesRequest,
    MessagesResponse,
    MetricsEnabledResponse,
    MetricsRequest,
    MetricsResponse,
    ModelAccessResponse,
    ProfileResponse,
    ReferralEligibilityResponse,
    ReferralRedemptionsResponse,
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

    events: Sequence[SSEEvent] = pydantic.Field(description='Parsed SSE events')


class ApiErrorDetailInfo(StrictModel):
    """Additional error details (e.g., visibility info)."""

    error_visibility: str  # e.g., "user_facing"


class ApiErrorDetail(StrictModel):
    """Error detail in API error response."""

    type: str  # e.g., "invalid_request_error", "overloaded_error", "not_found_error"
    message: str
    details: ApiErrorDetailInfo | None = None  # Optional detailed error info


class ApiError(StrictModel):
    """
    API error response structure.

    Returned on 4xx/5xx responses instead of a Message.
    """

    type: Literal['error']
    error: ApiErrorDetail
    request_id: str | None = None  # Present on some error responses


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


# ==============================================================================
# Grove (/api/claude_code_grove)
# ==============================================================================


class GroveRequestCapture(AnthropicRequestCapture):
    """Captured GET /api/claude_code_grove request (feature gating)."""

    method: Literal['GET']
    body: EmptyBody


class GroveResponseCapture(AnthropicResponseCapture):
    """Captured GET /api/claude_code_grove response (feature gating)."""

    body: GroveResponse


# ==============================================================================
# Settings (/api/claude_code/settings)
# ==============================================================================


class SettingsRequestCapture(AnthropicRequestCapture):
    """Captured GET /api/claude_code/settings request."""

    method: Literal['GET']
    body: EmptyBody


class SettingsResponseCapture(AnthropicResponseCapture):
    """Captured GET /api/claude_code/settings response (may be error)."""

    body: ApiError  # Observed as error response in captures


# ==============================================================================
# Profile (/api/oauth/profile)
# ==============================================================================


class ProfileRequestCapture(AnthropicRequestCapture):
    """Captured GET /api/oauth/profile request."""

    method: Literal['GET']
    body: EmptyBody


class ProfileResponseCapture(AnthropicResponseCapture):
    """Captured GET /api/oauth/profile response."""

    body: ProfileResponse


# ==============================================================================
# Roles (/api/oauth/claude_cli/roles)
# ==============================================================================


class RolesRequestCapture(AnthropicRequestCapture):
    """Captured GET /api/oauth/claude_cli/roles request."""

    method: Literal['GET']
    body: EmptyBody


class RolesResponseCapture(AnthropicResponseCapture):
    """Captured GET /api/oauth/claude_cli/roles response."""

    body: CliRolesResponse


# ==============================================================================
# Account Settings (/api/oauth/account/settings)
# ==============================================================================


class AccountSettingsRequestCapture(AnthropicRequestCapture):
    """Captured GET /api/oauth/account/settings request."""

    method: Literal['GET']
    body: EmptyBody


class AccountSettingsResponseCapture(AnthropicResponseCapture):
    """Captured GET /api/oauth/account/settings response."""

    body: AccountSettingsResponse


# ==============================================================================
# Create API Key (/api/oauth/claude_cli/create_api_key)
# ==============================================================================


class CreateApiKeyRequestCapture(AnthropicRequestCapture):
    """Captured POST /api/oauth/claude_cli/create_api_key request."""

    method: Literal['POST']
    body: EmptyBody  # POST with empty body, auth in headers


class CreateApiKeyResponseCapture(AnthropicResponseCapture):
    """Captured POST /api/oauth/claude_cli/create_api_key response."""

    body: CreateApiKeyResponse


# ==============================================================================
# Referral Eligibility (/api/oauth/organizations/{uuid}/referral/eligibility)
# ==============================================================================


class ReferralEligibilityRequestCapture(AnthropicRequestCapture):
    """Captured GET /api/oauth/organizations/{uuid}/referral/eligibility request."""

    method: Literal['GET']
    body: EmptyBody


class ReferralEligibilityResponseCapture(AnthropicResponseCapture):
    """Captured GET /api/oauth/organizations/{uuid}/referral/eligibility response."""

    body: ReferralEligibilityResponse


# ==============================================================================
# Referral Redemptions (/api/oauth/organizations/{uuid}/referral/redemptions)
# ==============================================================================


class ReferralRedemptionsRequestCapture(AnthropicRequestCapture):
    """Captured GET /api/oauth/organizations/{uuid}/referral/redemptions request."""

    method: Literal['GET']
    body: EmptyBody


class ReferralRedemptionsResponseCapture(AnthropicResponseCapture):
    """Captured GET /api/oauth/organizations/{uuid}/referral/redemptions response."""

    body: ReferralRedemptionsResponse


# ==============================================================================
# Model Access (/api/organization/{uuid}/claude_code_sonnet_1m_access)
# ==============================================================================


class ModelAccessRequestCapture(AnthropicRequestCapture):
    """Captured GET /api/organization/{uuid}/claude_code_sonnet_1m_access request."""

    method: Literal['GET']
    body: EmptyBody


class ModelAccessResponseCapture(AnthropicResponseCapture):
    """Captured GET /api/organization/{uuid}/claude_code_sonnet_1m_access response."""

    body: ModelAccessResponse
