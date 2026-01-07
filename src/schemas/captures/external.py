"""
External service capture classes.

This module contains capture wrappers for non-Anthropic API endpoints:
- console.anthropic.com - OAuth token exchange
- api.segment.io - Analytics batching
- claude.ai - Domain info checks
- code.claude.com, platform.claude.com - Documentation fetches
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from src.schemas.captures.base import RequestCapture, ResponseCapture
from src.schemas.captures.gcs import RawTextBody
from src.schemas.cc_internal_api import EmptyBody
from src.schemas.cc_internal_api.base import StrictModel

# ==============================================================================
# OAuth Token (console.anthropic.com)
# ==============================================================================


class OAuthTokenRequest(StrictModel):
    """OAuth token exchange request body."""

    grant_type: Literal['authorization_code']
    code: str  # Authorization code
    redirect_uri: str  # Callback URL
    client_id: str  # OAuth client ID
    code_verifier: str  # PKCE code verifier
    state: str  # CSRF state token


class OAuthTokenOrganization(StrictModel):
    """Organization info in OAuth token response."""

    uuid: str
    name: str


class OAuthTokenAccount(StrictModel):
    """Account info in OAuth token response."""

    uuid: str
    email_address: str


class OAuthTokenResponse(StrictModel):
    """OAuth token exchange response body."""

    token_type: Literal['Bearer']
    access_token: str  # OAuth access token
    expires_in: int  # Token lifetime in seconds
    refresh_token: str  # Refresh token for renewal
    scope: str  # Space-separated scopes
    organization: OAuthTokenOrganization
    account: OAuthTokenAccount


class ConsoleRequestCapture(RequestCapture):
    """Base for console.anthropic.com requests."""

    host: Literal['console.anthropic.com']


class ConsoleResponseCapture(ResponseCapture):
    """Base for console.anthropic.com responses."""

    host: Literal['console.anthropic.com']


class OAuthTokenRequestCapture(ConsoleRequestCapture):
    """Captured POST /v1/oauth/token request."""

    method: Literal['POST']
    body: OAuthTokenRequest


class OAuthTokenResponseCapture(ConsoleResponseCapture):
    """Captured POST /v1/oauth/token response."""

    body: OAuthTokenResponse


# ==============================================================================
# Segment Analytics (api.segment.io)
# ==============================================================================


class SegmentEvent(StrictModel):
    """Individual event in Segment batch."""

    # Segment events have varying structures - use relaxed typing
    type: str  # e.g., "track", "identify"
    # Allow any additional fields
    model_config = {'extra': 'allow'}


class SegmentBatchRequest(StrictModel):
    """Segment batch request body."""

    batch: Sequence[Mapping[str, Any]]  # noqa: loose-typing # Segment analytics events have varied structures
    sentAt: str  # ISO timestamp


class SegmentBatchResponse(StrictModel):
    """Segment batch response body."""

    success: bool


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


# ==============================================================================
# Domain Info (claude.ai)
# ==============================================================================


class DomainInfoResponse(StrictModel):
    """Domain info response body."""

    domain: str  # e.g., "code.claude.com"
    can_fetch: bool  # Whether domain is fetchable


class ClaudeAiRequestCapture(RequestCapture):
    """Base for claude.ai requests."""

    host: Literal['claude.ai']


class ClaudeAiResponseCapture(ResponseCapture):
    """Base for claude.ai responses."""

    host: Literal['claude.ai']


class DomainInfoRequestCapture(ClaudeAiRequestCapture):
    """Captured GET /api/web/domain_info request."""

    method: Literal['GET']
    body: EmptyBody


class DomainInfoResponseCapture(ClaudeAiResponseCapture):
    """Captured GET /api/web/domain_info response."""

    body: DomainInfoResponse


# ==============================================================================
# Documentation Fetches (code.claude.com, platform.claude.com)
# ==============================================================================


class CodeClaudeComRequestCapture(RequestCapture):
    """Base for code.claude.com requests."""

    host: Literal['code.claude.com']


class CodeClaudeComResponseCapture(ResponseCapture):
    """Base for code.claude.com responses."""

    host: Literal['code.claude.com']


class PlatformClaudeComRequestCapture(RequestCapture):
    """Base for platform.claude.com requests."""

    host: Literal['platform.claude.com']


class PlatformClaudeComResponseCapture(ResponseCapture):
    """Base for platform.claude.com responses."""

    host: Literal['platform.claude.com']


class DocFetchRequestCapture(RequestCapture):
    """Captured GET request for documentation files (markdown, txt)."""

    method: Literal['GET']
    body: EmptyBody


class DocFetchResponseCapture(ResponseCapture):
    """Captured GET response for documentation files (raw text)."""

    body: RawTextBody  # Markdown/text content wrapped in raw_text


# Specific doc fetch captures for each host
class CodeClaudeComDocRequestCapture(CodeClaudeComRequestCapture):
    """Captured GET request for code.claude.com docs."""

    method: Literal['GET']
    body: EmptyBody


class CodeClaudeComDocResponseCapture(CodeClaudeComResponseCapture):
    """Captured GET response for code.claude.com docs."""

    body: RawTextBody


class PlatformClaudeComDocRequestCapture(PlatformClaudeComRequestCapture):
    """Captured GET request for platform.claude.com docs."""

    method: Literal['GET']
    body: EmptyBody


class PlatformClaudeComDocResponseCapture(PlatformClaudeComResponseCapture):
    """Captured GET response for platform.claude.com docs."""

    body: RawTextBody
