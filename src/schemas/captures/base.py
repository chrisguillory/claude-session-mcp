"""
Base capture classes for HTTP traffic modeling.

This module defines the foundational classes for capturing HTTP requests
and responses from the Claude Code observability platform.

Architecture:
- RequestCapture / ResponseCapture: Separate bases for requests vs responses
- Service-level inheritance: AnthropicRequestCapture, StatsigRequestCapture, etc.

Design Decision: Decomposed Request/Response Bases

HTTP requests and responses are fundamentally different entities. Using a
shared base class with `status_code: int | None` loses type precision and
prevents fail-fast validation. Decomposed bases provide:
- Type precision: ResponseCapture.status_code is `int`, not `int | None`
- Fail-fast: Missing status_code on response fails validation immediately
- Clean semantics: No defensive coding for impossible None cases
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field

from src.schemas.cc_internal_api.base import StrictModel

# ==============================================================================
# Network address types
# ==============================================================================


def _coerce_to_tuple(v: Any) -> Any:
    """Convert lists to tuples (JSON doesn't have tuples, only arrays)."""
    if isinstance(v, list):
        return tuple(v)
    return v


# IPv4 address: (host, port)
IPv4Address = Annotated[tuple[str, int], BeforeValidator(_coerce_to_tuple)]

# IPv6 address: (host, port, flowinfo, scope_id)
IPv6Address = Annotated[tuple[str, int, int, int], BeforeValidator(_coerce_to_tuple)]

# Network address can be either IPv4 or IPv6
NetworkAddress = IPv4Address | IPv6Address

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
    address: NetworkAddress = Field(
        description='Client address: IPv4 (host, port) or IPv6 (host, port, flowinfo, scope_id)'
    )
    tls_version: str | None = None
    timing: ConnectionTiming


class ServerConnection(StrictModel):
    """Server connection metadata from mitmproxy."""

    id: str = Field(description='Connection UUID')
    address: NetworkAddress = Field(
        description='Server address: IPv4 (host, port) or IPv6 (host, port, flowinfo, scope_id)'
    )
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
    # Cookie values can be strings or lists (e.g., GCLB includes [value, attrs])
    cookies: Mapping[str, str | Sequence[str]] = Field(description='HTTP cookies')
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
    # Cookie values can be strings or lists (e.g., GCLB includes [value, attrs])
    cookies: Mapping[str, str | Sequence[str]] = Field(description='HTTP cookies')
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
