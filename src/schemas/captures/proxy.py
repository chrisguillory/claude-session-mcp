"""
Infrastructure-layer types for the mitmproxy system.

This module contains fully-typed captures representing the proxy
infrastructure itself, not the HTTP traffic flowing through it.
These types are stable and well-understood, representing mitmproxy's
internal state and error conditions.

This is separate from application-layer captures (captures.py) because
proxy infrastructure concerns operate at a different architectural layer
than domain/service captures. The distinction:

- captures.py: Application layer - incomplete domain knowledge (unmapped endpoints)
- proxy.py: Infrastructure layer - complete infrastructure knowledge (proxy events)

Both share one property (no service host) but represent categorically
different concerns that should evolve independently.
"""

from __future__ import annotations

from typing import Literal

from src.schemas.cc_internal_api.base import StrictModel

# ==============================================================================
# Proxy Error (direction="error")
# ==============================================================================


class ProxyErrorRequest(StrictModel):
    """Nested request info in a proxy error capture."""

    host: str
    url: str  # Full URL
    method: str


class ProxyErrorDetail(StrictModel):
    """Error detail in a proxy error capture."""

    message: str  # e.g., "Client disconnected."
    timestamp: float  # Unix timestamp


class ProxyErrorCapture(StrictModel):
    """
    Capture of mitmproxy infrastructure errors.

    These represent failures in the proxy system itself, not in
    the application traffic being proxied. Examples include:
    - Client disconnections during transfer
    - TLS handshake failures
    - Upstream connection timeouts

    This type is fully specified because mitmproxy's error structure
    is well-defined and stable. Unlike UnknownRequestCapture (which
    represents gaps in domain knowledge), these errors are completely
    understood infrastructure events.

    Has direction="error" and contains error message + original request.
    """

    direction: Literal['error']
    error: ProxyErrorDetail  # Nested error info
    request: ProxyErrorRequest  # Original request that failed
    flow_id: str  # Flow identifier from mitmproxy
    sequence: int  # Sequence number in capture session
    session_id: str  # Claude Code session ID
    timestamp_iso: str  # ISO timestamp
