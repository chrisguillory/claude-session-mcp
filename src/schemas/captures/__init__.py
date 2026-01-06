"""
HTTP traffic capture modeling with type-safe validation.

This module provides Pydantic v2 models for capturing, validating, and
analyzing HTTP traffic from the Claude Code observability platform.

Usage:
    from src.schemas.captures import load_capture, CapturedTraffic

    capture = load_capture(Path("captures/req_001_api_anthropic_com_v1_messages.json"))

    if isinstance(capture, MessagesRequestCapture):
        print(f"Model: {capture.body.model}")
        print(f"Messages: {len(capture.body.messages)}")

For the full philosophy and design decisions, see README.md in this directory.
"""

from __future__ import annotations

from src.schemas.captures.anthropic import (
    ApiError,
    ApiErrorDetail,
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
from src.schemas.captures.base import (
    AnthropicRequestCapture,
    AnthropicResponseCapture,
    ClientConnection,
    ConnectionTiming,
    IPv4Address,
    IPv6Address,
    NetworkAddress,
    RequestCapture,
    ResponseCapture,
    ServerConnection,
    StatsigRequestCapture,
    StatsigResponseCapture,
)
from src.schemas.captures.datadog import (
    DatadogAgentLogEntry,
    DatadogAgentQueryLogEntry,
    DatadogBaseLogEntry,
    DatadogLogEntry,
    DatadogQueryLogEntry,
    DatadogRequestCapture,
    DatadogResponseCapture,
)
from src.schemas.captures.gcs import (
    GCSVersionRequestCapture,
    GCSVersionResponseCapture,
    RawTextBody,
    UnknownRequestCapture,
    UnknownResponseCapture,
)
from src.schemas.captures.loader import (
    CapturedTraffic,
    load_capture,
    load_captures_batch,
)
from src.schemas.captures.registry import (
    CAPTURE_REGISTRY,
    extract_endpoint_from_filename,
    get_capture_type,
    normalize_path,
)
from src.schemas.captures.statsig import (
    StatsigInitializeRequestCapture,
    StatsigInitializeResponseCapture,
    StatsigRegisterRequestCapture,
    StatsigRegisterResponseCapture,
)

__all__ = [
    # Main API
    'CapturedTraffic',
    'load_capture',
    'load_captures_batch',
    # Registry
    'CAPTURE_REGISTRY',
    'get_capture_type',
    'normalize_path',
    'extract_endpoint_from_filename',
    # Base classes
    'RequestCapture',
    'ResponseCapture',
    'AnthropicRequestCapture',
    'AnthropicResponseCapture',
    'StatsigRequestCapture',
    'StatsigResponseCapture',
    # Connection metadata
    'ConnectionTiming',
    'ClientConnection',
    'ServerConnection',
    # Network address types
    'IPv4Address',
    'IPv6Address',
    'NetworkAddress',
    # Anthropic captures
    'MessagesRequestCapture',
    'MessagesStreamResponseCapture',
    'MessagesJsonResponseCapture',
    'ApiError',
    'ApiErrorDetail',
    'TelemetryRequestCapture',
    'TelemetryResponseCapture',
    'CountTokensRequestCapture',
    'CountTokensResponseCapture',
    'EvalRequestCapture',
    'EvalResponseCapture',
    'MetricsRequestCapture',
    'MetricsResponseCapture',
    'MetricsEnabledRequestCapture',
    'MetricsEnabledResponseCapture',
    'HelloRequestCapture',
    'HelloResponseCapture',
    'ClientDataRequestCapture',
    'ClientDataResponseCapture',
    # Statsig captures
    'StatsigRegisterRequestCapture',
    'StatsigRegisterResponseCapture',
    'StatsigInitializeRequestCapture',
    'StatsigInitializeResponseCapture',
    # Datadog captures
    'DatadogRequestCapture',
    'DatadogResponseCapture',
    'DatadogBaseLogEntry',
    'DatadogQueryLogEntry',
    'DatadogAgentLogEntry',
    'DatadogAgentQueryLogEntry',
    'DatadogLogEntry',
    # GCS captures
    'GCSVersionRequestCapture',
    'GCSVersionResponseCapture',
    'RawTextBody',
    # Fallback captures
    'UnknownRequestCapture',
    'UnknownResponseCapture',
]
