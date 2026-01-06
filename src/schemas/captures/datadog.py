"""
Datadog telemetry endpoint capture classes.

This module contains capture wrappers for Datadog logging endpoints:
- /api/v2/logs - Log ingestion

Datadog log entries have a polymorphic structure with base fields always present
and optional query/agent context fields. The union type handles discrimination.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from src.schemas.captures.base import RequestCapture, ResponseCapture
from src.schemas.cc_internal_api.base import EmptyDict, StrictModel

# ==============================================================================
# Datadog Log Entry Types
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


# Union of all Datadog log entry types.
# IMPORTANT: Order matters! Pydantic v2 uses left-to-right matching for non-discriminated unions.
# Most specific types (with more fields) must come first, or they'll never match.
# Order: AgentQuery (4 extra) > Agent (2 extra) > Query (2 extra) > Base (0 extra)
DatadogLogEntry = DatadogAgentQueryLogEntry | DatadogAgentLogEntry | DatadogQueryLogEntry | DatadogBaseLogEntry


# ==============================================================================
# Datadog Capture Types
# ==============================================================================


class DatadogRequestCapture(RequestCapture):
    """Captured POST /api/v2/logs request (Datadog logging)."""

    host: Literal['http-intake.logs.us5.datadoghq.com']
    method: Literal['POST']
    body: Sequence[DatadogLogEntry]


class DatadogResponseCapture(ResponseCapture):
    """
    Captured POST /api/v2/logs response (202 Accepted).

    Datadog returns empty JSON on success.
    """

    host: Literal['http-intake.logs.us5.datadoghq.com']
    body: EmptyDict  # Always {} on 202 success
