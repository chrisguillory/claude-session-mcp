"""
Datadog telemetry endpoint capture classes.

This module contains capture wrappers for Datadog logging endpoints:
- /api/v2/logs - Log ingestion

Datadog log entries are discriminated by the `message` field:
- tengu_api_success: API response success logs (full metrics)
- tengu_api_error: API error logs (partial metrics)
- tengu_tool_use_success: Tool execution logs (different shape)
- tengu_oauth_success: OAuth success logs (minimal fields)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal

import pydantic

from src.schemas.captures.base import RequestCapture, ResponseCapture
from src.schemas.cc_internal_api.base import EmptyDict, StrictModel

# ==============================================================================
# Shared Base (32 fields present in ALL log entry types)
# ==============================================================================


class DatadogLogEntryBase(StrictModel):
    """
    Base log entry fields present in ALL Datadog log types.

    Contains 32 fields that appear in every log entry regardless of type.
    Subclasses add type-specific fields and override `message` with a Literal.
    """

    # --- Datadog standard fields ---
    ddsource: str  # "nodejs"
    ddtags: str  # Comma-separated tags
    message: str  # Event type - overridden with Literal in subclasses
    service: str  # "claude-code"
    hostname: str  # "claude-code"
    env: str  # "external"

    # --- Claude Code context ---
    model: str  # Model ID
    session_id: str  # Session UUID
    user_type: str  # "external" or "internal"
    betas: str  # Comma-separated beta features
    entrypoint: str  # "cli"
    is_interactive: str  # "true" or "false" (STRING, not bool!)
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


# ==============================================================================
# API Success Log Entry (message="tengu_api_success")
# ==============================================================================


class DatadogApiSuccessLogEntry(DatadogLogEntryBase):
    """
    API success log entry with full token/cost/timing metrics.

    Discriminated by message="tengu_api_success".

    INNER BIFURCATION TODO: Further split by query context:
    - AgentChain: agentId + queryChainId (REQUIRED)
    - AgentSimple: agentId only (REQUIRED)
    - MainChain: queryChainId (REQUIRED), preNormalizedModel (OPTIONAL)
    - Utility: base fields only
    - Compact: preNormalizedModel (REQUIRED)
    """

    message: Literal['tengu_api_success']

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
    stop_reason: str  # "end_turn", "tool_use"
    cost_u_s_d: float  # Cost in USD

    # --- Session flags ---
    did_fall_back_to_non_streaming: bool
    is_non_interactive_session: bool
    print: bool
    is_t_t_y: bool

    # --- Query context ---
    query_source: str  # "repl_main_thread", "prompt_suggestion", etc.
    permission_mode: str  # "default"

    # --- Optional context fields (TODO: bifurcate by querySource) ---
    # These vary based on querySource - inner bifurcation needed
    pre_normalized_model: str | None = None  # TODO: bifurcate by querySource
    query_chain_id: str | None = None  # TODO: bifurcate by querySource
    query_depth: int | None = None  # TODO: bifurcate by querySource
    agent_id: str | None = None  # TODO: bifurcate by querySource
    agent_type: str | None = None  # TODO: bifurcate by querySource


# ==============================================================================
# API Error Log Entry (message="tengu_api_error")
# ==============================================================================


class DatadogApiErrorLogEntry(DatadogLogEntryBase):
    """
    API error log entry with error details but no token/cost metrics.

    Discriminated by message="tengu_api_error".
    Missing from errors: input_tokens, output_tokens, cached_input_tokens,
    uncached_input_tokens, ttft_ms, build_age_mins, stop_reason, cost_u_s_d,
    is_non_interactive_session, print, is_t_t_y, permission_mode
    """

    message: Literal['tengu_api_error']

    # --- Error info ---
    error: str  # Error message text
    error_type: str  # "client_error", "unknown", "prompt_too_long"

    # --- Partial metrics (NO token counts!) ---
    message_count: int
    message_tokens: int
    duration_ms: int
    duration_ms_including_retries: int
    attempt: int
    provider: str
    did_fall_back_to_non_streaming: bool

    # --- Query context ---
    query_chain_id: str
    query_depth: int
    query_source: str

    # --- Optional HTTP status ---
    http_status: str | None = None  # genuinely optional: (e.g., "400", "undefined")
    http_status_range: str | None = None  # genuinely optional: (e.g., "4xx")
    request_id: str | None = None  # genuinely optional: (not always present on errors)


# ==============================================================================
# Tool Use Success Log Entry (message="tengu_tool_use_success")
# ==============================================================================


class DatadogToolUseSuccessLogEntry(DatadogLogEntryBase):
    """
    Tool execution log entry - completely different structure from API logs.

    Discriminated by message="tengu_tool_use_success".
    Does NOT have: token counts, costs, ttft, stop_reason, provider, etc.
    """

    message: Literal['tengu_tool_use_success']

    # --- Tool execution metrics ---
    message_i_d: str  # NOTE: underscore naming, not camelCase! (msg_...)
    tool_name: str  # e.g., "Read", "mcp"
    is_mcp: bool  # Whether it's an MCP tool
    duration_ms: int  # Tool execution time
    tool_result_size_bytes: int  # Size of tool result

    # --- Query context ---
    query_chain_id: str
    query_depth: int
    request_id: str  # The API request that triggered this tool

    # --- Agent context (when tool used by agent) ---
    agent_id: str | None = None  # genuinely optional: (only for agent tool calls)
    agent_type: str | None = None  # genuinely optional: (only for agent tool calls)

    # --- MCP-specific (only when is_mcp=True) ---
    mcp_server_type: str | None = None  # genuinely optional: (e.g., "stdio")


# ==============================================================================
# OAuth Success Log Entry (message="tengu_oauth_success")
# ==============================================================================


class DatadogOAuthSuccessLogEntry(DatadogLogEntryBase):
    """
    OAuth success log entry - minimal fields beyond base.

    Discriminated by message="tengu_oauth_success".
    """

    message: Literal['tengu_oauth_success']

    login_with_claude_ai: bool  # Whether using Claude AI auth


# ==============================================================================
# Discriminator Function and Union
# ==============================================================================


def _get_datadog_log_entry_type(v: Any) -> str:
    """
    Callable discriminator for DatadogLogEntry union.

    Routes based on `message` field value.
    """
    if isinstance(v, dict):
        message = v.get('message', '')
    else:
        message = getattr(v, 'message', '')

    message_to_tag = {
        'tengu_api_success': 'api_success',
        'tengu_api_error': 'api_error',
        'tengu_tool_use_success': 'tool_use_success',
        'tengu_oauth_success': 'oauth_success',
    }
    # Fallback for unknown message types - use api_success as default
    # This may fail validation, which is the desired behavior
    return message_to_tag.get(message, 'api_success')


DatadogLogEntry = Annotated[
    Annotated[DatadogApiSuccessLogEntry, pydantic.Tag('api_success')]
    | Annotated[DatadogApiErrorLogEntry, pydantic.Tag('api_error')]
    | Annotated[DatadogToolUseSuccessLogEntry, pydantic.Tag('tool_use_success')]
    | Annotated[DatadogOAuthSuccessLogEntry, pydantic.Tag('oauth_success')],
    pydantic.Discriminator(_get_datadog_log_entry_type),
]


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
