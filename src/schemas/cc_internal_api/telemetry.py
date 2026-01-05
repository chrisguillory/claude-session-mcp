"""
Telemetry schemas for Claude Code internal API.

Claude Code reports telemetry to /api/event_logging/batch.
All events use the "tengu" prefix (Claude Code's internal codename).

Privacy controls:
- DISABLE_TELEMETRY=1 environment variable
- DISABLE_ERROR_REPORTING=1 environment variable
- Settings in ~/.config/claude-code/settings.json
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal

from src.schemas.cc_internal_api.base import PermissiveModel

# ==============================================================================
# Type Aliases (strict Literal types - add new values when discovered)
# ==============================================================================

Platform = Literal['darwin', 'linux', 'win32', 'freebsd', 'openbsd']
Arch = Literal['arm64', 'x64', 'x86', 'arm']
Entrypoint = Literal['cli', 'api', 'sdk']
ClientType = Literal['cli', 'sdk']
UserType = Literal['external', 'internal']


# ==============================================================================
# Environment Information
# ==============================================================================


class TelemetryEnv(PermissiveModel):
    """
    Environment information included in telemetry events.

    VALIDATION STATUS: VALIDATED
    Observed in event_data.env.
    """

    platform: Platform
    node_version: str
    terminal: str  # e.g., "pycharm", "vscode", "iterm2"
    package_managers: str  # e.g., "npm", "yarn,pnpm"
    runtimes: str  # e.g., "bun,node"
    is_running_with_bun: bool
    is_ci: bool
    is_claubbit: bool  # Whether running as Claubbit bot
    is_github_action: bool
    is_claude_code_action: bool
    is_claude_ai_auth: bool  # Using claude.ai OAuth vs API key
    version: str  # Claude Code version, e.g., "2.0.76"
    arch: Arch
    is_claude_code_remote: bool
    deployment_environment: str  # e.g., "unknown-darwin"


# ==============================================================================
# Event Data
# ==============================================================================


class TelemetryEventData(PermissiveModel):
    """
    Core telemetry event data.

    VALIDATION STATUS: VALIDATED
    Observed in events[].event_data.

    Note: additional_metadata is a JSON-encoded string with event-specific data.
    """

    event_name: str  # e.g., "tengu_api_success", "tengu_mcp_servers"
    client_timestamp: str  # ISO 8601 timestamp
    model: str  # Model being used
    session_id: str  # Session UUID
    user_type: UserType
    betas: str  # Comma-separated beta features
    env: TelemetryEnv
    entrypoint: Entrypoint
    is_interactive: bool
    client_type: ClientType
    device_id: str  # Hashed device identifier

    # Optional - JSON-encoded string with event-specific metadata
    additional_metadata: str | None = None

    def get_metadata(self) -> dict[str, Any]:
        """Parse additional_metadata JSON string to dict."""
        if not self.additional_metadata:
            return {}
        result = json.loads(self.additional_metadata)
        if not isinstance(result, dict):
            msg = f'Expected dict, got {type(result).__name__}'
            raise TypeError(msg)
        return dict(result)


# ==============================================================================
# Event Wrapper
# ==============================================================================


class TelemetryEvent(PermissiveModel):
    """
    Single telemetry event in a batch.

    VALIDATION STATUS: VALIDATED
    Observed in events[].
    """

    event_type: Literal['ClaudeCodeInternalEvent']
    event_data: TelemetryEventData


# ==============================================================================
# Batch Request/Response
# ==============================================================================


class TelemetryBatchRequest(PermissiveModel):
    """
    Batch telemetry request to /api/event_logging/batch.

    VALIDATION STATUS: VALIDATED
    Observed in request body.
    """

    events: Sequence[TelemetryEvent]


class TelemetryBatchResponse(PermissiveModel):
    """
    Response from /api/event_logging/batch.

    VALIDATION STATUS: INFERRED
    Expected based on API patterns.
    """

    accepted_count: int
    rejected_count: int


# ==============================================================================
# Common Event Names (observed)
# ==============================================================================

# Documented for reference, not enforced as a type
KNOWN_EVENT_NAMES = [
    # Lifecycle
    'tengu_init',
    'tengu_exit',
    'tengu_startup_telemetry',
    'tengu_startup_manual_model_config',
    # API interactions
    'tengu_api_query',
    'tengu_api_success',
    'tengu_api_before_normalize',
    'tengu_api_after_normalize',
    'tengu_api_cache_breakpoints',
    # Context management
    'tengu_context_size',
    'tengu_sysprompt_block',
    # MCP
    'tengu_mcp_servers',
    'tengu_mcp_server_connection_succeeded',
    'tengu_mcp_ide_server_connection_succeeded',
    'tengu_mcp_tools_commands_loaded',
    'tengu_mcp_cli_status',
    # Version management
    'tengu_version_check_success',
    'tengu_version_check_failure',
    'tengu_version_lock_failed',
    'tengu_native_auto_updater_start',
    'tengu_native_auto_updater_success',
    'tengu_native_auto_updater_fail',
    'tengu_native_update_complete',
    # User interactions
    'tengu_input_prompt',
    'tengu_prompt_suggestion_init',
    'tengu_paste_text',
    'tengu_trust_dialog_shown',
    # Tool/Agent usage
    'tengu_agent_tool_selected',
    'tengu_agent_tool_completed',
    'tengu_fork_agent_query',
    # File operations
    'tengu_file_operation',
    'tengu_dir_search',
    'tengu_attachments',
    'tengu_attachment_compute_duration',
    # Shell/Hooks
    'tengu_shell_set_cwd',
    'tengu_run_hook',
    'tengu_repl_hook_finished',
    # Configuration
    'tengu_config_cache_stats',
    'tengu_config_stale_write',
    # Performance
    'tengu_timer',
    'tengu_thinking',
    # Notifications
    'tengu_notification_method_used',
    'tengu_claudeai_limits_status_changed',
    # Features
    'tengu_ripgrep_availability',
]
