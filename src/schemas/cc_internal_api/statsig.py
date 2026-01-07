"""
Statsig schemas for Claude Code feature flag system.

Claude Code uses Statsig (statsig.anthropic.com) for A/B testing and feature flags.
This is separate from the Anthropic-internal `/api/eval/sdk-*` feature flags.

Endpoints:
- /v1/initialize - Initialize feature flag state
- /v1/rgstr - Register/log feature flag events

BIFURCATION STATUS: Fully bifurcated based on observed capture data.
- StatsigMetadata: fallbackUrl always present (no default)
- StatsigInitialize*: Full vs Delta responses discriminated by is_delta
- StatsigMarker*: 9 shapes discriminated by action + field presence
- StatsigEvent*: 3 shapes discriminated by eventName pattern
- Statsig*EventMetadata: 43+ per-event-type metadata classes
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

import pydantic

from src.schemas.cc_internal_api.base import EmptyBody, EmptyDict, StrictModel

# ==============================================================================
# Common Types
# ==============================================================================


class StatsigCustomIDs(StrictModel):
    """Custom identifiers for Statsig user tracking.

    organizationUUID and accountUUID are optional - missing during OAuth flow
    before user is fully authenticated.
    """

    sessionId: str
    organizationUUID: str | None = None  # Missing during OAuth
    accountUUID: str | None = None  # Missing during OAuth


class StatsigCustom(StrictModel):
    """Custom user attributes for feature targeting.

    organizationUuid and accountUuid are optional - missing during OAuth flow
    before user is fully authenticated. subscriptionType can be empty string
    during OAuth before subscription is known.
    """

    userType: Literal['external', 'internal']
    organizationUuid: str | None = None  # Missing during OAuth
    accountUuid: str | None = None  # Missing during OAuth
    subscriptionType: Literal['', 'free', 'pro', 'team', 'max', 'enterprise']
    firstTokenTime: int


class StatsigEnvironment(StrictModel):
    """Statsig environment configuration."""

    tier: Literal['production', 'staging', 'development']


class StatsigUser(StrictModel):
    """User object for Statsig requests."""

    userID: str
    appVersion: str
    customIDs: StatsigCustomIDs
    custom: StatsigCustom
    statsigEnvironment: StatsigEnvironment


class StatsigMetadata(StrictModel):
    """
    Statsig SDK metadata.

    BIFURCATION: fallbackUrl is always present (no default), value is always None.
    """

    sdkType: str
    sdkVersion: str
    stableID: str
    sessionID: str
    fallbackUrl: None  # Always present, always None in captures


# ==============================================================================
# Initialize Endpoint (/v1/initialize)
# ==============================================================================


class StatsigEvaluatedKeys(StrictModel):
    """Evaluated user keys in Statsig initialize response."""

    userID: str
    stableID: str
    customIDs: StatsigCustomIDs


class AutoCaptureDisabledEvents(StrictModel):
    """Disabled events configuration. Always {} in observed captures."""

    pass


class AutoCaptureSettings(StrictModel):
    """Auto-capture settings for session recording."""

    disabled_events: AutoCaptureDisabledEvents


class DerivedFields(StrictModel):
    """Derived fields from Statsig server, echoed back in subsequent requests."""

    ip: str
    country: str
    appVersion: str
    app_version: str
    browserName: str
    browserVersion: str
    osName: str
    osVersion: str
    browser_name: str
    browser_version: str
    os_name: str
    os_version: str


class StatsigInitializeRequest(StrictModel):
    """Request to /v1/initialize."""

    user: StatsigUser
    statsigMetadata: StatsigMetadata
    sinceTime: int
    hash: str
    deltasResponseRequested: bool
    full_checksum: str
    previousDerivedFields: DerivedFields | EmptyDict


class StatsigFeatureGate(StrictModel):
    """Feature gate value in Statsig initialize response."""

    name: str
    value: bool
    rule_id: str
    id_type: str
    secondary_exposures: Sequence[Mapping[str, str]]


class StatsigBaseDynamicConfig(StrictModel):
    """Base fields for all Statsig dynamic configs."""

    name: str
    value: Mapping[str, Any]  # noqa: loose-typing - genuinely polymorphic config payloads
    rule_id: str
    group: str
    is_device_based: bool
    id_type: str
    secondary_exposures: Sequence[Mapping[str, str]]


class StatsigEvaluatedConfig(StatsigBaseDynamicConfig):
    """Dynamic config that was directly evaluated (has `passed` field)."""

    passed: bool


class StatsigExperimentConfig(StatsigBaseDynamicConfig):
    """Experiment-type dynamic config (has experiment tracking fields)."""

    is_user_in_experiment: bool
    is_experiment_active: bool


class StatsigNamedExperimentConfig(StatsigExperimentConfig):
    """Named experiment config (experiment + group_name)."""

    group_name: str


StatsigDynamicConfig = StatsigNamedExperimentConfig | StatsigExperimentConfig | StatsigEvaluatedConfig


# ==============================================================================
# Initialize Response - BIFURCATED into Full vs Delta
# ==============================================================================


class StatsigInitializeFullResponse(StrictModel):
    """
    Full initialize response (initial request, no delta fields).

    BIFURCATION: This is the response when is_delta is absent.
    All 18 fields are required, no defaults.
    """

    feature_gates: Mapping[str, StatsigFeatureGate]
    dynamic_configs: Mapping[str, StatsigDynamicConfig]
    layer_configs: EmptyDict
    has_updates: bool
    generator: str
    time: int
    company_lcut: int
    hash_used: str
    hashed_sdk_key_used: str  # Always present (was optional)
    sdkParams: EmptyDict
    evaluated_keys: StatsigEvaluatedKeys
    derived_fields: DerivedFields
    can_record_session: bool
    recording_blocked: bool
    session_recording_rate: float
    auto_capture_settings: AutoCaptureSettings
    target_app_used: str  # Always present (was optional)
    full_checksum: str


class StatsigInitializeDeltaResponse(StrictModel):
    """
    Delta initialize response (subsequent request with updates).

    BIFURCATION: This is the response when is_delta=true.
    Includes all base fields plus delta-specific fields.
    """

    feature_gates: Mapping[str, StatsigFeatureGate]
    dynamic_configs: Mapping[str, StatsigDynamicConfig]
    layer_configs: EmptyDict
    has_updates: bool
    generator: str
    time: int
    company_lcut: int
    hash_used: str
    hashed_sdk_key_used: str
    sdkParams: EmptyDict
    evaluated_keys: StatsigEvaluatedKeys
    derived_fields: DerivedFields
    can_record_session: bool
    recording_blocked: bool
    session_recording_rate: float
    auto_capture_settings: AutoCaptureSettings
    target_app_used: str
    full_checksum: str
    # Delta-specific fields (always present in delta responses)
    is_delta: Literal[True]  # Discriminator
    deleted_configs: Sequence[str]
    deleted_gates: Sequence[str]
    deleted_layers: Sequence[str]
    checksum: str
    checksumV2: str
    deltas_full_response: None  # Always null in captures


def _discriminate_initialize_response(v: Any) -> str:
    """Discriminator for initialize response types."""
    if isinstance(v, dict):
        if v.get('empty') is True:
            return 'empty'
        if v.get('is_delta') is True:
            return 'delta'
        return 'full'
    return 'full'


StatsigInitializeResponse = Annotated[
    Annotated[EmptyBody, pydantic.Tag('empty')]
    | Annotated[StatsigInitializeDeltaResponse, pydantic.Tag('delta')]
    | Annotated[StatsigInitializeFullResponse, pydantic.Tag('full')],
    pydantic.Discriminator(_discriminate_initialize_response),
]


# ==============================================================================
# Statsig Marker Types - BIFURCATED into 9 shapes
# ==============================================================================


class StatsigMarkerEvaluationDetails(StrictModel):
    """Evaluation details in a diagnostics marker."""

    reason: str
    lcut: int
    receivedAt: int


class StatsigMarkerErrorWithCode(StrictModel):
    """Error details with error code in a diagnostics marker."""

    name: str
    message: str
    code: int


class StatsigMarkerErrorWithoutCode(StrictModel):
    """Error details without error code in a diagnostics marker."""

    name: str
    message: str


StatsigMarkerError = StatsigMarkerErrorWithCode | StatsigMarkerErrorWithoutCode


# Start markers (action='start')
class StatsigMarkerStartBase(StrictModel):
    """Start marker with just base fields (15 instances)."""

    key: str
    action: Literal['start']
    timestamp: int


class StatsigMarkerStartWithStep(StrictModel):
    """Start marker with step field (7 instances)."""

    key: str
    action: Literal['start']
    timestamp: int
    step: str


class StatsigMarkerStartWithStepAttempt(StrictModel):
    """Start marker with step and attempt fields (15 instances)."""

    key: str
    action: Literal['start']
    timestamp: int
    step: str
    attempt: int


# End markers (action='end')
class StatsigMarkerEndStepSuccess(StrictModel):
    """End marker with step and success (7 instances)."""

    key: str
    action: Literal['end']
    timestamp: int
    step: str
    success: bool


class StatsigMarkerEndEvaluation(StrictModel):
    """End marker with evaluationDetails and success (7 instances)."""

    key: str
    action: Literal['end']
    timestamp: int
    success: bool
    evaluationDetails: StatsigMarkerEvaluationDetails


class StatsigMarkerEndEvaluationError(StrictModel):
    """End marker with evaluationDetails, success, and error (8 instances)."""

    key: str
    action: Literal['end']
    timestamp: int
    success: bool
    evaluationDetails: StatsigMarkerEvaluationDetails
    error: StatsigMarkerError


class StatsigMarkerEndNetworkSuccess(StrictModel):
    """End marker with network fields, no error (1 instance)."""

    key: str
    action: Literal['end']
    timestamp: int
    step: str
    attempt: int
    success: bool
    statusCode: int
    sdkRegion: str


class StatsigMarkerEndNetworkDelta(StrictModel):
    """End marker with network fields and isDelta (6 instances)."""

    key: str
    action: Literal['end']
    timestamp: int
    step: str
    attempt: int
    success: bool
    statusCode: int
    sdkRegion: str
    isDelta: bool


class StatsigMarkerEndNetworkError(StrictModel):
    """End marker with network fields and error (8 instances)."""

    key: str
    action: Literal['end']
    timestamp: int
    step: str
    attempt: int
    success: bool
    statusCode: int
    sdkRegion: str
    error: StatsigMarkerError


def _discriminate_marker(v: Any) -> str:
    """Discriminator for marker types based on action and fields."""
    if not isinstance(v, dict):
        return 'start_base'
    action = v.get('action')
    if action == 'start':
        if 'attempt' in v:
            return 'start_step_attempt'
        if 'step' in v:
            return 'start_step'
        return 'start_base'
    # action == 'end'
    if 'evaluationDetails' in v:
        if 'error' in v:
            return 'end_eval_error'
        return 'end_eval'
    if 'sdkRegion' in v:
        if 'error' in v:
            return 'end_network_error'
        if 'isDelta' in v:
            return 'end_network_delta'
        return 'end_network_success'
    return 'end_step_success'


StatsigMarker = Annotated[
    Annotated[StatsigMarkerStartBase, pydantic.Tag('start_base')]
    | Annotated[StatsigMarkerStartWithStep, pydantic.Tag('start_step')]
    | Annotated[StatsigMarkerStartWithStepAttempt, pydantic.Tag('start_step_attempt')]
    | Annotated[StatsigMarkerEndStepSuccess, pydantic.Tag('end_step_success')]
    | Annotated[StatsigMarkerEndEvaluation, pydantic.Tag('end_eval')]
    | Annotated[StatsigMarkerEndEvaluationError, pydantic.Tag('end_eval_error')]
    | Annotated[StatsigMarkerEndNetworkSuccess, pydantic.Tag('end_network_success')]
    | Annotated[StatsigMarkerEndNetworkDelta, pydantic.Tag('end_network_delta')]
    | Annotated[StatsigMarkerEndNetworkError, pydantic.Tag('end_network_error')],
    pydantic.Discriminator(_discriminate_marker),
]


# ==============================================================================
# Statsig Options (for diagnostics events)
# ==============================================================================


class StatsigOptionsNetworkConfig(StrictModel):
    """Network config in Statsig options."""

    api: str


class StatsigOptionsEnvironment(StrictModel):
    """Environment in Statsig options."""

    tier: str


class StatsigOptionsStorageProviderCache(StrictModel):
    """Storage provider cache. Always {} in observed captures."""

    pass


class StatsigOptionsStorageProvider(StrictModel):
    """Storage provider in Statsig options."""

    cache: StatsigOptionsStorageProviderCache
    ready: bool


class StatsigOptions(StrictModel):
    """Statsig SDK options in diagnostics events."""

    networkConfig: StatsigOptionsNetworkConfig
    environment: StatsigOptionsEnvironment
    includeCurrentPageUrlWithEvents: bool
    logLevel: int
    storageProvider: StatsigOptionsStorageProvider


# ==============================================================================
# Event Metadata Types - BIFURCATED per event type
# ==============================================================================


# Common base for most tengu_* events
class TenguEventMetadataBase(StrictModel):
    """
    Common fields for most tengu_* events.

    These 13 fields appear in nearly all tengu events.
    """

    betas: str
    clientType: str
    entrypoint: str
    env: str
    isInteractive: str
    model: str
    sessionId: str
    sweBenchInstanceId: str
    sweBenchRunId: str
    sweBenchTaskId: str
    userType: str


# Statsig internal events (statsig::*)
class ConfigExposureMetadata(StrictModel):
    """Metadata for statsig::config_exposure events (40 instances)."""

    config: str
    lcut: str
    reason: str
    receivedAt: str
    ruleID: str
    rulePassed: str | None = None  # Sometimes present


class GateExposureMetadata(StrictModel):
    """Metadata for statsig::gate_exposure events (6 instances)."""

    gate: str
    gateValue: str
    lcut: str
    reason: str
    receivedAt: str
    ruleID: str


class DiagnosticsMetadata(StrictModel):
    """Metadata for statsig::diagnostics events (15 instances)."""

    context: str
    markers: Sequence[StatsigMarker]
    statsigOptions: StatsigOptions


# Tengu events - each with specific required fields
class TenguPromptSuggestionInitMetadata(TenguEventMetadataBase):
    """Metadata for tengu_prompt_suggestion_init (16 instances)."""

    enabled: str
    source: str
    agentId: str | None = None  # Sometimes present
    agentType: str | None = None  # Sometimes present


class TenguMcpServersMetadata(TenguEventMetadataBase):
    """Metadata for tengu_mcp_servers (5 instances)."""

    claudeai: str
    enterprise: str
    global_: str = pydantic.Field(alias='global')
    plugin: str
    project: str
    user: str


# TenguApiQueryMetadata - BIFURCATED by querySource discriminator
class TenguApiQueryMetadataBase(TenguEventMetadataBase):
    """Base fields for tengu_api_query events."""

    buildAgeMins: str
    messagesLength: str
    permissionMode: str
    provider: str
    querySource: str
    temperature: str


class TenguApiQueryAgentMetadata(TenguApiQueryMetadataBase):
    """Metadata for tengu_api_query from agent context (querySource starts with 'agent:').

    queryChainId/queryDepth are optional - may be absent for initial agent events
    before query chain is established.
    """

    agentId: str
    agentType: str
    queryChainId: str | None = None  # Optional - absent for initial agent events
    queryDepth: str | None = None  # Optional - absent for initial agent events


class TenguApiQueryMainMetadata(TenguApiQueryMetadataBase):
    """Metadata for tengu_api_query from main thread (querySource='repl_main_thread' or 'prompt_suggestion')."""

    queryChainId: str | None = None  # Present in most but not all
    queryDepth: str | None = None  # Present in most but not all


class TenguApiQueryTerminalMetadata(TenguApiQueryMetadataBase):
    """Metadata for tengu_api_query from terminal (querySource='terminal_update_title')."""

    pass  # No additional fields


TenguApiQueryMetadata = TenguApiQueryAgentMetadata | TenguApiQueryMainMetadata | TenguApiQueryTerminalMetadata


class TenguSyspromptBlockMetadata(TenguEventMetadataBase):
    """Metadata for tengu_sysprompt_block (5 instances)."""

    hash: str
    length: str
    snippet: str
    agentId: str | None = None
    agentType: str | None = None


class TenguApiAfterNormalizeMetadata(TenguEventMetadataBase):
    """Metadata for tengu_api_after_normalize (5 instances)."""

    postNormalizedMessageCount: str
    agentId: str | None = None
    agentType: str | None = None


class TenguApiBeforeNormalizeMetadata(TenguEventMetadataBase):
    """Metadata for tengu_api_before_normalize (5 instances)."""

    preNormalizedMessageCount: str
    agentId: str | None = None
    agentType: str | None = None


class TenguApiCacheBreakpointsMetadata(TenguEventMetadataBase):
    """Metadata for tengu_api_cache_breakpoints (5 instances)."""

    cachingEnabled: str
    totalMessageCount: str
    agentId: str | None = None
    agentType: str | None = None


# TenguApiSuccessMetadata - BIFURCATED by querySource discriminator
class TenguApiSuccessMetadataBase(TenguEventMetadataBase):
    """Base fields for tengu_api_success events."""

    attempt: str
    buildAgeMins: str
    cachedInputTokens: str
    costUSD: str
    didFallBackToNonStreaming: str
    durationMs: str
    durationMsIncludingRetries: str
    inputTokens: str
    isNonInteractiveSession: str
    isTTY: str
    messageCount: str
    messageTokens: str
    outputTokens: str
    permissionMode: str
    print: str
    provider: str
    querySource: str
    requestId: str
    stop_reason: str
    ttftMs: str
    uncachedInputTokens: str


class TenguApiSuccessAgentMetadata(TenguApiSuccessMetadataBase):
    """Metadata for tengu_api_success from agent context (querySource starts with 'agent:').

    queryChainId/queryDepth are optional - may be absent for initial agent events
    before query chain is established.
    """

    agentId: str
    agentType: str
    queryChainId: str | None = None  # Optional - absent for initial agent events
    queryDepth: str | None = None  # Optional - absent for initial agent events


class TenguApiSuccessMainMetadata(TenguApiSuccessMetadataBase):
    """Metadata for tengu_api_success from main thread (querySource='repl_main_thread' or 'prompt_suggestion')."""

    queryChainId: str
    queryDepth: str
    preNormalizedModel: str | None = None  # Present in some captures


class TenguApiSuccessTerminalMetadata(TenguApiSuccessMetadataBase):
    """Metadata for tengu_api_success from terminal (querySource='terminal_update_title')."""

    preNormalizedModel: str | None = None  # Present in some captures


TenguApiSuccessMetadata = TenguApiSuccessAgentMetadata | TenguApiSuccessMainMetadata | TenguApiSuccessTerminalMetadata


class TenguDirSearchMetadata(TenguEventMetadataBase):
    """Metadata for tengu_dir_search (3 instances)."""

    durationMs: str
    managedFilesFound: str
    projectDirsSearched: str
    projectFilesFound: str
    subdir: str
    userFilesFound: str


class TenguAgentToolCompletedMetadata(TenguEventMetadataBase):
    """Metadata for tengu_agent_tool_completed (2 instances)."""

    agentId: str
    agentType: str
    assistant_message_count: str
    duration_ms: str
    is_built_in_agent: str
    prompt_char_count: str
    response_char_count: str
    total_tokens: str
    total_tool_uses: str


class TenguAgentToolSelectedMetadata(TenguEventMetadataBase):
    """Metadata for tengu_agent_tool_selected (2 instances)."""

    agent_type: str
    is_built_in_agent: str
    source: str


class TenguVersionCheckSuccessMetadata(TenguEventMetadataBase):
    """Metadata for tengu_version_check_success (2 instances)."""

    latency_ms: str
    source_gcs: str


class TenguMcpToolsCommandsLoadedMetadata(TenguEventMetadataBase):
    """Metadata for tengu_mcp_tools_commands_loaded (1 instance)."""

    commands_count: str
    commands_metadata_length: str
    tools_count: str


class TenguShellSetCwdMetadata(TenguEventMetadataBase):
    """Metadata for tengu_shell_set_cwd (1 instance)."""

    success: str


class TenguReplHookFinishedMetadata(TenguEventMetadataBase):
    """Metadata for tengu_repl_hook_finished (1 instance)."""

    hookName: str
    numBlocking: str
    numCancelled: str
    numCommands: str
    numNonBlockingError: str
    numSuccess: str


class TenguTimerMetadata(TenguEventMetadataBase):
    """Metadata for tengu_timer (1 instance)."""

    durationMs: str
    event: str


class TenguStartupTelemetryMetadata(TenguEventMetadataBase):
    """Metadata for tengu_startup_telemetry (1 instance)."""

    are_unsandboxed_commands_allowed: str
    auto_updater_disabled: str
    is_auto_bash_allowed_if_sandbox_enabled: str
    is_git: str
    sandbox_enabled: str
    worktree_count: str


class TenguExitMetadata(TenguEventMetadataBase):
    """Metadata for tengu_exit (1 instance)."""

    last_session_api_duration: str
    last_session_cost: str
    last_session_duration: str
    last_session_id: str
    last_session_lines_added: str
    last_session_lines_removed: str
    last_session_tool_duration: str
    last_session_total_cache_creation_input_tokens: str
    last_session_total_cache_read_input_tokens: str
    last_session_total_input_tokens: str
    last_session_total_output_tokens: str


class TenguTrustDialogShownMetadata(TenguEventMetadataBase):
    """Metadata for tengu_trust_dialog_shown (1 instance)."""

    copyVariant: str
    folderType: str
    hasApiKeyHelper: str
    hasAwsCommands: str
    hasBashExecution: str
    hasHooks: str
    hasMcpServers: str
    hasOtelHeadersHelper: str
    isHomeDir: str


class TenguRipgrepAvailabilityMetadata(TenguEventMetadataBase):
    """Metadata for tengu_ripgrep_availability (1 instance)."""

    using_system: str
    working: str


class TenguInitMetadata(TenguEventMetadataBase):
    """Metadata for tengu_init (1 instance)."""

    allowDangerouslySkipPermissionsPassed: str
    dangerouslySkipPermissionsPassed: str
    debug: str
    debugToStderr: str
    hasInitialPrompt: str
    hasStdin: str
    inputFormat: str
    mcpClientCount: str
    modeIsBypass: str
    numAllowedTools: str
    numDisallowedTools: str
    outputFormat: str
    print: str
    rh: str
    verbose: str
    worktree: str


class TenguRunHookMetadata(TenguEventMetadataBase):
    """Metadata for tengu_run_hook (1 instance)."""

    hookName: str
    numCommands: str


class TenguVersionLockFailedMetadata(TenguEventMetadataBase):
    """Metadata for tengu_version_lock_failed (1 instance)."""

    is_lifetime_lock: str
    is_pid_based: str


class TenguNativeAutoUpdaterStartMetadata(TenguEventMetadataBase):
    """Metadata for tengu_native_auto_updater_start (1 instance)."""

    pass  # Just base fields


class TenguNativeAutoUpdaterSuccessMetadata(TenguEventMetadataBase):
    """Metadata for tengu_native_auto_updater_success (1 instance)."""

    latency_ms: str


class TenguConfigStaleWriteMetadata(TenguEventMetadataBase):
    """Metadata for tengu_config_stale_write (1 instance)."""

    read_mtime: str
    read_size: str
    write_mtime: str
    write_size: str


class TenguContextSizeMetadata(TenguEventMetadataBase):
    """Metadata for tengu_context_size (1 instance)."""

    claude_md_size: str
    git_status_size: str
    mcp_servers_count: str
    mcp_tools_count: str
    mcp_tools_tokens: str
    non_mcp_tools_count: str
    non_mcp_tools_tokens: str
    project_file_count_rounded: str
    total_context_size: str


class TenguForkAgentQueryMetadata(TenguEventMetadataBase):
    """Metadata for tengu_fork_agent_query (1 instance)."""

    cacheCreationEphemeral1hTokens: str
    cacheCreationEphemeral5mTokens: str
    cacheCreationInputTokens: str
    cacheHitRate: str
    cacheReadInputTokens: str
    durationMs: str
    forkLabel: str
    inputTokens: str
    messageCount: str
    outputTokens: str
    queryChainId: str
    queryDepth: str
    querySource: str
    serviceTier: str


class TenguThinkingMetadata(TenguEventMetadataBase):
    """Metadata for tengu_thinking (1 instance)."""

    provider: str
    tokenCount: str


class TenguPasteTextMetadata(TenguEventMetadataBase):
    """Metadata for tengu_paste_text (1 instance)."""

    pastedTextCount: str


class TenguInputPromptMetadata(TenguEventMetadataBase):
    """Metadata for tengu_input_prompt (1 instance)."""

    is_keep_going: str
    is_negative: str


class TenguAttachmentsMetadata(TenguEventMetadataBase):
    """Metadata for tengu_attachments (1 instance)."""

    attachment_types: str
    agentId: str | None = None  # Present when called from agent context
    agentType: str | None = None  # Present when called from agent context


class TenguMcpCliStatusMetadata(TenguEventMetadataBase):
    """Metadata for tengu_mcp_cli_status (1 instance)."""

    enabled: str
    legacy_env_var_set: str
    source: str


class TenguMcpServerConnectionSucceededMetadata(TenguEventMetadataBase):
    """Metadata for tengu_mcp_server_connection_succeeded (1 instance)."""

    pass  # Just base fields


class TenguMcpIdeServerConnectionSucceededMetadata(TenguEventMetadataBase):
    """Metadata for tengu_mcp_ide_server_connection_succeeded (1 instance)."""

    serverVersion: str


class TenguStartupManualModelConfigMetadata(TenguEventMetadataBase):
    """Metadata for tengu_startup_manual_model_config (1 instance)."""

    subscriptionType: str


class TenguNativeUpdateCompleteMetadata(TenguEventMetadataBase):
    """Metadata for tengu_native_update_complete (1 instance)."""

    latency_ms: str
    was_already_running: str
    was_force_reinstall: str
    was_new_install: str


class TenguClaudeaiLimitsStatusChangedMetadata(TenguEventMetadataBase):
    """Metadata for tengu_claudeai_limits_status_changed (1 instance)."""

    hoursTillReset: str
    status: str
    unifiedRateLimitFallbackAvailable: str
    agentId: str | None = None  # Present when called from agent context
    agentType: str | None = None  # Present when called from agent context


class TenguNotificationMethodUsedMetadata(TenguEventMetadataBase):
    """Metadata for tengu_notification_method_used (1 instance)."""

    configured_channel: str
    method_used: str
    term: str


class TenguFileOperationMetadata(TenguEventMetadataBase):
    """Metadata for tengu_file_operation (1 instance)."""

    contentHash: str
    filePathHash: str
    operation: str
    tool: str


# ==============================================================================
# Event Types - BIFURCATED by value/secondaryExposures presence
# ==============================================================================


def _discriminate_event_by_name(v: Any) -> str:
    """Discriminate events by eventName pattern."""
    if not isinstance(v, dict):
        return 'base'
    name = v.get('eventName', '')
    if name == 'statsig::config_exposure':
        return 'config_exposure'
    if name == 'statsig::gate_exposure':
        return 'gate_exposure'
    if name == 'statsig::diagnostics':
        return 'diagnostics'
    if name == 'tengu_prompt_suggestion_init':
        return 'tengu_prompt_suggestion_init'
    if name == 'tengu_mcp_servers':
        return 'tengu_mcp_servers'
    if name == 'tengu_api_query':
        return 'tengu_api_query'
    if name == 'tengu_sysprompt_block':
        return 'tengu_sysprompt_block'
    if name == 'tengu_api_after_normalize':
        return 'tengu_api_after_normalize'
    if name == 'tengu_api_before_normalize':
        return 'tengu_api_before_normalize'
    if name == 'tengu_api_cache_breakpoints':
        return 'tengu_api_cache_breakpoints'
    if name == 'tengu_api_success':
        return 'tengu_api_success'
    if name == 'tengu_dir_search':
        return 'tengu_dir_search'
    if name == 'tengu_agent_tool_completed':
        return 'tengu_agent_tool_completed'
    if name == 'tengu_agent_tool_selected':
        return 'tengu_agent_tool_selected'
    if name == 'tengu_version_check_success':
        return 'tengu_version_check_success'
    if name == 'tengu_mcp_tools_commands_loaded':
        return 'tengu_mcp_tools_commands_loaded'
    if name == 'tengu_shell_set_cwd':
        return 'tengu_shell_set_cwd'
    if name == 'tengu_repl_hook_finished':
        return 'tengu_repl_hook_finished'
    if name == 'tengu_timer':
        return 'tengu_timer'
    if name == 'tengu_startup_telemetry':
        return 'tengu_startup_telemetry'
    if name == 'tengu_exit':
        return 'tengu_exit'
    if name == 'tengu_trust_dialog_shown':
        return 'tengu_trust_dialog_shown'
    if name == 'tengu_ripgrep_availability':
        return 'tengu_ripgrep_availability'
    if name == 'tengu_init':
        return 'tengu_init'
    if name == 'tengu_run_hook':
        return 'tengu_run_hook'
    if name == 'tengu_version_lock_failed':
        return 'tengu_version_lock_failed'
    if name == 'tengu_native_auto_updater_start':
        return 'tengu_native_auto_updater_start'
    if name == 'tengu_native_auto_updater_success':
        return 'tengu_native_auto_updater_success'
    if name == 'tengu_config_stale_write':
        return 'tengu_config_stale_write'
    if name == 'tengu_context_size':
        return 'tengu_context_size'
    if name == 'tengu_fork_agent_query':
        return 'tengu_fork_agent_query'
    if name == 'tengu_thinking':
        return 'tengu_thinking'
    if name == 'tengu_paste_text':
        return 'tengu_paste_text'
    if name == 'tengu_input_prompt':
        return 'tengu_input_prompt'
    if name == 'tengu_attachments':
        return 'tengu_attachments'
    if name == 'tengu_mcp_cli_status':
        return 'tengu_mcp_cli_status'
    if name == 'tengu_mcp_server_connection_succeeded':
        return 'tengu_mcp_server_connection_succeeded'
    if name == 'tengu_mcp_ide_server_connection_succeeded':
        return 'tengu_mcp_ide_server_connection_succeeded'
    if name == 'tengu_startup_manual_model_config':
        return 'tengu_startup_manual_model_config'
    if name == 'tengu_native_update_complete':
        return 'tengu_native_update_complete'
    if name == 'tengu_claudeai_limits_status_changed':
        return 'tengu_claudeai_limits_status_changed'
    if name == 'tengu_notification_method_used':
        return 'tengu_notification_method_used'
    if name == 'tengu_file_operation':
        return 'tengu_file_operation'
    # Unknown event - use base
    return 'base'


# Statsig exposure events (have value=null + secondaryExposures)
class StatsigConfigExposureEvent(StrictModel):
    """Event for statsig::config_exposure."""

    eventName: Literal['statsig::config_exposure']
    metadata: ConfigExposureMetadata
    user: StatsigUser
    time: int
    value: None  # Always null in rgstr requests
    secondaryExposures: Sequence[Mapping[str, str]]


class StatsigGateExposureEvent(StrictModel):
    """Event for statsig::gate_exposure."""

    eventName: Literal['statsig::gate_exposure']
    metadata: GateExposureMetadata
    user: StatsigUser
    time: int
    value: None  # Always null in rgstr requests
    secondaryExposures: Sequence[Mapping[str, str]]


class StatsigDiagnosticsEvent(StrictModel):
    """Event for statsig::diagnostics (has value=null, no secondaryExposures)."""

    eventName: Literal['statsig::diagnostics']
    metadata: DiagnosticsMetadata
    user: StatsigUser
    time: int
    value: None  # Always null in rgstr requests


# Tengu events (no value/secondaryExposures)
class TenguPromptSuggestionInitEvent(StrictModel):
    """Event for tengu_prompt_suggestion_init."""

    eventName: Literal['tengu_prompt_suggestion_init']
    metadata: TenguPromptSuggestionInitMetadata
    user: StatsigUser
    time: int


class TenguMcpServersEvent(StrictModel):
    """Event for tengu_mcp_servers."""

    eventName: Literal['tengu_mcp_servers']
    metadata: TenguMcpServersMetadata
    user: StatsigUser
    time: int


class TenguApiQueryEvent(StrictModel):
    """Event for tengu_api_query."""

    eventName: Literal['tengu_api_query']
    metadata: TenguApiQueryMetadata
    user: StatsigUser
    time: int


class TenguSyspromptBlockEvent(StrictModel):
    """Event for tengu_sysprompt_block."""

    eventName: Literal['tengu_sysprompt_block']
    metadata: TenguSyspromptBlockMetadata
    user: StatsigUser
    time: int


class TenguApiAfterNormalizeEvent(StrictModel):
    """Event for tengu_api_after_normalize."""

    eventName: Literal['tengu_api_after_normalize']
    metadata: TenguApiAfterNormalizeMetadata
    user: StatsigUser
    time: int


class TenguApiBeforeNormalizeEvent(StrictModel):
    """Event for tengu_api_before_normalize."""

    eventName: Literal['tengu_api_before_normalize']
    metadata: TenguApiBeforeNormalizeMetadata
    user: StatsigUser
    time: int


class TenguApiCacheBreakpointsEvent(StrictModel):
    """Event for tengu_api_cache_breakpoints."""

    eventName: Literal['tengu_api_cache_breakpoints']
    metadata: TenguApiCacheBreakpointsMetadata
    user: StatsigUser
    time: int


class TenguApiSuccessEvent(StrictModel):
    """Event for tengu_api_success."""

    eventName: Literal['tengu_api_success']
    metadata: TenguApiSuccessMetadata
    user: StatsigUser
    time: int


class TenguDirSearchEvent(StrictModel):
    """Event for tengu_dir_search."""

    eventName: Literal['tengu_dir_search']
    metadata: TenguDirSearchMetadata
    user: StatsigUser
    time: int


class TenguAgentToolCompletedEvent(StrictModel):
    """Event for tengu_agent_tool_completed."""

    eventName: Literal['tengu_agent_tool_completed']
    metadata: TenguAgentToolCompletedMetadata
    user: StatsigUser
    time: int


class TenguAgentToolSelectedEvent(StrictModel):
    """Event for tengu_agent_tool_selected."""

    eventName: Literal['tengu_agent_tool_selected']
    metadata: TenguAgentToolSelectedMetadata
    user: StatsigUser
    time: int


class TenguVersionCheckSuccessEvent(StrictModel):
    """Event for tengu_version_check_success."""

    eventName: Literal['tengu_version_check_success']
    metadata: TenguVersionCheckSuccessMetadata
    user: StatsigUser
    time: int


class TenguMcpToolsCommandsLoadedEvent(StrictModel):
    """Event for tengu_mcp_tools_commands_loaded."""

    eventName: Literal['tengu_mcp_tools_commands_loaded']
    metadata: TenguMcpToolsCommandsLoadedMetadata
    user: StatsigUser
    time: int


class TenguShellSetCwdEvent(StrictModel):
    """Event for tengu_shell_set_cwd."""

    eventName: Literal['tengu_shell_set_cwd']
    metadata: TenguShellSetCwdMetadata
    user: StatsigUser
    time: int


class TenguReplHookFinishedEvent(StrictModel):
    """Event for tengu_repl_hook_finished."""

    eventName: Literal['tengu_repl_hook_finished']
    metadata: TenguReplHookFinishedMetadata
    user: StatsigUser
    time: int


class TenguTimerEvent(StrictModel):
    """Event for tengu_timer."""

    eventName: Literal['tengu_timer']
    metadata: TenguTimerMetadata
    user: StatsigUser
    time: int


class TenguStartupTelemetryEvent(StrictModel):
    """Event for tengu_startup_telemetry."""

    eventName: Literal['tengu_startup_telemetry']
    metadata: TenguStartupTelemetryMetadata
    user: StatsigUser
    time: int


class TenguExitEvent(StrictModel):
    """Event for tengu_exit."""

    eventName: Literal['tengu_exit']
    metadata: TenguExitMetadata
    user: StatsigUser
    time: int


class TenguTrustDialogShownEvent(StrictModel):
    """Event for tengu_trust_dialog_shown."""

    eventName: Literal['tengu_trust_dialog_shown']
    metadata: TenguTrustDialogShownMetadata
    user: StatsigUser
    time: int


class TenguRipgrepAvailabilityEvent(StrictModel):
    """Event for tengu_ripgrep_availability."""

    eventName: Literal['tengu_ripgrep_availability']
    metadata: TenguRipgrepAvailabilityMetadata
    user: StatsigUser
    time: int


class TenguInitEvent(StrictModel):
    """Event for tengu_init."""

    eventName: Literal['tengu_init']
    metadata: TenguInitMetadata
    user: StatsigUser
    time: int


class TenguRunHookEvent(StrictModel):
    """Event for tengu_run_hook."""

    eventName: Literal['tengu_run_hook']
    metadata: TenguRunHookMetadata
    user: StatsigUser
    time: int


class TenguVersionLockFailedEvent(StrictModel):
    """Event for tengu_version_lock_failed."""

    eventName: Literal['tengu_version_lock_failed']
    metadata: TenguVersionLockFailedMetadata
    user: StatsigUser
    time: int


class TenguNativeAutoUpdaterStartEvent(StrictModel):
    """Event for tengu_native_auto_updater_start."""

    eventName: Literal['tengu_native_auto_updater_start']
    metadata: TenguNativeAutoUpdaterStartMetadata
    user: StatsigUser
    time: int


class TenguNativeAutoUpdaterSuccessEvent(StrictModel):
    """Event for tengu_native_auto_updater_success."""

    eventName: Literal['tengu_native_auto_updater_success']
    metadata: TenguNativeAutoUpdaterSuccessMetadata
    user: StatsigUser
    time: int


class TenguConfigStaleWriteEvent(StrictModel):
    """Event for tengu_config_stale_write."""

    eventName: Literal['tengu_config_stale_write']
    metadata: TenguConfigStaleWriteMetadata
    user: StatsigUser
    time: int


class TenguContextSizeEvent(StrictModel):
    """Event for tengu_context_size."""

    eventName: Literal['tengu_context_size']
    metadata: TenguContextSizeMetadata
    user: StatsigUser
    time: int


class TenguForkAgentQueryEvent(StrictModel):
    """Event for tengu_fork_agent_query."""

    eventName: Literal['tengu_fork_agent_query']
    metadata: TenguForkAgentQueryMetadata
    user: StatsigUser
    time: int


class TenguThinkingEvent(StrictModel):
    """Event for tengu_thinking."""

    eventName: Literal['tengu_thinking']
    metadata: TenguThinkingMetadata
    user: StatsigUser
    time: int


class TenguPasteTextEvent(StrictModel):
    """Event for tengu_paste_text."""

    eventName: Literal['tengu_paste_text']
    metadata: TenguPasteTextMetadata
    user: StatsigUser
    time: int


class TenguInputPromptEvent(StrictModel):
    """Event for tengu_input_prompt."""

    eventName: Literal['tengu_input_prompt']
    metadata: TenguInputPromptMetadata
    user: StatsigUser
    time: int


class TenguAttachmentsEvent(StrictModel):
    """Event for tengu_attachments."""

    eventName: Literal['tengu_attachments']
    metadata: TenguAttachmentsMetadata
    user: StatsigUser
    time: int


class TenguMcpCliStatusEvent(StrictModel):
    """Event for tengu_mcp_cli_status."""

    eventName: Literal['tengu_mcp_cli_status']
    metadata: TenguMcpCliStatusMetadata
    user: StatsigUser
    time: int


class TenguMcpServerConnectionSucceededEvent(StrictModel):
    """Event for tengu_mcp_server_connection_succeeded."""

    eventName: Literal['tengu_mcp_server_connection_succeeded']
    metadata: TenguMcpServerConnectionSucceededMetadata
    user: StatsigUser
    time: int


class TenguMcpIdeServerConnectionSucceededEvent(StrictModel):
    """Event for tengu_mcp_ide_server_connection_succeeded."""

    eventName: Literal['tengu_mcp_ide_server_connection_succeeded']
    metadata: TenguMcpIdeServerConnectionSucceededMetadata
    user: StatsigUser
    time: int


class TenguStartupManualModelConfigEvent(StrictModel):
    """Event for tengu_startup_manual_model_config."""

    eventName: Literal['tengu_startup_manual_model_config']
    metadata: TenguStartupManualModelConfigMetadata
    user: StatsigUser
    time: int


class TenguNativeUpdateCompleteEvent(StrictModel):
    """Event for tengu_native_update_complete."""

    eventName: Literal['tengu_native_update_complete']
    metadata: TenguNativeUpdateCompleteMetadata
    user: StatsigUser
    time: int


class TenguClaudeaiLimitsStatusChangedEvent(StrictModel):
    """Event for tengu_claudeai_limits_status_changed."""

    eventName: Literal['tengu_claudeai_limits_status_changed']
    metadata: TenguClaudeaiLimitsStatusChangedMetadata
    user: StatsigUser
    time: int


class TenguNotificationMethodUsedEvent(StrictModel):
    """Event for tengu_notification_method_used."""

    eventName: Literal['tengu_notification_method_used']
    metadata: TenguNotificationMethodUsedMetadata
    user: StatsigUser
    time: int


class TenguFileOperationEvent(StrictModel):
    """Event for tengu_file_operation."""

    eventName: Literal['tengu_file_operation']
    metadata: TenguFileOperationMetadata
    user: StatsigUser
    time: int


# Fallback for unknown events - keeps loose typing only for truly unknown events
class UnknownStatsigEventMetadata(StrictModel):
    """
    Metadata for unknown/future event types.

    This is the escape hatch for events not yet modeled.
    All fields are optional to accommodate any structure.
    """

    model_config = pydantic.ConfigDict(extra='allow')  # Allow unknown fields


class UnknownStatsigEvent(StrictModel):
    """Fallback event type for unknown eventNames."""

    eventName: str
    metadata: UnknownStatsigEventMetadata
    user: StatsigUser
    time: int
    value: str | None = None
    secondaryExposures: Sequence[Mapping[str, str]] | None = None


# Full discriminated union of all event types
StatsigEvent = Annotated[
    Annotated[StatsigConfigExposureEvent, pydantic.Tag('config_exposure')]
    | Annotated[StatsigGateExposureEvent, pydantic.Tag('gate_exposure')]
    | Annotated[StatsigDiagnosticsEvent, pydantic.Tag('diagnostics')]
    | Annotated[TenguPromptSuggestionInitEvent, pydantic.Tag('tengu_prompt_suggestion_init')]
    | Annotated[TenguMcpServersEvent, pydantic.Tag('tengu_mcp_servers')]
    | Annotated[TenguApiQueryEvent, pydantic.Tag('tengu_api_query')]
    | Annotated[TenguSyspromptBlockEvent, pydantic.Tag('tengu_sysprompt_block')]
    | Annotated[TenguApiAfterNormalizeEvent, pydantic.Tag('tengu_api_after_normalize')]
    | Annotated[TenguApiBeforeNormalizeEvent, pydantic.Tag('tengu_api_before_normalize')]
    | Annotated[TenguApiCacheBreakpointsEvent, pydantic.Tag('tengu_api_cache_breakpoints')]
    | Annotated[TenguApiSuccessEvent, pydantic.Tag('tengu_api_success')]
    | Annotated[TenguDirSearchEvent, pydantic.Tag('tengu_dir_search')]
    | Annotated[TenguAgentToolCompletedEvent, pydantic.Tag('tengu_agent_tool_completed')]
    | Annotated[TenguAgentToolSelectedEvent, pydantic.Tag('tengu_agent_tool_selected')]
    | Annotated[TenguVersionCheckSuccessEvent, pydantic.Tag('tengu_version_check_success')]
    | Annotated[TenguMcpToolsCommandsLoadedEvent, pydantic.Tag('tengu_mcp_tools_commands_loaded')]
    | Annotated[TenguShellSetCwdEvent, pydantic.Tag('tengu_shell_set_cwd')]
    | Annotated[TenguReplHookFinishedEvent, pydantic.Tag('tengu_repl_hook_finished')]
    | Annotated[TenguTimerEvent, pydantic.Tag('tengu_timer')]
    | Annotated[TenguStartupTelemetryEvent, pydantic.Tag('tengu_startup_telemetry')]
    | Annotated[TenguExitEvent, pydantic.Tag('tengu_exit')]
    | Annotated[TenguTrustDialogShownEvent, pydantic.Tag('tengu_trust_dialog_shown')]
    | Annotated[TenguRipgrepAvailabilityEvent, pydantic.Tag('tengu_ripgrep_availability')]
    | Annotated[TenguInitEvent, pydantic.Tag('tengu_init')]
    | Annotated[TenguRunHookEvent, pydantic.Tag('tengu_run_hook')]
    | Annotated[TenguVersionLockFailedEvent, pydantic.Tag('tengu_version_lock_failed')]
    | Annotated[TenguNativeAutoUpdaterStartEvent, pydantic.Tag('tengu_native_auto_updater_start')]
    | Annotated[TenguNativeAutoUpdaterSuccessEvent, pydantic.Tag('tengu_native_auto_updater_success')]
    | Annotated[TenguConfigStaleWriteEvent, pydantic.Tag('tengu_config_stale_write')]
    | Annotated[TenguContextSizeEvent, pydantic.Tag('tengu_context_size')]
    | Annotated[TenguForkAgentQueryEvent, pydantic.Tag('tengu_fork_agent_query')]
    | Annotated[TenguThinkingEvent, pydantic.Tag('tengu_thinking')]
    | Annotated[TenguPasteTextEvent, pydantic.Tag('tengu_paste_text')]
    | Annotated[TenguInputPromptEvent, pydantic.Tag('tengu_input_prompt')]
    | Annotated[TenguAttachmentsEvent, pydantic.Tag('tengu_attachments')]
    | Annotated[TenguMcpCliStatusEvent, pydantic.Tag('tengu_mcp_cli_status')]
    | Annotated[TenguMcpServerConnectionSucceededEvent, pydantic.Tag('tengu_mcp_server_connection_succeeded')]
    | Annotated[TenguMcpIdeServerConnectionSucceededEvent, pydantic.Tag('tengu_mcp_ide_server_connection_succeeded')]
    | Annotated[TenguStartupManualModelConfigEvent, pydantic.Tag('tengu_startup_manual_model_config')]
    | Annotated[TenguNativeUpdateCompleteEvent, pydantic.Tag('tengu_native_update_complete')]
    | Annotated[TenguClaudeaiLimitsStatusChangedEvent, pydantic.Tag('tengu_claudeai_limits_status_changed')]
    | Annotated[TenguNotificationMethodUsedEvent, pydantic.Tag('tengu_notification_method_used')]
    | Annotated[TenguFileOperationEvent, pydantic.Tag('tengu_file_operation')]
    | Annotated[UnknownStatsigEvent, pydantic.Tag('base')],
    pydantic.Discriminator(_discriminate_event_by_name),
]


# ==============================================================================
# Register Endpoint (/v1/rgstr)
# ==============================================================================


class StatsigRegisterRequest(StrictModel):
    """Request to /v1/rgstr (register events)."""

    events: Sequence[StatsigEvent]
    statsigMetadata: StatsigMetadata


class StatsigRegisterResponse(StrictModel):
    """Response from /v1/rgstr."""

    success: bool
