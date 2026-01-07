"""
Statsig schemas for Claude Code feature flag system.

Claude Code uses Statsig (statsig.anthropic.com) for A/B testing and feature flags.
This is separate from the Anthropic-internal `/api/eval/sdk-*` feature flags.

Endpoints:
- /v1/initialize - Initialize feature flag state
- /v1/rgstr - Register/log feature flag events
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import pydantic

from src.schemas.cc_internal_api.base import EmptyBody, EmptyDict, StrictModel

# ==============================================================================
# Common Types
# ==============================================================================


class StatsigCustomIDs(StrictModel):
    """
    Custom identifiers for Statsig user tracking.

    VALIDATION STATUS: VALIDATED
    """

    sessionId: str  # Session UUID
    organizationUUID: str  # Organization UUID
    accountUUID: str  # Account UUID


class StatsigCustom(StrictModel):
    """
    Custom user attributes for feature targeting.

    VALIDATION STATUS: VALIDATED
    """

    userType: Literal['external', 'internal']
    organizationUuid: str
    accountUuid: str
    subscriptionType: Literal['free', 'pro', 'team', 'max', 'enterprise']
    firstTokenTime: int  # Timestamp of first token


class StatsigEnvironment(StrictModel):
    """
    Statsig environment configuration.

    VALIDATION STATUS: VALIDATED
    """

    tier: Literal['production', 'staging', 'development']


class StatsigUser(StrictModel):
    """
    User object for Statsig requests.

    VALIDATION STATUS: VALIDATED
    """

    userID: str  # Hashed device ID
    appVersion: str  # Claude Code version
    customIDs: StatsigCustomIDs
    custom: StatsigCustom
    statsigEnvironment: StatsigEnvironment


class StatsigMetadata(StrictModel):
    """
    Statsig SDK metadata.

    VALIDATION STATUS: VALIDATED
    """

    sdkType: str  # e.g., "js-client"
    sdkVersion: str  # e.g., "5.6.0"
    stableID: str  # Stable identifier
    sessionID: str  # SDK session ID
    fallbackUrl: str | None = None  # Fallback URL for SDK


# ==============================================================================
# Initialize Endpoint (/v1/initialize)
# ==============================================================================


class StatsigEvaluatedKeys(StrictModel):
    """
    Evaluated user keys in Statsig initialize response.

    VALIDATION STATUS: VALIDATED
    Observed structure is consistent across all captures.
    """

    userID: str  # Hashed device ID
    stableID: str  # Stable identifier
    customIDs: StatsigCustomIDs  # Reuses existing model


class AutoCaptureDisabledEvents(StrictModel):
    """
    Disabled events configuration.

    VALIDATION STATUS: VALIDATED
    Always {} in observed captures.
    """

    # Note: This is always empty in captures. If events appear,
    # we'll need to update the schema.
    pass


class AutoCaptureSettings(StrictModel):
    """
    Auto-capture settings for session recording.

    VALIDATION STATUS: VALIDATED
    Observed structure is consistent across all captures.
    """

    disabled_events: AutoCaptureDisabledEvents


class DerivedFields(StrictModel):
    """
    Derived fields from Statsig server, echoed back in subsequent requests.

    VALIDATION STATUS: VALIDATED
    Observed in response.derived_fields and request.previousDerivedFields.
    """

    ip: str
    country: str
    appVersion: str
    app_version: str  # Duplicate of appVersion (snake_case)
    browserName: str
    browserVersion: str
    osName: str
    osVersion: str
    browser_name: str  # Duplicate of browserName (snake_case)
    browser_version: str  # Duplicate of browserVersion (snake_case)
    os_name: str  # Duplicate of osName (snake_case)
    os_version: str  # Duplicate of osVersion (snake_case)


class StatsigInitializeRequest(StrictModel):
    """
    Request to /v1/initialize.

    VALIDATION STATUS: VALIDATED
    Initializes feature flag state for a user.
    All fields are always present in observed requests.
    """

    user: StatsigUser
    statsigMetadata: StatsigMetadata
    sinceTime: int  # 0 on first request, timestamp on subsequent
    hash: str  # Hash algorithm, e.g., "djb2"
    deltasResponseRequested: bool  # Whether to request delta updates
    full_checksum: str  # Checksum for delta comparison
    # Empty {} on first request, populated DerivedFields on subsequent
    previousDerivedFields: DerivedFields | EmptyDict


class StatsigFeatureGate(StrictModel):
    """
    Feature gate value in Statsig initialize response.

    VALIDATION STATUS: VALIDATED
    """

    name: str  # Hash identifier
    value: bool  # Gate on/off
    rule_id: str  # e.g., "default"
    id_type: str  # e.g., "userID"
    secondary_exposures: Sequence[Mapping[str, str]]


class StatsigBaseDynamicConfig(StrictModel):
    """
    Base fields for all Statsig dynamic configs.

    VALIDATION STATUS: VALIDATED
    Always present in all 47 dynamic configs observed.
    """

    name: str  # Hash identifier
    # GENUINELY POLYMORPHIC: Dynamic config payloads vary by config type.
    # Each config has different schema (e.g., experiment params, targeting rules).
    # Cannot be typed more strictly without per-config schema enumeration.
    value: Mapping[str, Any]  # noqa: loose-typing
    rule_id: str  # e.g., "default"
    group: str  # e.g., "default"
    is_device_based: bool
    id_type: str  # e.g., "userID"
    secondary_exposures: Sequence[Mapping[str, str]]


class StatsigEvaluatedConfig(StatsigBaseDynamicConfig):
    """
    Dynamic config that was directly evaluated (has `passed` field).

    VALIDATION STATUS: VALIDATED
    18/47 configs in captures have this pattern.
    """

    passed: bool


class StatsigExperimentConfig(StatsigBaseDynamicConfig):
    """
    Experiment-type dynamic config (has experiment tracking fields).

    VALIDATION STATUS: VALIDATED
    18/47 configs in captures have this pattern.
    """

    is_user_in_experiment: bool
    is_experiment_active: bool


class StatsigNamedExperimentConfig(StatsigExperimentConfig):
    """
    Named experiment config (experiment + group_name).

    VALIDATION STATUS: VALIDATED
    11/47 configs in captures have this pattern.
    """

    group_name: str


# Union of all dynamic config types - Pydantic tries most specific first
StatsigDynamicConfig = StatsigNamedExperimentConfig | StatsigExperimentConfig | StatsigEvaluatedConfig


class StatsigInitializeFullBody(StrictModel):
    """
    Body structure for 200 OK response with feature flags.

    VALIDATION STATUS: VALIDATED
    Returned on initial request or when updates are available.
    The intercept script extracts this from body.data (json type wrapper).
    """

    # Core feature flag data
    feature_gates: Mapping[str, StatsigFeatureGate]
    dynamic_configs: Mapping[str, StatsigDynamicConfig]
    layer_configs: EmptyDict  # Always {} in observed captures
    has_updates: bool

    # Metadata fields
    generator: str  # e.g., "statsig-go-sdk"
    time: int  # Unix timestamp
    company_lcut: int  # Last config update timestamp
    hash_used: str  # Hash algorithm, e.g., "djb2"
    hashed_sdk_key_used: str | None = None  # Hashed SDK key

    # SDK parameters - always {} in observed captures
    sdkParams: EmptyDict
    evaluated_keys: StatsigEvaluatedKeys

    # Derived fields (user context detected by server)
    derived_fields: DerivedFields

    # Session recording settings
    can_record_session: bool
    recording_blocked: bool
    session_recording_rate: float  # 0.0 to 1.0
    auto_capture_settings: AutoCaptureSettings

    # Target app
    target_app_used: str | None = None

    # Checksum for delta updates
    full_checksum: str

    # Delta update fields (present on subsequent requests)
    deleted_configs: Sequence[str] | None = None
    deleted_gates: Sequence[str] | None = None
    deleted_layers: Sequence[str] | None = None
    is_delta: bool | None = None
    checksum: str | None = None
    checksumV2: str | None = None
    deltas_full_response: bool | None = None


# Union type for discriminated dispatch in captures
# EmptyBody = 204 No Content (no updates), FullBody = 200 OK (has updates)
StatsigInitializeResponse = EmptyBody | StatsigInitializeFullBody


# ==============================================================================
# Register/Log Endpoint (/v1/rgstr)
# ==============================================================================


# ==============================================================================
# Statsig Diagnostics Metadata Types
# ==============================================================================


class StatsigMarkerEvaluationDetails(StrictModel):
    """
    Evaluation details in a diagnostics marker.

    VALIDATION STATUS: VALIDATED
    Observed in markers with action="end" for overall key.
    """

    reason: str  # e.g., "Network"
    lcut: int  # Last config update timestamp
    receivedAt: int  # When response was received


class StatsigMarkerErrorWithCode(StrictModel):
    """
    Error details with error code in a diagnostics marker.

    VALIDATION STATUS: VALIDATED
    Observed in network/timeout errors.
    """

    name: str  # e.g., "TimeoutError"
    message: str  # Error description
    code: int  # Error code (e.g., 23 for timeout)


class StatsigMarkerErrorWithoutCode(StrictModel):
    """
    Error details without error code in a diagnostics marker.

    VALIDATION STATUS: VALIDATED
    Observed in high-level errors like InitializeError.
    """

    name: str  # e.g., "InitializeError"
    message: str  # Error description


# Union - Pydantic tries more specific (with code) first
StatsigMarkerError = StatsigMarkerErrorWithCode | StatsigMarkerErrorWithoutCode


class StatsigMarker(StrictModel):
    """
    Timing marker in Statsig diagnostics events.

    VALIDATION STATUS: VALIDATED
    Observed in statsig::diagnostics event metadata.markers.
    """

    key: str  # e.g., "overall", "initialize"
    action: str  # e.g., "start", "end"
    timestamp: int  # Unix timestamp in milliseconds
    # Optional fields vary by marker type
    step: str | None = None  # e.g., "network_request", "process"
    attempt: int | None = None
    success: bool | None = None
    statusCode: int | None = None
    sdkRegion: str | None = None
    isDelta: bool | None = None
    error: StatsigMarkerError | None = None
    evaluationDetails: StatsigMarkerEvaluationDetails | None = None


class StatsigOptionsNetworkConfig(StrictModel):
    """
    Network config in Statsig options.

    VALIDATION STATUS: VALIDATED
    """

    api: str  # e.g., "https://statsig.anthropic.com/v1/"


class StatsigOptionsEnvironment(StrictModel):
    """
    Environment in Statsig options.

    VALIDATION STATUS: VALIDATED
    """

    tier: str  # e.g., "production"


class StatsigOptionsStorageProviderCache(StrictModel):
    """
    Storage provider cache in Statsig options.

    VALIDATION STATUS: VALIDATED
    Always {} in observed captures.
    """

    pass


class StatsigOptionsStorageProvider(StrictModel):
    """
    Storage provider in Statsig options.

    VALIDATION STATUS: VALIDATED
    """

    cache: StatsigOptionsStorageProviderCache
    ready: bool


class StatsigOptions(StrictModel):
    """
    Statsig SDK options in diagnostics events.

    VALIDATION STATUS: VALIDATED
    Observed in statsig::diagnostics event metadata.statsigOptions.
    """

    networkConfig: StatsigOptionsNetworkConfig
    environment: StatsigOptionsEnvironment
    includeCurrentPageUrlWithEvents: bool
    logLevel: int
    storageProvider: StatsigOptionsStorageProvider


class StatsigEventMetadata(StrictModel):
    """
    Metadata for a Statsig event.

    VALIDATION STATUS: VALIDATED
    Contains highly polymorphic event-specific data. All fields optional since
    different event types include different subsets of fields.
    """

    # Exposure/gate event fields
    config: str | None = None
    gate: str | None = None
    gateValue: str | None = None
    lcut: str | None = None
    reason: str | None = None
    receivedAt: str | None = None
    ruleID: str | None = None
    rulePassed: str | None = None

    # Session/environment fields
    betas: str | None = None
    clientType: str | None = None
    entrypoint: str | None = None
    env: str | None = None
    hookName: str | None = None
    isInteractive: str | None = None
    isNonInteractiveSession: str | None = None
    isTTY: str | None = None
    model: str | None = None
    sessionId: str | None = None
    userType: str | None = None

    # SWE bench fields
    sweBenchInstanceId: str | None = None
    sweBenchRunId: str | None = None
    sweBenchTaskId: str | None = None

    # Agent fields
    agentId: str | None = None
    agentType: str | None = None
    agent_type: str | None = None
    is_built_in_agent: str | None = None
    is_keep_going: str | None = None

    # Tool/command fields
    commands_count: str | None = None
    commands_metadata_length: str | None = None
    numAllowedTools: str | None = None
    numCommands: str | None = None
    numDisallowedTools: str | None = None
    tool: str | None = None
    tools_count: str | None = None

    # Token/cost metrics
    cacheCreationEphemeral1hTokens: str | None = None
    cacheCreationEphemeral5mTokens: str | None = None
    cacheCreationInputTokens: str | None = None
    cacheHitRate: str | None = None
    cacheReadInputTokens: str | None = None
    cachedInputTokens: str | None = None
    cachingEnabled: str | None = None
    costUSD: str | None = None
    inputTokens: str | None = None
    messageTokens: str | None = None
    outputTokens: str | None = None
    tokenCount: str | None = None
    total_tokens: str | None = None
    uncachedInputTokens: str | None = None

    # Context/message metrics
    assistant_message_count: str | None = None
    claude_md_size: str | None = None
    git_status_size: str | None = None
    messageCount: str | None = None
    messagesLength: str | None = None
    postNormalizedMessageCount: str | None = None
    preNormalizedMessageCount: str | None = None
    prompt_char_count: str | None = None
    response_char_count: str | None = None
    totalMessageCount: str | None = None
    total_context_size: str | None = None
    total_tool_uses: str | None = None

    # Duration/timing fields
    durationMs: str | None = None
    durationMsIncludingRetries: str | None = None
    duration_ms: str | None = None
    latency_ms: str | None = None
    ttftMs: str | None = None

    # MCP fields
    mcpClientCount: str | None = None
    mcp_servers_count: str | None = None
    mcp_tools_count: str | None = None
    mcp_tools_tokens: str | None = None
    non_mcp_tools_count: str | None = None
    non_mcp_tools_tokens: str | None = None

    # File discovery fields
    managedFilesFound: str | None = None
    projectDirsSearched: str | None = None
    projectFilesFound: str | None = None
    project_file_count_rounded: str | None = None
    userFilesFound: str | None = None

    # Config source fields
    claudeai: str | None = None
    enterprise: str | None = None
    project: str | None = None
    user: str | None = None
    plugin: str | None = None

    # Permission/sandbox fields
    allowDangerouslySkipPermissionsPassed: str | None = None
    are_unsandboxed_commands_allowed: str | None = None
    dangerouslySkipPermissionsPassed: str | None = None
    is_auto_bash_allowed_if_sandbox_enabled: str | None = None
    modeIsBypass: str | None = None
    permissionMode: str | None = None
    sandbox_enabled: str | None = None

    # Session history fields
    last_session_api_duration: str | None = None
    last_session_cost: str | None = None
    last_session_duration: str | None = None
    last_session_id: str | None = None
    last_session_lines_added: str | None = None
    last_session_lines_removed: str | None = None
    last_session_tool_duration: str | None = None
    last_session_total_cache_creation_input_tokens: str | None = None
    last_session_total_cache_read_input_tokens: str | None = None
    last_session_total_input_tokens: str | None = None
    last_session_total_output_tokens: str | None = None

    # Build/version fields
    auto_updater_disabled: str | None = None
    buildAgeMins: str | None = None
    configured_channel: str | None = None
    forkLabel: str | None = None
    serverVersion: str | None = None
    source_gcs: str | None = None
    was_already_running: str | None = None
    was_force_reinstall: str | None = None
    was_new_install: str | None = None

    # Request/query fields
    attempt: str | None = None
    provider: str | None = None
    queryChainId: str | None = None
    queryDepth: str | None = None
    querySource: str | None = None
    requestId: str | None = None
    serviceTier: str | None = None
    temperature: str | None = None

    # Status/result fields
    didFallBackToNonStreaming: str | None = None
    enabled: str | None = None
    event: str | None = None
    is_negative: str | None = None
    numBlocking: str | None = None
    numCancelled: str | None = None
    numNonBlockingError: str | None = None
    numSuccess: str | None = None
    operation: str | None = None
    status: str | None = None
    stop_reason: str | None = None
    success: str | None = None

    # File operation fields
    contentHash: str | None = None
    filePathHash: str | None = None
    hash: str | None = None
    length: str | None = None
    read_mtime: str | None = None
    read_size: str | None = None
    snippet: str | None = None
    write_mtime: str | None = None
    write_size: str | None = None

    # CLI/input fields
    attachment_types: str | None = None
    copyVariant: str | None = None
    debug: str | None = None
    debugToStderr: str | None = None
    hasInitialPrompt: str | None = None
    hasStdin: str | None = None
    inputFormat: str | None = None
    outputFormat: str | None = None
    pastedTextCount: str | None = None
    print: str | None = None
    verbose: str | None = None

    # Workspace fields
    context: str | None = None
    folderType: str | None = None
    hasApiKeyHelper: str | None = None
    hasAwsCommands: str | None = None
    hasBashExecution: str | None = None
    hasHooks: str | None = None
    hasMcpServers: str | None = None
    hasOtelHeadersHelper: str | None = None
    isHomeDir: str | None = None
    is_git: str | None = None
    is_lifetime_lock: str | None = None
    is_pid_based: str | None = None
    # Diagnostics event fields (statsig::diagnostics events)
    markers: Sequence[StatsigMarker] | None = None
    method_used: str | None = None
    rh: str | None = None
    source: str | None = None
    statsigOptions: StatsigOptions | None = None
    subdir: str | None = None
    subscriptionType: str | None = None
    term: str | None = None
    using_system: str | None = None
    working: str | None = None
    worktree: str | None = None
    worktree_count: str | None = None

    # Rate limit fields
    hoursTillReset: str | None = None
    legacy_env_var_set: str | None = None
    unifiedRateLimitFallbackAvailable: str | None = None

    # Python keyword fields (use alias)
    global_: str | None = pydantic.Field(default=None, alias='global')


class StatsigEvent(StrictModel):
    """
    Single event in Statsig rgstr request.

    VALIDATION STATUS: VALIDATED
    """

    eventName: str  # e.g., "tengu_run_hook"
    metadata: StatsigEventMetadata
    user: StatsigUser
    time: int  # Unix timestamp in milliseconds
    # Optional fields for exposure events
    value: str | None = None
    secondaryExposures: Sequence[Mapping[str, str]] | None = None


class StatsigRegisterRequest(StrictModel):
    """
    Request to /v1/rgstr (register events).

    VALIDATION STATUS: VALIDATED
    Logs feature flag usage and events.
    """

    events: Sequence[StatsigEvent]
    statsigMetadata: StatsigMetadata


class StatsigRegisterResponse(StrictModel):
    """
    Response from /v1/rgstr.

    VALIDATION STATUS: VALIDATED
    Returns success acknowledgment.
    """

    success: bool  # Required - no default (fail-fast)
