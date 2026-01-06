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


# ==============================================================================
# Initialize Endpoint (/v1/initialize)
# ==============================================================================


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
    value: Mapping[str, Any]
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

    Note: With strict validation (extra='forbid'), any additional fields from the API
    (generator, time, derived_fields, evaluated_keys, company_lcut, hash_used, etc.)
    will cause validation failure - these must be added if observed.
    """

    feature_gates: Mapping[str, StatsigFeatureGate]
    dynamic_configs: Mapping[str, StatsigDynamicConfig]
    # ALWAYS EMPTY in observed captures - strict typing for fail-fast validation
    layer_configs: EmptyDict  # Always {} - will fail if API starts sending data
    has_updates: bool


# Union type for discriminated dispatch in captures
# EmptyBody = 204 No Content (no updates), FullBody = 200 OK (has updates)
StatsigInitializeResponse = EmptyBody | StatsigInitializeFullBody


# ==============================================================================
# Register/Log Endpoint (/v1/rgstr)
# ==============================================================================


class StatsigEventMetadata(StrictModel):
    """
    Metadata for a Statsig event.

    VALIDATION STATUS: VALIDATED
    Contains event-specific data including environment info.
    """

    hookName: str | None = None  # e.g., "SessionEnd:prompt_input_exit"
    numCommands: str | None = None
    model: str | None = None
    sessionId: str | None = None
    userType: str | None = None
    betas: str | None = None  # Comma-separated beta features
    env: str | None = None  # JSON-encoded environment info
    entrypoint: str | None = None
    isInteractive: str | None = None  # "true" or "false" as string
    clientType: str | None = None
    sweBenchRunId: str | None = None
    sweBenchInstanceId: str | None = None
    sweBenchTaskId: str | None = None


class StatsigEvent(StrictModel):
    """
    Single event in Statsig rgstr request.

    VALIDATION STATUS: VALIDATED
    """

    eventName: str  # e.g., "tengu_run_hook"
    metadata: StatsigEventMetadata
    user: StatsigUser
    time: int  # Unix timestamp in milliseconds


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
