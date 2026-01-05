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

from src.schemas.cc_internal_api.base import EmptyBody, PermissiveModel

# ==============================================================================
# Common Types
# ==============================================================================


class StatsigCustomIDs(PermissiveModel):
    """
    Custom identifiers for Statsig user tracking.

    VALIDATION STATUS: VALIDATED
    """

    sessionId: str  # Session UUID
    organizationUUID: str  # Organization UUID
    accountUUID: str  # Account UUID


class StatsigCustom(PermissiveModel):
    """
    Custom user attributes for feature targeting.

    VALIDATION STATUS: VALIDATED
    """

    userType: Literal['external', 'internal']
    organizationUuid: str
    accountUuid: str
    subscriptionType: Literal['free', 'pro', 'team', 'max', 'enterprise']
    firstTokenTime: int  # Timestamp of first token


class StatsigEnvironment(PermissiveModel):
    """
    Statsig environment configuration.

    VALIDATION STATUS: VALIDATED
    """

    tier: Literal['production', 'staging', 'development']


class StatsigUser(PermissiveModel):
    """
    User object for Statsig requests.

    VALIDATION STATUS: VALIDATED
    """

    userID: str  # Hashed device ID
    appVersion: str  # Claude Code version
    customIDs: StatsigCustomIDs
    custom: StatsigCustom
    statsigEnvironment: StatsigEnvironment


class StatsigMetadata(PermissiveModel):
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


class StatsigInitializeRequest(PermissiveModel):
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
    previousDerivedFields: Mapping[str, Any]  # {} on first request, populated on subsequent


class StatsigInitializeFullBody(PermissiveModel):
    """
    Body structure for 200 OK response with feature flags.

    VALIDATION STATUS: VALIDATED
    Returned on initial request or when updates are available.
    The intercept script extracts this from body.data (json type wrapper).

    Note: Many additional fields (generator, time, derived_fields, evaluated_keys,
    company_lcut, hash_used, etc.) are captured by PermissiveModel.extra='allow'.
    """

    feature_gates: Mapping[str, Any]
    dynamic_configs: Mapping[str, Any]
    layer_configs: Mapping[str, Any]
    has_updates: bool


# Union type for discriminated dispatch in captures
# EmptyBody = 204 No Content (no updates), FullBody = 200 OK (has updates)
StatsigInitializeResponse = EmptyBody | StatsigInitializeFullBody


# ==============================================================================
# Register/Log Endpoint (/v1/rgstr)
# ==============================================================================


class StatsigEventMetadata(PermissiveModel):
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


class StatsigEvent(PermissiveModel):
    """
    Single event in Statsig rgstr request.

    VALIDATION STATUS: VALIDATED
    """

    eventName: str  # e.g., "tengu_run_hook"
    metadata: StatsigEventMetadata
    user: StatsigUser
    time: int  # Unix timestamp in milliseconds


class StatsigRegisterRequest(PermissiveModel):
    """
    Request to /v1/rgstr (register events).

    VALIDATION STATUS: VALIDATED
    Logs feature flag usage and events.
    """

    events: Sequence[StatsigEvent]
    statsigMetadata: StatsigMetadata


class StatsigRegisterResponse(PermissiveModel):
    """
    Response from /v1/rgstr.

    VALIDATION STATUS: VALIDATED
    Returns success acknowledgment.
    """

    success: bool  # Required - no default (fail-fast)
