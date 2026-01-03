"""
Statsig schemas for Claude Code feature flag system.

Claude Code uses Statsig (statsig.anthropic.com) for A/B testing and feature flags.
This is separate from the Anthropic-internal `/api/eval/sdk-*` feature flags.

Endpoints:
- /v1/initialize - Initialize feature flag state
- /v1/rgstr - Register/log feature flag events
"""

from __future__ import annotations

from typing import Any, Literal

from src.schemas.cc_internal_api.base import PermissiveModel

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
    """

    user: StatsigUser
    statsigMetadata: StatsigMetadata
    sinceTime: int | None = None  # Timestamp for delta updates
    hash: str | None = None  # Hash algorithm (e.g., "djb2")
    deltasResponseRequested: bool = False
    full_checksum: str | None = None
    previousDerivedFields: dict[str, Any] | None = None


class StatsigInitializeResponse(PermissiveModel):
    """
    Response from /v1/initialize.

    VALIDATION STATUS: VALIDATED
    Note: Response is often empty when deltasResponseRequested=true.
    """

    empty: bool = False
    size: int = 0
    # Full response (when not empty) would include feature_gates, dynamic_configs, etc.
    feature_gates: dict[str, Any] | None = None
    dynamic_configs: dict[str, Any] | None = None
    layer_configs: dict[str, Any] | None = None
    has_updates: bool | None = None


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

    events: list[StatsigEvent]
    statsigMetadata: StatsigMetadata


class StatsigRegisterResponse(PermissiveModel):
    """
    Response from /v1/rgstr.

    VALIDATION STATUS: INFERRED
    Typically returns success acknowledgment.
    """

    success: bool = True
