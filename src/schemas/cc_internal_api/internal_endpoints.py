"""
Claude Code internal endpoint schemas.

These model Claude Code's internal API endpoints beyond the Messages API:
- /v1/messages/count_tokens - Token counting
- /api/claude_code/grove - Feature gating by region/tier
- /api/claude_code/organizations/metrics_enabled - Telemetry opt-in
- /api/oauth/account/settings - User preferences
- /api/oauth/claude_cli/client_data - Client configuration
- /api/hello - Health check

Note: Feature flag schemas (/api/eval/sdk-*) moved to feature_flags.py
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import pydantic

from src.schemas.cc_internal_api.base import EmptyDict, StrictModel
from src.schemas.cc_internal_api.request import RequestMessage, SystemBlock, ToolDefinition
from src.schemas.types import ModelId

# ==============================================================================
# Count Tokens (/v1/messages/count_tokens)
# ==============================================================================


class CountTokensRequest(StrictModel):
    """
    Request to /v1/messages/count_tokens.

    VALIDATION STATUS: VALIDATED
    Observed in request body.

    Used by Claude Code to measure token overhead for each tool definition.
    """

    model: ModelId
    messages: Sequence[RequestMessage]
    tools: Sequence[ToolDefinition] | None = None
    system: Sequence[SystemBlock] | None = None


class CountTokensResponse(StrictModel):
    """
    Response from /v1/messages/count_tokens.

    VALIDATION STATUS: VALIDATED
    Observed: {"input_tokens": 14363}
    """

    input_tokens: int


# ==============================================================================
# Grove (/api/claude_code/grove, /api/claude_code_grove)
# ==============================================================================


class GroveResponse(StrictModel):
    """
    Response from /api/claude_code/grove or /api/claude_code_grove.

    VALIDATION STATUS: VALIDATED
    Feature gating by region/subscription tier.

    Grove appears to control access to Claude Code features based on
    geographic region or subscription status, with grace periods for
    transitions.
    """

    grove_enabled: bool
    domain_excluded: bool  # Whether user's domain is excluded
    notice_is_grace_period: bool  # Whether in grace period
    notice_reminder_frequency: int  # Reminder frequency (0 = no reminders)


# ==============================================================================
# Metrics (/api/claude_code/metrics, /api/claude_code/organizations/metrics_enabled)
# ==============================================================================


class ResourceAttributes(StrictModel):
    """
    Resource attributes in OpenTelemetry metrics.

    VALIDATION STATUS: VALIDATED
    Service identification and environment metadata.

    JSON keys use dots per OpenTelemetry convention (e.g., "service.name").
    Python field names use underscores with Field(alias=...) for mapping.
    """

    service_name: str = pydantic.Field(alias='service.name')
    service_version: str = pydantic.Field(alias='service.version')
    os_type: str = pydantic.Field(alias='os.type')
    os_version: str = pydantic.Field(alias='os.version')
    host_arch: str = pydantic.Field(alias='host.arch')
    aggregation_temporality: str = pydantic.Field(alias='aggregation.temporality')
    user_customer_type: str = pydantic.Field(alias='user.customer_type')
    user_subscription_type: str = pydantic.Field(alias='user.subscription_type')


class MetricsAttributes(StrictModel):
    """
    Attributes on a metric data point.

    VALIDATION STATUS: VALIDATED
    OpenTelemetry-style dimensions with dot-notation keys.
    """

    # --- Always present ---
    user_id: str = pydantic.Field(alias='user.id')
    session_id: str = pydantic.Field(alias='session.id')
    organization_id: str = pydantic.Field(alias='organization.id')
    user_email: str = pydantic.Field(alias='user.email')
    user_account_uuid: str = pydantic.Field(alias='user.account_uuid')
    terminal_type: str = pydantic.Field(alias='terminal.type')

    # --- Present on token usage metrics only ---
    type: Literal['input', 'output', 'cacheRead', 'cacheCreation', 'user', 'cli'] | None = None
    model: str | None = None  # Model ID for token metrics


class MetricsDataPoint(StrictModel):
    """
    Single data point in a metrics report.

    VALIDATION STATUS: VALIDATED
    OpenTelemetry-style data point.
    """

    attributes: MetricsAttributes | None  # Null for some metrics
    value: float | int
    timestamp: str  # ISO 8601 timestamp


class MetricDefinition(StrictModel):
    """
    Single metric in a metrics report.

    VALIDATION STATUS: VALIDATED
    """

    name: str  # e.g., "claude_code.session.count", "claude_code.cost.usage"
    description: str
    unit: str  # e.g., "", "USD"
    data_points: Sequence[MetricsDataPoint]


class MetricsRequest(StrictModel):
    """
    Request to /api/claude_code/metrics.

    VALIDATION STATUS: VALIDATED
    OpenTelemetry-style metrics reporting.
    """

    resource_attributes: ResourceAttributes
    metrics: Sequence[MetricDefinition]


class MetricsResponse(StrictModel):
    """
    Response from /api/claude_code/metrics.

    VALIDATION STATUS: VALIDATED
    Same structure as telemetry response.
    """

    accepted_count: int
    rejected_count: int


class MetricsEnabledResponse(StrictModel):
    """
    Response from /api/claude_code/organizations/metrics_enabled.

    VALIDATION STATUS: VALIDATED
    Controls telemetry and VCS integration opt-in.
    """

    metrics_logging_enabled: bool  # Whether telemetry is enabled
    vcs_account_linking_enabled: bool  # Whether VCS account linking is enabled


# ==============================================================================
# OAuth Client Data (/api/oauth/claude_cli/client_data)
# ==============================================================================


class ClientDataResponse(StrictModel):
    """
    Response from /api/oauth/claude_cli/client_data.

    VALIDATION STATUS: VALIDATED
    Observed: {"client_data": {}}

    Contains client-specific configuration.
    ALWAYS EMPTY in observed captures - strict typing for fail-fast validation.
    """

    client_data: EmptyDict  # Always {} - will fail if API starts sending data


# ==============================================================================
# OAuth Account Settings (/api/oauth/account/settings)
# ==============================================================================

PaprikaMode = Literal['basic', 'extended', 'disabled']


class DismissedBanner(StrictModel):
    """
    Dismissed banner record.

    VALIDATION STATUS: VALIDATED
    """

    banner_id: str
    dismissed_at: str  # ISO 8601 timestamp


class AccountSettingsResponse(StrictModel):
    """
    Response from /api/oauth/account/settings.

    VALIDATION STATUS: VALIDATED
    User preferences and feature toggles.

    Note: This is a large structure with ~47 fields. With strict validation
    (extra='forbid'), all fields must be modeled or validation fails.
    """

    # Onboarding state
    has_seen_mm_examples: bool | None = None
    has_seen_starter_prompts: bool | None = None
    has_started_claudeai_onboarding: bool | None = None
    has_finished_claudeai_onboarding: bool | None = None
    has_acknowledged_mcp_app_dev_terms: bool | None = None
    onboarding_use_case: str | None = None

    # Feature toggles
    grove_enabled: bool | None = None
    paprika_mode: PaprikaMode | None = None  # "basic", "extended", or "disabled"
    enabled_saffron: bool | None = None
    enabled_saffron_search: bool | None = None
    enabled_artifacts_attachments: bool | None = None
    enabled_mm_pdfs: bool | None = None
    enabled_web_search: bool | None = None
    enabled_gdrive: bool | None = None
    enabled_gdrive_indexing: bool | None = None
    enabled_geolocation: bool | None = None
    enabled_compass: bool | None = None
    enabled_bananagrams: bool | None = None
    enabled_foccacia: bool | None = None
    enabled_sourdough: bool | None = None
    enabled_turmeric: bool | None = None
    enabled_yukon_gold: bool | None = None
    enabled_wiggle_egress: bool | None = None
    enabled_monkeys_in_a_barrel: bool | None = None
    enable_chat_suggestions: bool | None = None

    # MCP tools permissions (per-tool boolean map)
    enabled_mcp_tools: Mapping[str, bool] | None = None

    # UI state
    input_menu_pinned_items: Sequence[str] | None = None
    dismissed_claudeai_banners: Sequence[DismissedBanner] | None = None
    dismissed_artifacts_announcement: bool | None = None
    dismissed_artifact_feedback_form: bool | None = None
    dismissed_claude_code_spotlight: bool | None = None
    dismissed_saffron_themes: bool | None = None

    # Preview features
    preview_feature_uses_artifacts: bool | None = None
    preview_feature_uses_latex: bool | None = None
    preview_feature_uses_citations: bool | None = None
    preview_feature_uses_harmony: bool | None = None

    # Grove metadata
    grove_updated_at: str | None = None
    grove_notice_viewed_at: str | None = None

    # Wiggle egress
    wiggle_egress_allowed_hosts: Sequence[str] | None = None
    wiggle_egress_hosts_template: str | None = None
    wiggle_egress_spotlight_viewed_at: str | None = None

    # Internal tier info
    internal_tier_org_type: str | None = None
    internal_tier_rate_limit_tier: str | None = None
    internal_tier_seat_tier: str | None = None
    internal_tier_override_expires_at: str | None = None

    # Code review sharing
    ccr_sharing_auto_share_on_pr: bool | None = None
    ccr_sharing_enforce_repo_check: bool | None = None
    ccr_sharing_show_display_name: bool | None = None


# ==============================================================================
# Health Check (/api/hello)
# ==============================================================================


class HelloResponse(StrictModel):
    """
    Response from /api/hello.

    VALIDATION STATUS: VALIDATED
    Simple health check endpoint.
    """

    message: Literal['hello']


# ==============================================================================
# Model Access Check (/api/organization/{uuid}/claude_code_sonnet_1m_access)
# ==============================================================================


class ModelAccessResponse(StrictModel):
    """
    Response from /api/organization/{uuid}/claude_code_sonnet_1m_access.

    VALIDATION STATUS: VALIDATED
    Checks if organization has access to Sonnet 1M context model.
    """

    has_access: bool
    has_access_not_as_default: bool  # Has access but not set as default


# ==============================================================================
# Referral Eligibility (/api/oauth/organizations/{uuid}/referral/eligibility)
# ==============================================================================


class ReferralCodeDetails(StrictModel):
    """
    Referral code details.

    VALIDATION STATUS: VALIDATED
    """

    code: str  # Referral code
    campaign: str  # Campaign name (e.g., "claude_code_guest_pass")
    referral_link: str  # Full referral URL


class ReferralEligibilityResponse(StrictModel):
    """
    Response from /api/oauth/organizations/{uuid}/referral/eligibility.

    VALIDATION STATUS: VALIDATED
    Checks if user is eligible for referral program.
    """

    eligible: bool
    referral_code_details: ReferralCodeDetails | None = None
