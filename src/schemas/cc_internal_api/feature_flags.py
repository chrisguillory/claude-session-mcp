"""
Feature flag schemas for Claude Code internal API.

This module provides strict per-flag typing for the /api/eval/sdk-* endpoint.
Each feature flag has its value typed according to observed captures:
- Boolean flags: Simple on/off toggles
- String flags: Experiment variation names
- Config flags: Complex configuration objects

Separated from internal_endpoints.py for clarity and maintainability.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal, TypedDict

import pydantic

from src.schemas.cc_internal_api.base import EmptyDict, StrictModel

# ==============================================================================
# Feature Source
# ==============================================================================

FeatureSource = Literal['force', 'defaultValue', 'experiment']


# ==============================================================================
# Experiment Types
# ==============================================================================


class ExperimentConfig(StrictModel):
    """
    Experiment configuration for a feature flag.

    VALIDATION STATUS: VALIDATED
    """

    key: str  # Experiment key
    variations: Sequence[str]  # Possible variations


class ExperimentResult(StrictModel):
    """
    Result of experiment assignment.

    VALIDATION STATUS: VALIDATED
    """

    inExperiment: bool  # Whether user is in experiment
    variationId: int  # Assigned variation ID
    value: str  # Assigned variation value
    hashUsed: bool  # Whether hash was used for assignment


# ==============================================================================
# Feature Flag Config Value Schemas
# ==============================================================================


class FeedbackSurveyConfig(StrictModel):
    """
    Config value for tengu_feedback_survey_config flag.

    VALIDATION STATUS: VALIDATED
    Controls feedback survey timing and probability.
    """

    minTimeBeforeFeedbackMs: int
    minTimeBetweenFeedbackMs: int
    minTimeBetweenGlobalFeedbackMs: int
    minUserTurnsBeforeFeedback: int
    minUserTurnsBetweenFeedback: int
    hideThanksAfterMs: int
    onForModels: Sequence[str]  # e.g., ["*"]
    probability: float  # 0.0 to 1.0


class VersionConfig(StrictModel):
    """
    Config value for tengu_version_config flag.

    VALIDATION STATUS: VALIDATED
    Specifies minimum required version.
    """

    minVersion: str  # e.g., "1.0.24"


class SpinnerWordsConfig(StrictModel):
    """
    Config value for tengu_spinner_words flag.

    VALIDATION STATUS: VALIDATED
    List of words shown in the CLI spinner during processing.
    """

    words: Sequence[str]  # e.g., ["Thinking", "Processing", "Clauding", ...]


# ==============================================================================
# Per-Flag Feature Value Types
# ==============================================================================


class BoolFeatureValue(StrictModel):
    """
    Feature flag with boolean value.

    Used for simple on/off toggle flags like tengu_sumi, tengu_scratch, etc.
    """

    value: bool
    on: bool
    off: bool
    source: FeatureSource
    experiment: ExperimentConfig | None = None
    experimentResult: ExperimentResult | None = None


class StringFeatureValue(StrictModel):
    """
    Feature flag with string value.

    Used for experiment flags that return variation names like "OFF", "inherit".
    """

    value: str
    on: bool
    off: bool
    source: FeatureSource
    experiment: ExperimentConfig | None = None
    experimentResult: ExperimentResult | None = None


class FeedbackSurveyFeatureValue(StrictModel):
    """Feature flag with feedback survey config value."""

    value: FeedbackSurveyConfig
    on: bool
    off: bool
    source: FeatureSource
    experiment: ExperimentConfig | None = None
    experimentResult: ExperimentResult | None = None


class VersionConfigFeatureValue(StrictModel):
    """Feature flag with version config value."""

    value: VersionConfig
    on: bool
    off: bool
    source: FeatureSource
    experiment: ExperimentConfig | None = None
    experimentResult: ExperimentResult | None = None


class SpinnerWordsFeatureValue(StrictModel):
    """Feature flag with spinner words config value."""

    value: SpinnerWordsConfig
    on: bool
    off: bool
    source: FeatureSource
    experiment: ExperimentConfig | None = None
    experimentResult: ExperimentResult | None = None


# ==============================================================================
# Strictly Typed Features Dict
# ==============================================================================


class FeaturesDict(TypedDict, total=False):
    """
    All known feature flags with strict per-flag typing.

    Using total=False since different users may see different subsets of flags.
    New flags from the API will cause validation failure (fail-fast).
    """

    # --- Boolean flags (simple toggles) ---
    auto_migrate_to_native: BoolFeatureValue
    tengu_accept_with_feedback: BoolFeatureValue
    tengu_ant_attribution_header_new: BoolFeatureValue
    tengu_c4w_usage_limit_notifications_enabled: BoolFeatureValue
    tengu_disable_bypass_permissions_mode: BoolFeatureValue
    tengu_gha_plugin_code_review: BoolFeatureValue
    tengu_mcp_tool_search: BoolFeatureValue
    tengu_pid_based_version_locking: BoolFeatureValue
    tengu_prompt_suggestion: BoolFeatureValue
    tengu_react_vulnerability_warning: BoolFeatureValue
    tengu_scratch: BoolFeatureValue
    tengu_sumi: BoolFeatureValue
    tengu_thinkback: BoolFeatureValue
    tengu_tool_pear: BoolFeatureValue
    tengu_tool_result_persistence: BoolFeatureValue
    tengu_vscode_review_upsell: BoolFeatureValue
    tengu_year_end_2025_campaign_promo: BoolFeatureValue

    # --- String flags (experiment variations) ---
    cc_test_experiment_deviceID_flag: StringFeatureValue
    doorbell_bottle: StringFeatureValue
    persimmon_marble_flag: StringFeatureValue
    strawberry_granite_flag: StringFeatureValue

    # --- Config flags (complex objects) ---
    tengu_feedback_survey_config: FeedbackSurveyFeatureValue
    tengu_spinner_words: SpinnerWordsFeatureValue
    tengu_version_config: VersionConfigFeatureValue


# Union of all feature value types for generic handling
FeatureValue = (
    BoolFeatureValue
    | StringFeatureValue
    | FeedbackSurveyFeatureValue
    | VersionConfigFeatureValue
    | SpinnerWordsFeatureValue
)


# ==============================================================================
# Eval Request/Response (moved from internal_endpoints.py)
# ==============================================================================


class EvalAttributes(StrictModel):
    """
    Attributes sent with feature flag evaluation request.

    VALIDATION STATUS: VALIDATED
    Identifies user/device for feature targeting.
    """

    id: str  # Device ID (hashed)
    sessionId: str  # Session UUID
    deviceID: str  # Device ID (same as id)
    organizationUUID: str  # Organization UUID
    accountUUID: str  # Account UUID
    userType: Literal['external', 'internal']
    subscriptionType: Literal['free', 'pro', 'team', 'max', 'enterprise']
    firstTokenTime: int  # Timestamp of first token
    appVersion: str  # Claude Code version


class EvalRequest(StrictModel):
    """
    Request to /api/eval/sdk-{code}.

    VALIDATION STATUS: VALIDATED
    Feature flag evaluation request.
    """

    attributes: EvalAttributes
    # ALWAYS EMPTY in observed captures - strict typing for fail-fast validation
    forcedVariations: EmptyDict  # Always {} - will fail if API starts sending data
    # max_length=0 enforces empty list (Pydantic doesn't support Sequence[Never])
    forcedFeatures: Annotated[Sequence[str], pydantic.Field(max_length=0)]
    url: Literal['']  # Always "" - will fail if API starts sending data


class EvalResponse(StrictModel):
    """
    Response from /api/eval/sdk-{code}.

    VALIDATION STATUS: VALIDATED
    Contains all feature flags for the user with strict per-flag typing.
    """

    features: FeaturesDict


# Known feature flag names for reference
KNOWN_FEATURE_FLAGS = list(FeaturesDict.__annotations__.keys())
