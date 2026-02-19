"""
Shared type definitions for schemas.

Centralizes common type annotations used across session and API schemas.

Layering:
- This module provides FOUNDATION types (BaseStrictModel, empty markers, ModelId)
- Domain packages (session/, cc_internal_api/) import from here
- Domain packages may define their own StrictModel that inherits from BaseStrictModel
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Annotated, Any, Literal

import pydantic

# ==============================================================================
# Base Strict Model (Foundation)
# ==============================================================================


class BaseStrictModel(pydantic.BaseModel):
    """
    Foundation strict model - domain packages inherit from this.

    Uses extra='forbid' to reject unknown fields - any field not modeled
    causes immediate validation failure (fail-fast).

    Domain packages (session/, cc_internal_api/) should define their own
    StrictModel that inherits from this, enabling domain-specific customization
    while sharing the core validation config.
    """

    model_config = pydantic.ConfigDict(
        extra='forbid',  # Reject unknown fields (fail-fast)
        strict=True,  # Strict type coercion
        frozen=True,  # Immutable after creation
    )


# ==============================================================================
# Permissive Model (Foundation)
# ==============================================================================


class PermissiveModel(pydantic.BaseModel):
    """
    Foundation permissive model for typed fallbacks in unions.

    Symmetry with BaseStrictModel:
    - BaseStrictModel: extra='forbid' (rejects unknown fields)
    - PermissiveModel: extra='allow' (accepts unknown fields)

    Use as the LAST type in typed unions to catch unknown structures:

        DynamicConfigValue = Annotated[
            TypedConfig1 | TypedConfig2 | UnknownConfigValue,
            pydantic.Field(union_mode='left_to_right'),
        ]

        class UnknownConfigValue(PermissiveModel):
            pass

    Detection: isinstance(x, PermissiveModel) catches all fallback usages,
    enabling validation scripts to report untyped structures.
    """

    model_config = pydantic.ConfigDict(
        extra='allow',  # Accept unknown fields (graceful fallback)
        strict=True,  # Strict type coercion for known fields
        frozen=True,  # Immutable after creation
    )

    def get_extra_fields(self) -> dict[str, object]:
        """Get extra fields captured by this permissive model.

        Returns only the unknown fields, not defined model fields.
        Useful for inspection and logging of untyped structures.
        """
        return dict(self.__pydantic_extra__) if self.__pydantic_extra__ else {}


# ==============================================================================
# Empty JSON Types
# ==============================================================================
#
# These types represent always-empty JSON structures observed in API traffic.
# Using explicit types rather than inline constraints because:
# 1. Pydantic can't validate against Never
# 2. Named types document semantic meaning clearly
# 3. Validation fails immediately if the API starts sending data
# 4. Union with populated types works naturally
#
# Usage:
#     sdk_params: EmptyDict       # Always {} - fails if API sends {"key": "value"}
#     applied_edits: EmptySequence # Always [] - fails if API sends ["item"]
#     previous_fields: DerivedFields | EmptyDict  # Sometimes empty, sometimes populated
#
# TODO: Analyze session models for EmptyDict opportunities - there may be
# always-empty dict fields that should use EmptyDict for strictness.
# ==============================================================================


class EmptyDict(BaseStrictModel):
    """
    Marker type for empty JSON object {} in API traffic.

    With extra='forbid', a model with no fields will only validate against
    an empty dict. Used for fields that are always {} in observed captures.

    Example:
        sdk_params: EmptyDict  # Always {} in API traffic
    """

    pass


EmptySequence = Annotated[Sequence[Any], pydantic.Field(max_length=0)]
"""
Marker type for empty JSON array [] in API traffic.

Used for fields that are always [] in observed captures.
The element type (Any) doesn't matter since the sequence is always empty.

Example:
    applied_edits: EmptySequence  # Always [] in API traffic
"""


# ==============================================================================
# Primitive Types
# ==============================================================================

type JsonDatetime = Annotated[datetime, pydantic.Field(strict=False)]
"""Pydantic-enhanced datetime for JSON serialization (allows string->datetime conversion)."""

type PathStr = str
"""A filesystem path (file or directory) as a string."""


# ==============================================================================
# Model ID Types
# ==============================================================================

# Model IDs actually observed in session data (validated against 180k+ records)
# Used for strict validation of the model field in Message and AssistantRecord
ModelId = Literal[
    '<synthetic>',
    'claude-haiku-4-5-20251001',
    'claude-opus-4-1-20250805',
    'claude-opus-4-5-20251101',
    'claude-opus-4-6',
    'claude-sonnet-4-6',
    'claude-sonnet-4-5-20250929',
    'haiku',
    'opus',
    'sonnet',
]

# All model IDs known to Claude Code 2.0.73 (superset - not currently used)
# Includes older models and aliases that may appear in future sessions
_AllModelIds = Literal[
    # Claude 3 models
    'claude-3-5-haiku',
    'claude-3-5-haiku-20241022',
    'claude-3-5-haiku@20241022',
    'claude-3-5-sonnet',
    'claude-3-5-sonnet-20241022',
    'claude-3-5-sonnet-v2@20241022',
    'claude-3-7-sonnet',
    'claude-3-7-sonnet-20250219',
    'claude-3-7-sonnet-latest',
    'claude-3-7-sonnet@20250219',
    'claude-3-haiku',
    'claude-3-opus',
    'claude-3-opus-20240229',
    'claude-3-sonnet',
    'claude-3-sonnet-20240229',
    # Claude 4 models
    'claude-4-opus-20250514',
    'claude-haiku-4',
    'claude-haiku-4-5',
    'claude-haiku-4-5-20251001',
    'claude-haiku-4-5@20251001',
    'claude-opus-4',
    'claude-opus-4-0',
    'claude-opus-4-1',
    'claude-opus-4-1-20250805',
    'claude-opus-4-1@20250805',
    'claude-opus-4-20250514',
    'claude-opus-4-5',
    'claude-opus-4-5-20251101',
    'claude-opus-4-5@20251101',
    'claude-opus-4@20250514',
    'claude-sonnet-4',
    'claude-sonnet-4-20250514',
    'claude-sonnet-4-5',
    'claude-sonnet-4-5-20250929',
    'claude-sonnet-4-5@20250929',
    'claude-sonnet-4-6',
    'claude-sonnet-4@20250514',
    # Short aliases (used in Task tool input.model)
    'haiku',
    'sonnet',
    'opus',
    # Special
    '<synthetic>',
]
