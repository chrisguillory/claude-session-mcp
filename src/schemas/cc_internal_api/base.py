"""
Base classes and type markers for Claude Code internal API schemas.

This module provides:
- StrictModel: Base class with strict validation (extra='forbid')
- Type correspondence markers: Link fields to session schemas and SDK types
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

# ==============================================================================
# Strict Base Model
# ==============================================================================


class StrictModel(BaseModel):
    """
    Base model for API schemas with strict validation.

    Uses extra='forbid' to reject unknown fields - any field not modeled
    causes immediate validation failure (fail-fast).

    This was previously 'ignore' during the observation phase. Now that
    schemas are complete, we enforce strict field coverage.
    """

    model_config = ConfigDict(
        extra='forbid',  # Reject unknown fields (fail-fast)
        strict=True,  # Strict type coercion
        frozen=True,  # Immutable after creation
    )


# ==============================================================================
# Empty Body Type
# ==============================================================================


class EmptyBody(StrictModel):
    """
    Marker type for empty HTTP bodies at the capture layer.

    This structure is produced by intercept_traffic.py's _parse_body()
    when HTTP content is empty (size=0). It's an infrastructure convention
    for the capture layer, not an API protocol.

    Used for:
    - GET request bodies (no request content)
    - 204 No Content responses (e.g., Statsig delta with no updates)
    - Any HTTP message with empty body
    """

    empty: Literal[True]
    size: int


class EmptyDict(StrictModel):
    """
    Marker type for empty JSON object {} in API traffic.

    With extra='forbid', a model with no fields will only validate against
    an empty dict. This is used for fields that are {} on first request and
    populated on subsequent requests (e.g., Statsig previousDerivedFields).

    Using a named model rather than bare Mapping[str, Never] because:
    1. Pydantic can't validate against Never
    2. This clearly documents the semantic meaning
    3. Union with populated model works naturally
    """

    pass


# ==============================================================================
# Type Correspondence Markers
# ==============================================================================
#
# These markers are used with typing.Annotated to create typed links between:
# 1. Our API schemas (this module)
# 2. Session schemas (src.schemas.session) - what CC persists
# 3. Anthropic SDK types (anthropic.types) - public API reference
#
# Example usage:
#   from typing import Annotated
#   from src.schemas import session
#   import anthropic.types
#
#   class ApiUsage(StrictModel):
#       input_tokens: Annotated[int,
#           FromSession(session.models.TokenUsage, 'input_tokens'),
#           FromSdk(anthropic.types.Usage, 'input_tokens'),
#       ]
#
# Benefits:
# - IDE navigation: Ctrl+click navigates to source schema
# - Runtime inspection: get_type_hints(cls, include_extras=True) exposes markers
# - Documentation: Explicit about which types correspond
#
# Note: Fields WITHOUT markers are implicitly "API only" - they exist in
# CC's internal API traffic but have no session or SDK counterpart.
# ==============================================================================

ValidationStatus = Literal['validated', 'inferred', 'reference']


@dataclass(frozen=True)
class FromSession:
    """
    Marks a field as corresponding to a session schema field.

    Use in Annotated[] to document the relationship between API types
    and session types (what Claude Code persists to JSONL files).

    Args:
        source_type: The session schema class (e.g., session.models.TokenUsage)
        field: The field name in the session schema (e.g., 'input_tokens')
        status: Validation status of this correspondence
    """

    source_type: type
    field: str | None = None
    status: ValidationStatus = 'inferred'

    def __repr__(self) -> str:
        type_name = self.source_type.__name__
        if self.field:
            return f'FromSession({type_name}.{self.field}, status={self.status!r})'
        return f'FromSession({type_name}, status={self.status!r})'


@dataclass(frozen=True)
class FromSdk:
    """
    Marks a field as corresponding to an Anthropic SDK type field.

    Use in Annotated[] to document the relationship between our API types
    and the official anthropic-sdk-python types (reference only).

    Args:
        source_type: The SDK type class (e.g., anthropic.types.Usage)
        field: The field name in the SDK type (e.g., 'input_tokens')
    """

    source_type: type
    field: str | None = None

    def __repr__(self) -> str:
        type_name = self.source_type.__name__
        if self.field:
            return f'FromSdk({type_name}.{self.field})'
        return f'FromSdk({type_name})'
