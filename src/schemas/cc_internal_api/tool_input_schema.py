"""
Property types for tool.input_schema.properties in Claude Code API requests.

NOT a general-purpose JSON Schema implementation. This module models ONLY the
specific subset of JSON Schema that Claude Code uses when defining tool parameters
in /v1/messages requests.

Location in API request:
    request.tools[].input_schema.properties  →  Mapping[str, ToolInputProperty]

VALIDATION STATUS: VALIDATED against 206 property instances across all nesting levels.

Implementation approach:
- 15 distinct models for observed shapes (7 string, 2 number, 2 boolean, 2 array, 2 object)
- Union matching via extra='forbid' - only exact shapes validate
- No defaults, no optionality - each observed shape is a separate model

Observed shapes (from captures):
  String: 7 shapes (with/without description, enum, minLength, format)
  Number: 2 shapes (description only, or with bounds)
  Boolean: 2 shapes (description only, or with default)
  Array: 2 shapes (base, or with bounds) - recursive via items
  Object: 2 shapes (full with properties, or simple) - recursive via properties
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from src.schemas.cc_internal_api.base import StrictModel

# ==============================================================================
# String Schema Variants (type="string")
# ==============================================================================
# Observed shapes:
#   [description, type] - 78 instances
#   [description, enum, type] - 12 instances
#   [description, minLength, type] - 3 instances
#   [description, format, type] - 3 instances
#   [type] - 2 instances (type only, no description)
#   [minLength, type] - 6 instances (no description)
#   [enum, type] - 3 instances (no description)


class StringSchemaDescriptionOnly(StrictModel):
    """String property with description only."""

    type: Literal['string']
    description: str


class StringSchemaWithEnum(StrictModel):
    """String property with enum constraint."""

    type: Literal['string']
    description: str
    enum: Sequence[str]


class StringSchemaWithMinLength(StrictModel):
    """String property with minLength constraint."""

    type: Literal['string']
    description: str
    minLength: int


class StringSchemaWithFormat(StrictModel):
    """String property with format constraint (e.g., 'uri')."""

    type: Literal['string']
    description: str
    format: str


class StringSchemaTypeOnly(StrictModel):
    """String property with type only, no description."""

    type: Literal['string']


class StringSchemaMinLengthNoDesc(StrictModel):
    """String property with minLength but no description."""

    type: Literal['string']
    minLength: int


class StringSchemaEnumNoDesc(StrictModel):
    """String property with enum but no description."""

    type: Literal['string']
    enum: Sequence[str]


class StringSchemaWithTitle(StrictModel):
    """String property with title but no description (MCP tools)."""

    type: Literal['string']
    title: str


class StringSchemaWithDefault(StrictModel):
    """String property with default value."""

    type: Literal['string']
    description: str
    default: str


class StringSchemaWithTitleDefault(StrictModel):
    """String property with title and default but no description (MCP tools)."""

    type: Literal['string']
    title: str
    default: str


class StringSchemaWithTitleEnumDefault(StrictModel):
    """String property with title, enum, and default but no description (MCP tools)."""

    type: Literal['string']
    title: str
    enum: Sequence[str]
    default: str


# Union ordered: most specific first (more fields → less ambiguity)
StringSchema = (
    StringSchemaWithEnum
    | StringSchemaWithMinLength
    | StringSchemaWithFormat
    | StringSchemaWithDefault
    | StringSchemaWithTitleEnumDefault  # More specific than TitleDefault (has enum)
    | StringSchemaWithTitleDefault
    | StringSchemaMinLengthNoDesc
    | StringSchemaEnumNoDesc
    | StringSchemaWithTitle
    | StringSchemaDescriptionOnly
    | StringSchemaTypeOnly
)


# ==============================================================================
# Number Schema Variants (type="number" or "integer")
# ==============================================================================
# Observed shapes:
#   [description, type] - 24 instances
#   [default, description, maximum, minimum, type] - 3 instances


class NumberSchemaDescriptionOnly(StrictModel):
    """Number/integer property with description only."""

    type: Literal['number', 'integer']
    description: str


class NumberSchemaWithBounds(StrictModel):
    """Number property with min/max bounds and default."""

    type: Literal['number', 'integer']
    description: str
    default: int
    minimum: int
    maximum: int


class NumberSchemaWithBoundsNoDefault(StrictModel):
    """Number property with min/max bounds but no default (MCP tools)."""

    type: Literal['number', 'integer']
    description: str
    minimum: int
    maximum: int


class NumberSchemaWithDefaultOnly(StrictModel):
    """Number property with default but no bounds."""

    type: Literal['number', 'integer']
    description: str
    default: int


class NumberSchemaWithTitleDefault(StrictModel):
    """Number property with title and default but no description (MCP tools)."""

    type: Literal['number', 'integer']
    title: str
    default: int


class NumberSchemaTypeOnly(StrictModel):
    """Number property with type only, no description (used in anyOf)."""

    type: Literal['number', 'integer']


class NumberSchemaWithTitle(StrictModel):
    """Number property with title only, no default (MCP tools)."""

    type: Literal['number', 'integer']
    title: str


# Order: most specific first (more fields → less ambiguity)
NumberSchema = (
    NumberSchemaWithBounds
    | NumberSchemaWithBoundsNoDefault
    | NumberSchemaWithDefaultOnly
    | NumberSchemaWithTitleDefault
    | NumberSchemaWithTitle
    | NumberSchemaDescriptionOnly
    | NumberSchemaTypeOnly
)


# ==============================================================================
# Boolean Schema Variants (type="boolean")
# ==============================================================================
# Observed shapes:
#   [description, type] - 18 instances
#   [default, description, type] - 6 instances


class BooleanSchemaDescriptionOnly(StrictModel):
    """Boolean property with description only."""

    type: Literal['boolean']
    description: str


class BooleanSchemaWithDefault(StrictModel):
    """Boolean property with default value."""

    type: Literal['boolean']
    description: str
    default: bool


class BooleanSchemaWithTitleDefault(StrictModel):
    """Boolean property with title and default but no description (MCP tools)."""

    type: Literal['boolean']
    title: str
    default: bool


class BooleanSchemaWithTitle(StrictModel):
    """Boolean property with title only, no default (MCP tools)."""

    type: Literal['boolean']
    title: str


BooleanSchema = (
    BooleanSchemaWithDefault | BooleanSchemaWithTitleDefault | BooleanSchemaWithTitle | BooleanSchemaDescriptionOnly
)


# ==============================================================================
# Array Schema Variants (type="array")
# ==============================================================================
# Observed shapes:
#   [description, items, type] - 9 instances
#   [description, items, maxItems, minItems, type] - 6 instances
#
# Note: items is recursive (ToolInputProperty)


class ArraySchemaBase(StrictModel):
    """Array property with items definition."""

    type: Literal['array']
    description: str
    items: ToolInputProperty


class ArraySchemaWithBounds(StrictModel):
    """Array property with min/max items constraints."""

    type: Literal['array']
    description: str
    items: ToolInputProperty
    minItems: int
    maxItems: int


class ArraySchemaItemsOnly(StrictModel):
    """Array property with items but no description (used in anyOf)."""

    type: Literal['array']
    items: ToolInputProperty


ArraySchema = ArraySchemaWithBounds | ArraySchemaBase | ArraySchemaItemsOnly


# ==============================================================================
# Null Schema (type="null")
# ==============================================================================


class NullSchema(StrictModel):
    """Null type for nullable unions in anyOf."""

    type: Literal['null']


# ==============================================================================
# Object Schema Variants (type="object")
# ==============================================================================
# Observed shapes:
#   [additionalProperties, properties, required, type] - 9 instances (NO description)
#   [additionalProperties, description, type] - 3 instances (NO properties/required)
#   [additionalProperties, description, type] - NEW: additionalProperties is nested schema
#
# Note: properties is recursive (Mapping[str, ToolInputProperty])
# Note: additionalProperties can be bool OR nested schema (ToolInputProperty)


class ObjectSchemaFull(StrictModel):
    """Object property with full definition (properties, required, additionalProperties).

    Note: This shape has NO description field in observed captures.
    """

    type: Literal['object']
    properties: Mapping[str, ToolInputProperty]
    required: Sequence[str]
    additionalProperties: bool


class ObjectSchemaSimple(StrictModel):
    """Object property with just additionalProperties (bool) and description.

    Note: This shape has NO properties or required fields.
    additionalProperties is a boolean (true/false), not a nested schema.
    """

    type: Literal['object']
    description: str
    additionalProperties: bool


class ObjectSchemaWithNestedAdditionalProps(StrictModel):
    """Object property with additionalProperties as a nested schema.

    Example: {"type": "object", "additionalProperties": {"type": "string"}, "description": "..."}
    Used for dict-like objects where values must match a schema.
    """

    type: Literal['object']
    description: str
    additionalProperties: ToolInputProperty  # Nested schema, not bool!


class ObjectSchemaPropertiesOnly(StrictModel):
    """Object property with properties/required but no additionalProperties (MCP tools).

    This shape appears in MCP tool array items where the schema defines
    structure but doesn't explicitly forbid/allow additional properties.
    """

    type: Literal['object']
    properties: Mapping[str, ToolInputProperty]
    required: Sequence[str]


# Order matters: most specific first (more fields → less ambiguity)
ObjectSchema = (
    ObjectSchemaFull | ObjectSchemaWithNestedAdditionalProps | ObjectSchemaPropertiesOnly | ObjectSchemaSimple
)


# ==============================================================================
# AnyOf Schema (nullable unions)
# ==============================================================================
# Observed shape: {"anyOf": [...], "default": null, "title": "..."}
# Used for nullable arrays/objects in MCP tools


class AnyOfSchema(StrictModel):
    """Schema with anyOf union type (MCP tools with nullable arrays).

    Example: {"anyOf": [{"type": "array", "items": {...}}, {"type": "null"}], "default": null, "title": "..."}
    """

    anyOf: Sequence[ToolInputProperty]
    default: None  # Always null for nullable unions
    title: str


class AnyOfSchemaNoDefault(StrictModel):
    """Schema with anyOf union type but no default (some MCP tools).

    Example: {"anyOf": [{"type": "integer"}, {"type": "null"}], "title": "..."}
    """

    anyOf: Sequence[ToolInputProperty]
    title: str


# ==============================================================================
# Combined Tool Input Property Type
# ==============================================================================
# Union of all property types. Pydantic tries each variant in order.
# With extra='forbid' on each model, only exact shape matches validate.


ToolInputProperty = (
    StringSchema
    | NumberSchema
    | BooleanSchema
    | ArraySchema
    | ObjectSchema
    | NullSchema
    | AnyOfSchema
    | AnyOfSchemaNoDefault
)
"""
A single property in tool.input_schema.properties.

Pydantic tries each variant in union order. With extra='forbid', only exact
shape matches validate - no discriminator needed.

This is recursive:
- ArraySchema.items -> ToolInputProperty
- ObjectSchemaFull.properties -> Mapping[str, ToolInputProperty]
- AnyOfSchema.anyOf -> Sequence[ToolInputProperty]
"""

# Pydantic needs model_rebuild() for recursive forward references
ArraySchemaBase.model_rebuild()
ArraySchemaWithBounds.model_rebuild()
ArraySchemaItemsOnly.model_rebuild()
ObjectSchemaFull.model_rebuild()
ObjectSchemaWithNestedAdditionalProps.model_rebuild()
ObjectSchemaPropertiesOnly.model_rebuild()
AnyOfSchema.model_rebuild()
AnyOfSchemaNoDefault.model_rebuild()
