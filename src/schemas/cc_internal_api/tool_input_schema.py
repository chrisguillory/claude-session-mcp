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


# Union ordered: most specific first (more fields → less ambiguity)
StringSchema = (
    StringSchemaWithEnum
    | StringSchemaWithMinLength
    | StringSchemaWithFormat
    | StringSchemaMinLengthNoDesc
    | StringSchemaEnumNoDesc
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


NumberSchema = NumberSchemaWithBounds | NumberSchemaDescriptionOnly


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


BooleanSchema = BooleanSchemaWithDefault | BooleanSchemaDescriptionOnly


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


ArraySchema = ArraySchemaWithBounds | ArraySchemaBase


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


# Order matters: try nested schema variant before bool variant
ObjectSchema = ObjectSchemaFull | ObjectSchemaWithNestedAdditionalProps | ObjectSchemaSimple


# ==============================================================================
# Combined Tool Input Property Type
# ==============================================================================
# Union of all property types. Pydantic tries each variant in order.
# With extra='forbid' on each model, only exact shape matches validate.


ToolInputProperty = StringSchema | NumberSchema | BooleanSchema | ArraySchema | ObjectSchema
"""
A single property in tool.input_schema.properties.

Pydantic tries each variant in union order. With extra='forbid', only exact
shape matches validate - no discriminator needed.

This is recursive:
- ArraySchema.items -> ToolInputProperty
- ObjectSchemaFull.properties -> Mapping[str, ToolInputProperty]
"""

# Pydantic needs model_rebuild() for recursive forward references
ArraySchemaBase.model_rebuild()
ArraySchemaWithBounds.model_rebuild()
ObjectSchemaFull.model_rebuild()
ObjectSchemaWithNestedAdditionalProps.model_rebuild()
