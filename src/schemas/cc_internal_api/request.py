"""
Request schemas for Claude Code internal API.

These model the request payloads Claude Code sends to the API.
Validated against mitmproxy captures of actual Claude Code traffic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

import anthropic.types
import pydantic

from src.schemas.cc_internal_api.base import FromSdk, StrictModel
from src.schemas.cc_internal_api.common import CacheControl
from src.schemas.cc_internal_api.tool_input_schema import ToolInputProperty
from src.schemas.types import ModelId

# ==============================================================================
# System Block (API-only, not persisted)
# ==============================================================================


class SystemBlock(StrictModel):
    """
    Text block in the system parameter array.

    VALIDATION STATUS: VALIDATED
    Observed: system is array of {type, text, cache_control} blocks.
    CC uses cache_control: {"type": "ephemeral"} on system blocks.
    """

    type: Literal['text']
    text: str
    cache_control: CacheControl | None = None


# ==============================================================================
# Tool Definition (API-only, not persisted)
# ==============================================================================


class ToolInputSchema(StrictModel):
    """
    JSON Schema for tool input parameters.

    VALIDATION STATUS: VALIDATED
    Observed in tools[].input_schema.

    Properties are typed via ToolInputProperty - 15 distinct models covering
    all observed shapes (string, number, boolean, array, object variants).
    """

    type: Literal['object']
    properties: Mapping[str, ToolInputProperty] | None = None
    required: Sequence[str] | None = None
    additionalProperties: bool | None = None
    # JSON Schema version identifier (present in Claude Code tool definitions)
    schema_: str | None = pydantic.Field(default=None, alias='$schema')
    # MCP tools include title field for input schema name
    title: str | None = None


class ToolDefinition(StrictModel):
    """
    Tool definition sent in API requests.

    VALIDATION STATUS: VALIDATED
    Observed: 18 built-in tools in vanilla capture.

    Tool definitions are sent every request but NOT persisted to session files.
    """

    name: str
    description: str  # All Claude Code tools have descriptions
    input_schema: Annotated[
        ToolInputSchema,
        FromSdk(anthropic.types.ToolParam, 'input_schema'),
    ]


# ==============================================================================
# Message Content Block
# ==============================================================================


class TextContentBlock(StrictModel):
    """
    Text content block in messages.

    Observed in messages[].content[] for both user and assistant roles.
    May include cache_control for prompt caching.
    """

    type: Literal['text']
    text: str
    cache_control: CacheControl | None = None


class RequestThinkingBlock(StrictModel):
    """
    Thinking content block in assistant messages (multi-turn conversations).

    VALIDATION STATUS: VALIDATED (2737 cases)
    - signature: ALWAYS present (2737/2737 = 100%)

    When conversation history is sent back to the API, previous assistant
    thinking blocks are included with their cryptographic signatures.
    """

    type: Literal['thinking']
    thinking: str
    signature: str  # Always present - cryptographic signature for verification


class RequestToolUseBlock(StrictModel):
    """
    Tool use block in assistant messages (conversation history).

    VALIDATION STATUS: VALIDATED (3462 cases)
    - cache_control: Optional (31/3462 = 0.9%)

    When conversation history is sent back to the API, previous assistant
    tool calls are included.
    """

    type: Literal['tool_use']
    id: str
    name: str
    input: Mapping[str, Any]
    cache_control: CacheControl | None = None


class ToolResultTextItem(StrictModel):
    """
    Text item in tool_result list content.

    VALIDATION STATUS: VALIDATED (204 list items)
    All list items observed are this shape. Source: MCP tools (perplexity_research,
    perplexity_ask) and Task tool.
    """

    type: Literal['text']
    text: str


class RequestToolResultBlock(StrictModel):
    """
    Tool result block in user messages (conversation history).

    VALIDATION STATUS: VALIDATED (3462 cases)
    - content: Always present. String (3260/3462 = 94.2%) or list (202/3462 = 5.8%)
    - is_error: Optional (absent 2180, false 1042, true 240)
    - cache_control: Optional (30/3462 = 0.9%)

    Note: List content observed only from MCP tools and Task tool.
    When content is list, is_error is always absent (202/202 cases).
    """

    type: Literal['tool_result']
    tool_use_id: str
    content: str | Sequence[ToolResultTextItem]  # Never None (3462/3462)
    is_error: bool | None = None  # Absent=implicit success (genuinely optional)
    cache_control: CacheControl | None = None


# Union of all request content block types
RequestContentBlock = TextContentBlock | RequestThinkingBlock | RequestToolUseBlock | RequestToolResultBlock


# ==============================================================================
# Message
# ==============================================================================


class RequestMessage(StrictModel):
    """
    Message in the messages array.

    User messages contain text blocks (with optional cache_control).
    Assistant messages contain text and/or thinking blocks (with signature).
    """

    role: Literal['user', 'assistant']
    content: Sequence[RequestContentBlock] | str


# ==============================================================================
# Thinking Configuration
# ==============================================================================


class EnabledThinkingConfig(StrictModel):
    """Extended thinking enabled - budget_tokens is required."""

    type: Literal['enabled']
    budget_tokens: int


class DisabledThinkingConfig(StrictModel):
    """Extended thinking disabled - no budget_tokens field."""

    type: Literal['disabled']


ThinkingConfig = Annotated[
    EnabledThinkingConfig | DisabledThinkingConfig,
    pydantic.Field(discriminator='type'),
]


# ==============================================================================
# Context Management (Request)
# ==============================================================================


class ContextManagementEdit(StrictModel):
    """
    Context management edit directive.

    VALIDATION STATUS: VALIDATED
    Observed: {"type": "clear_thinking_20251015", "keep": "all"}
    """

    type: str  # e.g., "clear_thinking_20251015"
    keep: str  # e.g., "all" - always present when context_management is used


class RequestContextManagement(StrictModel):
    """
    Context management in requests.

    VALIDATION STATUS: VALIDATED
    Observed in request.context_management.
    """

    edits: Sequence[ContextManagementEdit]


# ==============================================================================
# Metadata
# ==============================================================================


class RequestMetadata(StrictModel):
    """
    Request metadata.

    VALIDATION STATUS: VALIDATED
    Observed: {"user_id": "..."}
    """

    user_id: str  # Always present in Claude Code requests


# ==============================================================================
# Messages Request
# ==============================================================================


class MessagesRequest(StrictModel):
    """
    Complete request payload for /v1/messages.

    VALIDATION STATUS: VALIDATED
    Observed: 9 top-level keys.

    This is the full structure Claude Code sends to the API.
    """

    model: ModelId
    max_tokens: int
    stream: bool | None = None
    system: Sequence[SystemBlock] | None = None
    messages: Sequence[RequestMessage]
    tools: Sequence[ToolDefinition] | None = None
    thinking: ThinkingConfig | None = None
    context_management: RequestContextManagement | None = None
    metadata: RequestMetadata  # Always present in Claude Code requests
    temperature: float | None = None  # Temperature for sampling
