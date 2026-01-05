"""
Request schemas for Claude Code internal API.

These model the request payloads Claude Code sends to the API.
Validated against mitmproxy captures of actual Claude Code traffic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

import anthropic.types

from src.schemas.cc_internal_api.base import FromSdk, PermissiveModel
from src.schemas.cc_internal_api.common import CacheControl
from src.schemas.types import ModelId

# ==============================================================================
# System Block (API-only, not persisted)
# ==============================================================================


class SystemBlock(PermissiveModel):
    """
    Text block in the system parameter array.

    VALIDATION STATUS: VALIDATED (2026-01-02)
    Observed: system is array of {type, text, cache_control} blocks.
    CC uses cache_control: {"type": "ephemeral"} on system blocks.
    """

    type: Literal['text']
    text: str
    cache_control: CacheControl | None = None


# ==============================================================================
# Tool Definition (API-only, not persisted)
# ==============================================================================


class ToolInputSchema(PermissiveModel):
    """
    JSON Schema for tool input parameters.

    VALIDATION STATUS: VALIDATED (2026-01-02)
    Observed in tools[].input_schema.
    """

    type: Literal['object']
    properties: Mapping[str, Any] | None = None
    required: Sequence[str] | None = None
    additionalProperties: bool | None = None


class ToolDefinition(PermissiveModel):
    """
    Tool definition sent in API requests.

    VALIDATION STATUS: VALIDATED (2026-01-02)
    Observed: 18 built-in tools in vanilla capture.

    Tool definitions are sent every request but NOT persisted to session files.
    """

    name: str
    description: str | None = None
    input_schema: Annotated[
        ToolInputSchema,
        FromSdk(anthropic.types.ToolParam, 'input_schema'),
    ]


# ==============================================================================
# Message Content Block
# ==============================================================================


class TextContentBlock(PermissiveModel):
    """
    Text content block in messages.

    Observed in messages[].content[] for both user and assistant roles.
    May include cache_control for prompt caching.
    """

    type: Literal['text']
    text: str
    cache_control: CacheControl | None = None


class RequestThinkingBlock(PermissiveModel):
    """
    Thinking content block in assistant messages (multi-turn conversations).

    When conversation history is sent back to the API, previous assistant
    thinking blocks are included with their cryptographic signatures.
    """

    type: Literal['thinking']
    thinking: str
    signature: str | None = None


# Union of all request content block types
RequestContentBlock = TextContentBlock | RequestThinkingBlock


# ==============================================================================
# Message
# ==============================================================================


class RequestMessage(PermissiveModel):
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


class ThinkingConfig(PermissiveModel):
    """
    Extended thinking configuration.

    VALIDATION STATUS: VALIDATED (2026-01-02)
    Observed: {"budget_tokens": 31999, "type": "enabled"}
    """

    type: Literal['enabled', 'disabled']
    budget_tokens: int | None = None


# ==============================================================================
# Context Management (Request)
# ==============================================================================


class ContextManagementEdit(PermissiveModel):
    """
    Context management edit directive.

    VALIDATION STATUS: VALIDATED (2026-01-02)
    Observed: {"type": "clear_thinking_20251015", "keep": "all"}
    """

    type: str  # e.g., "clear_thinking_20251015"
    keep: str | None = None  # e.g., "all"


class RequestContextManagement(PermissiveModel):
    """
    Context management in requests.

    VALIDATION STATUS: VALIDATED (2026-01-02)
    Observed in request.context_management.
    """

    edits: Sequence[ContextManagementEdit]


# ==============================================================================
# Metadata
# ==============================================================================


class RequestMetadata(PermissiveModel):
    """
    Request metadata.

    VALIDATION STATUS: VALIDATED (2026-01-02)
    Observed: {"user_id": "..."}
    """

    user_id: str | None = None


# ==============================================================================
# Messages Request
# ==============================================================================


class MessagesRequest(PermissiveModel):
    """
    Complete request payload for /v1/messages.

    VALIDATION STATUS: VALIDATED (2026-01-02)
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
    metadata: RequestMetadata | None = None
