"""
SSE streaming event schemas for Claude Code internal API.

These model the Server-Sent Events (SSE) that Claude Code receives
during streaming API responses. Validated against mitmproxy captures.

Event sequence:
    message_start -> content_block_start -> [content_block_delta]* ->
    content_block_stop -> message_delta -> message_stop

With 'ping' events interspersed for keepalive.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

from pydantic import Discriminator

from src.schemas.cc_internal_api.base import StrictModel
from src.schemas.cc_internal_api.common import ApiCacheCreation
from src.schemas.cc_internal_api.response import StopReason
from src.schemas.types import ModelId

# ==============================================================================
# Delta Types (content_block_delta payloads)
# ==============================================================================


class TextDelta(StrictModel):
    """
    Text delta in streaming response.

    VALIDATION STATUS: VALIDATED
    Observed in content_block_delta events.
    """

    type: Literal['text_delta']
    text: str


class ThinkingDelta(StrictModel):
    """
    Thinking delta in streaming response.

    VALIDATION STATUS: INFERRED
    Expected based on extended thinking feature.
    """

    type: Literal['thinking_delta']
    thinking: str


class SignatureDelta(StrictModel):
    """
    Signature delta for thinking blocks.

    VALIDATION STATUS: INFERRED
    Expected for cryptographic signatures on thinking blocks.
    """

    type: Literal['signature_delta']
    signature: str


class InputJsonDelta(StrictModel):
    """
    Input JSON delta for tool use blocks.

    VALIDATION STATUS: INFERRED
    Expected for streaming tool inputs.
    """

    type: Literal['input_json_delta']
    partial_json: str


# Union of all delta types
DeltaContent = Annotated[
    TextDelta | ThinkingDelta | SignatureDelta | InputJsonDelta,
    Discriminator('type'),
]


# ==============================================================================
# Content Block Types (content_block_start payloads)
# ==============================================================================


class TextBlockStart(StrictModel):
    """
    Text block start in streaming response.

    VALIDATION STATUS: VALIDATED
    Observed in content_block_start events.
    """

    type: Literal['text']
    text: str  # Usually empty string at start


class ThinkingBlockStart(StrictModel):
    """
    Thinking block start in streaming response.

    VALIDATION STATUS: INFERRED
    Expected based on extended thinking feature.
    """

    type: Literal['thinking']
    thinking: str  # Usually empty string at start
    signature: str  # Usually empty string at start


class ToolUseBlockStart(StrictModel):
    """
    Tool use block start in streaming response.

    VALIDATION STATUS: INFERRED
    Expected based on tool use feature.
    """

    type: Literal['tool_use']
    id: str
    name: str
    # STREAMING BEHAVIOR: Empty {} at block start, populated incrementally via deltas.
    # Final accumulated input should match session.models.ToolInput union types.
    # Cannot use EmptyDict here - value evolves during streaming.
    input: Mapping[str, Any]


# Union of content block start types
ContentBlockStart = Annotated[
    TextBlockStart | ThinkingBlockStart | ToolUseBlockStart,
    Discriminator('type'),
]


# ==============================================================================
# Initial Message (message_start payload)
# ==============================================================================


class InitialUsage(StrictModel):
    """
    Initial usage in message_start event.

    VALIDATION STATUS: VALIDATED
    Observed in message_start.message.usage.
    """

    input_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    cache_creation: ApiCacheCreation
    output_tokens: int  # Usually small at start (e.g., 8)
    service_tier: Literal['standard']


class InitialMessage(StrictModel):
    """
    Initial message in message_start event.

    VALIDATION STATUS: VALIDATED
    Observed in message_start.message.
    """

    model: ModelId
    id: str  # e.g., "msg_01WsgytDj2C14cRxDGqz36ZF"
    type: Literal['message']
    role: Literal['assistant']
    content: Sequence[Any]  # Usually empty at start
    stop_reason: None  # Always null at start
    stop_sequence: None  # Always null at start
    usage: InitialUsage


# ==============================================================================
# Message Delta (message_delta payload)
# ==============================================================================


class MessageDeltaUsage(StrictModel):
    """
    Usage in message_delta event.

    VALIDATION STATUS: VALIDATED
    Observed in message_delta.usage.

    Note: Simpler than InitialUsage - no cache_creation nested object.
    """

    input_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    output_tokens: int


class MessageDeltaPayload(StrictModel):
    """
    Delta payload in message_delta event.

    VALIDATION STATUS: VALIDATED
    Observed in message_delta.delta.
    """

    stop_reason: StopReason
    stop_sequence: str | None


class AppliedEdit(StrictModel):
    """
    Applied context management edit.

    VALIDATION STATUS: INFERRED
    Based on request-side ContextManagementEdit structure.
    Always observed as empty array - this types what we'd expect when populated.
    """

    type: str  # Edit type, e.g., "clear_thinking_20251015"


class MessageDeltaContextManagement(StrictModel):
    """
    Context management in message_delta event.

    VALIDATION STATUS: VALIDATED
    Observed in message_delta.context_management.
    """

    applied_edits: Sequence[AppliedEdit]


# ==============================================================================
# SSE Event Types
# ==============================================================================


class MessageStartEvent(StrictModel):
    """
    message_start SSE event.

    VALIDATION STATUS: VALIDATED
    First event in streaming response sequence.
    """

    type: Literal['message_start']
    message: InitialMessage


class ContentBlockStartEvent(StrictModel):
    """
    content_block_start SSE event.

    VALIDATION STATUS: VALIDATED
    Marks start of a content block.
    """

    type: Literal['content_block_start']
    index: int
    content_block: ContentBlockStart


class ContentBlockDeltaEvent(StrictModel):
    """
    content_block_delta SSE event.

    VALIDATION STATUS: VALIDATED
    Incremental update to a content block.
    """

    type: Literal['content_block_delta']
    index: int
    delta: DeltaContent


class ContentBlockStopEvent(StrictModel):
    """
    content_block_stop SSE event.

    VALIDATION STATUS: VALIDATED
    Marks end of a content block.
    """

    type: Literal['content_block_stop']
    index: int


class MessageDeltaEvent(StrictModel):
    """
    message_delta SSE event.

    VALIDATION STATUS: VALIDATED
    Final update with stop_reason and usage.
    """

    type: Literal['message_delta']
    delta: MessageDeltaPayload
    usage: MessageDeltaUsage
    context_management: MessageDeltaContextManagement | None = None


class MessageStopEvent(StrictModel):
    """
    message_stop SSE event.

    VALIDATION STATUS: VALIDATED
    Marks end of streaming response.
    """

    type: Literal['message_stop']


class PingEvent(StrictModel):
    """
    ping SSE event.

    VALIDATION STATUS: VALIDATED
    Keepalive event interspersed in stream.
    """

    type: Literal['ping']


class StreamError(StrictModel):
    """
    Error payload in streaming error event.

    VALIDATION STATUS: INFERRED
    Based on Anthropic API error structure.
    """

    type: str  # e.g., "overloaded_error", "api_error"
    message: str


class ErrorEvent(StrictModel):
    """
    error SSE event.

    VALIDATION STATUS: INFERRED
    Expected for streaming errors.
    """

    type: Literal['error']
    error: StreamError


# ==============================================================================
# Main SSE Event Union
# ==============================================================================


SSEEvent = Annotated[
    MessageStartEvent
    | ContentBlockStartEvent
    | ContentBlockDeltaEvent
    | ContentBlockStopEvent
    | MessageDeltaEvent
    | MessageStopEvent
    | PingEvent
    | ErrorEvent,
    Discriminator('type'),
]
