"""
Response schemas for Claude Code internal API.

These model the response payloads Claude Code receives from the API.
Validated against mitmproxy captures of actual Claude Code traffic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

import anthropic.types

from src.schemas import session
from src.schemas.cc_internal_api.base import FromSdk, FromSession, PermissiveModel
from src.schemas.cc_internal_api.common import ApiUsage
from src.schemas.types import ModelId

# ==============================================================================
# Stop Reason
# ==============================================================================


# Note: 'max_tokens' observed in API traffic but NOT in session schemas.
# Session schema has: tool_use, stop_sequence, end_turn, refusal, model_context_window_exceeded
# API traffic showed: max_tokens (from quota check response)
StopReason = Literal[
    'end_turn',
    'max_tokens',  # Observed in API, not in session files
    'stop_sequence',
    'tool_use',
    'refusal',
    'model_context_window_exceeded',
]


# ==============================================================================
# Response Content Blocks
# ==============================================================================


class ResponseTextContent(PermissiveModel):
    """
    Text content block in API response.

    VALIDATION STATUS: VALIDATED (2026-01-02)
    Observed in response.content[].

    Note: Unlike request TextContentBlock, no cache_control field in response.
    """

    type: Literal['text']
    text: str


class ThinkingContent(PermissiveModel):
    """
    Thinking content block in API response.

    VALIDATION STATUS: INFERRED from session schemas
    Not directly observed in captured traffic (quota check response was minimal).
    """

    type: Literal['thinking']
    thinking: Annotated[
        str,
        FromSession(session.models.ThinkingContent, 'thinking', status='inferred'),
    ]
    signature: Annotated[
        str | None,
        FromSession(session.models.ThinkingContent, 'signature', status='inferred'),
    ] = None


class ToolUseContent(PermissiveModel):
    """
    Tool use content block in API response.

    VALIDATION STATUS: INFERRED from session schemas
    Not directly observed in captured traffic (quota check response had no tool use).
    """

    type: Literal['tool_use']
    id: Annotated[
        str,
        FromSession(session.models.ToolUseContent, 'id', status='inferred'),
    ]
    name: Annotated[
        str,
        FromSession(session.models.ToolUseContent, 'name', status='inferred'),
    ]
    input: Annotated[
        Mapping[str, Any],
        FromSession(session.models.ToolUseContent, 'input', status='inferred'),
    ]


# Union of all response content types
ResponseContent = ResponseTextContent | ThinkingContent | ToolUseContent


# ==============================================================================
# Context Management (Response)
# ==============================================================================


class ResponseContextManagement(PermissiveModel):
    """
    Context management in API response.

    VALIDATION STATUS: VALIDATED (2026-01-02)
    Observed in response.context_management.

    Note: Response has 'applied_edits' vs request has 'edits'.
    """

    applied_edits: Sequence[Mapping[str, Any]]  # Observed as empty array


# ==============================================================================
# Messages Response
# ==============================================================================


class MessagesResponse(PermissiveModel):
    """
    Complete response payload from /v1/messages.

    VALIDATION STATUS: VALIDATED (2026-01-02)
    Observed in non-streaming quota check response.

    CORRESPONDING SESSION TYPE: session.models.Message (nested in AssistantRecord)
    Note: Session Message has additional fields like costUSD not in API response.

    CORRESPONDING SDK TYPE: anthropic.types.Message
    """

    model: Annotated[
        ModelId,
        FromSession(session.models.Message, 'model', status='validated'),
        FromSdk(anthropic.types.Message, 'model'),
    ]

    id: Annotated[
        str,
        FromSession(session.models.Message, 'id', status='validated'),
        FromSdk(anthropic.types.Message, 'id'),
    ]

    type: Annotated[
        Literal['message'],
        FromSession(session.models.Message, 'type', status='validated'),
        FromSdk(anthropic.types.Message, 'type'),
    ]

    role: Annotated[
        Literal['assistant'],
        FromSession(session.models.Message, 'role', status='validated'),
        FromSdk(anthropic.types.Message, 'role'),
    ]

    content: Annotated[
        Sequence[ResponseContent],
        FromSdk(anthropic.types.Message, 'content'),
    ]

    stop_reason: Annotated[
        StopReason | None,
        FromSession(session.models.Message, 'stop_reason', status='validated'),
        FromSdk(anthropic.types.Message, 'stop_reason'),
    ]

    stop_sequence: Annotated[
        str | None,
        FromSession(session.models.Message, 'stop_sequence', status='validated'),
        FromSdk(anthropic.types.Message, 'stop_sequence'),
    ]

    usage: Annotated[
        ApiUsage,
        FromSession(session.models.Message, 'usage', status='validated'),
        FromSdk(anthropic.types.Message, 'usage'),
    ]

    context_management: Annotated[
        ResponseContextManagement | None,
        FromSession(session.models.Message, 'context_management', status='validated'),
    ] = None
