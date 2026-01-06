"""
Common types for Claude Code internal API schemas.

These types are shared between request and response schemas.
Validated against mitmproxy captures of actual Claude Code traffic.
"""

from __future__ import annotations

from typing import Annotated, Literal

import anthropic.types

from src.schemas import session
from src.schemas.cc_internal_api.base import FromSdk, FromSession, StrictModel

# ==============================================================================
# Cache Control (API-only, not persisted to session files)
# ==============================================================================


class CacheControl(StrictModel):
    """
    Cache control directive for API requests.

    VALIDATION STATUS: VALIDATED
    Observed on system blocks and user message blocks.

    No ttl field was observed in captured traffic - CC uses default TTL.
    """

    type: Literal['ephemeral']


# ==============================================================================
# Cache Creation (nested in Usage)
# ==============================================================================


class ApiCacheCreation(StrictModel):
    """
    Detailed cache creation breakdown in API responses.

    VALIDATION STATUS: VALIDATED
    Observed in response.usage.cache_creation.

    CORRESPONDING SESSION TYPE: session.models.CacheCreation
    """

    ephemeral_5m_input_tokens: Annotated[
        int,
        FromSession(session.models.CacheCreation, 'ephemeral_5m_input_tokens', status='validated'),
    ]

    ephemeral_1h_input_tokens: Annotated[
        int,
        FromSession(session.models.CacheCreation, 'ephemeral_1h_input_tokens', status='validated'),
    ]


# ==============================================================================
# Token Usage
# ==============================================================================


class ApiUsage(StrictModel):
    """
    Token usage in API responses.

    VALIDATION STATUS: VALIDATED
    Observed in response.usage field.

    CORRESPONDING TYPES:
    - Session: session.models.TokenUsage
    - SDK: anthropic.types.Usage
    """

    input_tokens: Annotated[
        int,
        FromSession(session.models.TokenUsage, 'input_tokens', status='validated'),
        FromSdk(anthropic.types.Usage, 'input_tokens'),
    ]

    output_tokens: Annotated[
        int,
        FromSession(session.models.TokenUsage, 'output_tokens', status='validated'),
        FromSdk(anthropic.types.Usage, 'output_tokens'),
    ]

    cache_creation_input_tokens: Annotated[
        int,
        FromSession(session.models.TokenUsage, 'cache_creation_input_tokens', status='validated'),
        FromSdk(anthropic.types.Usage, 'cache_creation_input_tokens'),
    ]

    cache_read_input_tokens: Annotated[
        int,
        FromSession(session.models.TokenUsage, 'cache_read_input_tokens', status='validated'),
        FromSdk(anthropic.types.Usage, 'cache_read_input_tokens'),
    ]

    cache_creation: Annotated[
        ApiCacheCreation,
        FromSession(session.models.TokenUsage, 'cache_creation', status='validated'),
    ]

    service_tier: Annotated[
        Literal['standard'],
        FromSession(session.models.TokenUsage, 'service_tier', status='validated'),
    ]
