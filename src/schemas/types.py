"""
Shared type definitions for schemas.

Centralizes common type annotations used across session and API schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

import pydantic

# Pydantic-enhanced datetime for JSON serialization (allows string->datetime conversion)
JsonDatetime = Annotated[datetime, pydantic.Field(strict=False)]


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
    'claude-sonnet-4@20250514',
    # Short aliases (used in Task tool input.model)
    'haiku',
    'sonnet',
    'opus',
    # Special
    '<synthetic>',
]
