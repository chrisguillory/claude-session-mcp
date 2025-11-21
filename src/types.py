"""
Shared type definitions for the claude-session-mcp package.

Centralizes common type annotations used across multiple modules.
"""

from datetime import datetime
from typing import Annotated

import pydantic

# Pydantic-enhanced datetime for JSON serialization (allows string→datetime conversion)
JsonDatetime = Annotated[datetime, pydantic.Field(strict=False)]
