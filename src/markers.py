"""
Field markers for special serialization handling.

PathMarker enables runtime discovery of filesystem path fields for cross-machine
session transfer. Uses Annotated pattern (Pydantic v2 recommended approach).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated


@dataclass(frozen=True)
class PathMarker:
    """Mark field as filesystem path for translation."""

    pass


# Type aliases for path fields
type PathField = Annotated[str, PathMarker()]
type PathListField = Annotated[list[str], PathMarker()]
