"""
Field markers for special serialization handling.

PathMarker enables runtime discovery of filesystem path fields for cross-machine
session transfer. Uses Annotated pattern (Pydantic v2 recommended approach).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from src.schemas.types import PathStr


@dataclass(frozen=True)
class PathMarker:
    """Mark field as filesystem path for translation."""

    pass


type PathField = Annotated[PathStr, PathMarker()]
"""A path that needs translation during cross-machine restore."""

type PathListField = Annotated[list[PathStr], PathMarker()]
"""A list of paths that need translation during cross-machine restore."""
