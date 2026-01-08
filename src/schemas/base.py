"""
Shared Pydantic base model for strict validation.

All Pydantic schema models in the application should inherit from StrictModel.
This module re-exports BaseStrictModel as StrictModel for the operations/ package.
"""

from __future__ import annotations

from src.schemas.types import BaseStrictModel


class StrictModel(BaseStrictModel):
    """Operations-layer strict model.

    Inherits from BaseStrictModel (extra='forbid', strict=True, frozen=True).
    Used by src/schemas/operations/ package.
    """

    pass
