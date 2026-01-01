"""
Shared Pydantic base model for strict validation.

All Pydantic models in the application should inherit from StrictModel.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base model with strict validation settings."""

    model_config = ConfigDict(
        extra='forbid',  # Raise error on unexpected fields
        strict=True,  # Strict type validation
        frozen=True,  # Immutable (cannot modify after creation)
    )
