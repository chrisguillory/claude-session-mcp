"""Verification file for check_mutable_annotations.py.

This file contains test cases to verify the mutable type checker works correctly.
It is intentionally committed to demonstrate the checker's behavior.

VERIFIED BEHAVIOR (run ./scripts/check_mutable_annotations.py to confirm):

Should PASS (no violations):
- GoodModel: Uses Mapping, Sequence, tuple, frozenset
- Method return types: list return is fine (constructing new)
- Function body locals: list/dict inside methods is fine
- RegularClass: Not a model, not checked
- MyTypedDict: TypedDict is exempt
- GoodModelWithNoqa: Has # noqa: mutable-type

Should FAIL (6 violations):
- BadModelWithDict.headers: dict → Mapping
- BadModelWithList.items: list → Sequence
- BadModelWithSet.tags: set → Set (from collections.abc)
- BadModelWithNestedMutable.data: Mapping[str, list[int]] - nested list caught
- BadModelWithMethodParam.process(items: list[str]) - param caught
- BadBaseModel.config: dict - BaseModel subclass caught

TODO: Delete this file after demonstrating the checker.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

from pydantic import BaseModel

from src.schemas.base import StrictModel

# =============================================================================
# SHOULD PASS - These should NOT trigger violations
# =============================================================================


# noinspection PyMethodMayBeStatic
class GoodModel(StrictModel):
    """Model using immutable types - should pass."""

    # Good: Using Mapping instead of dict
    headers: Mapping[str, str]

    # Good: Using Sequence instead of list
    items: Sequence[str]

    # Good: Using tuple
    coordinates: tuple[float, float]

    # Good: Using frozenset
    tags: frozenset[str]

    def process(self, data: Mapping[str, int]) -> list[str]:
        """Method with Mapping param and list return type.

        Return types are NOT checked - constructing new collections is fine.
        """
        # Function body locals are NOT checked
        result: list[str] = []
        temp_dict: dict[str, int] = {}  # noqa: F841 - intentionally unused for verification
        for k, v in data.items():
            result.append(f'{k}={v}')
        return result


class RegularClass:
    """Non-model class - should NOT be checked at all."""

    # This would be a violation in a model, but we don't check regular classes
    items: list[str]
    data: dict[str, int]

    def method(self, items: list[str]) -> None:
        pass


class MyTypedDict(TypedDict):
    """TypedDict is exempt - interface definitions often need list/dict."""

    items: list[str]
    data: dict[str, int]


class GoodModelWithNoqa(StrictModel):
    """Model with explicit noqa for intentionally mutable field."""

    # Intentionally mutable - this model needs to mutate this dict
    sessions: dict[str, str]  # noqa: mutable-type


# =============================================================================
# SHOULD FAIL - These SHOULD trigger violations
# =============================================================================


class BadModelWithDict(StrictModel):
    """Model using dict - should FAIL."""

    # BAD: Using dict instead of Mapping
    headers: dict[str, str]


class BadModelWithList(StrictModel):
    """Model using list - should FAIL."""

    # BAD: Using list instead of Sequence
    items: list[str]


class BadModelWithSet(StrictModel):
    """Model using set - should FAIL."""

    # BAD: Using set instead of frozenset
    tags: set[str]


class BadModelWithNestedMutable(StrictModel):
    """Model with mutable type nested in Mapping - should FAIL."""

    # BAD: Mapping is fine, but list inside is not
    data: Mapping[str, list[int]]


# noinspection PyMethodMayBeStatic
class BadModelWithMethodParam(StrictModel):
    """Model with mutable type in method parameter - should FAIL."""

    name: str

    def process(self, items: list[str]) -> Sequence[str]:
        """BAD: Parameter uses list instead of Sequence."""
        return items


class BadBaseModel(BaseModel):
    """Using BaseModel directly - should also be checked."""

    # BAD: Using dict
    config: dict[str, str]
