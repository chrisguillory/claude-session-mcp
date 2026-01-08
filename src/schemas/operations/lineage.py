"""
Lineage operation schemas.

Models for session lineage tracking (parent-child relationships).
Extracted from services/lineage.py for reuse.
"""

from __future__ import annotations

from typing import Literal

import pydantic

from src.schemas.base import StrictModel
from src.schemas.types import JsonDatetime


class LineageEntry(StrictModel):
    """Lineage entry for a cloned/restored session.

    Self-contained model with consistent 'parent_*' prefixing for source references.

    Field ordering:
    - Identity (who)
    - Temporal (when)
    - Operation (how)
    - Parent context (from where)
    - Target context (to where)
    - Operational flags (details)
    """

    # Identity
    child_session_id: str  # Session ID created by clone/restore
    parent_session_id: str  # Source session that was cloned

    # Temporal
    cloned_at: JsonDatetime  # Timestamp of operation (UTC)

    # Operation
    method: Literal['clone', 'restore']  # How the session was created

    # Parent context (source)
    parent_project_path: str  # Project path where parent session exists
    parent_machine_id: str | None  # Machine where parent was created (from archive, None for clone)

    # Target context (destination)
    target_project_path: str  # Project path where clone was created
    target_machine_id: str  # Machine where clone/restore operation executed

    # Operational flags
    paths_translated: bool  # Whether project paths were translated
    archive_path: str | None  # Archive path (restore operations only)


class LineageResult(StrictModel):
    """API response for session lineage query.

    Separate from LineageEntry (storage model) to allow independent evolution
    of the API contract and storage format. Contains computed fields that are
    not persisted to storage.

    Field ordering matches LineageEntry plus computed fields at the end.
    """

    # Identity
    child_session_id: str
    parent_session_id: str

    # Temporal
    cloned_at: JsonDatetime

    # Operation
    method: Literal['clone', 'restore']

    # Parent context (source)
    parent_project_path: str
    parent_machine_id: str | None

    # Target context (destination)
    target_project_path: str
    target_machine_id: str

    # Operational flags
    paths_translated: bool
    archive_path: str | None

    # Computed (not in storage)
    is_cross_machine: bool | None  # True=cross-machine, False=same-machine, None=unknown (clones)


class LineageFile(StrictModel):
    """The lineage.json file structure.

    This model is NOT frozen to allow mutable sessions dict.
    """

    model_config = {'extra': 'forbid', 'strict': True, 'frozen': False}

    schema_version: str = '1.0'
    # Intentionally mutable - this model is frozen=False to allow dict mutation
    sessions: dict[str, LineageEntry] = pydantic.Field(default_factory=dict)  # check_schema_typing.py: mutable-type
