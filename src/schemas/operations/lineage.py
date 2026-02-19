"""
Lineage operation schemas.

Models for session lineage tracking (parent-child relationships).
Extracted from services/lineage.py for reuse.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


class LineageTree(StrictModel):
    """Complete lineage tree from root ancestor to all leaf descendants.

    Built by LineageService.get_full_tree(). Given any session ID in a lineage
    graph, walks up to the root and down to all leaves.

    This is the sole return type for lineage queries — it subsumes the
    per-entry view. Access tree.nodes[tree.queried_session_id] for the
    queried node's full metadata.
    """

    root_session_id: str  # Root ancestor (may be a native session with no lineage entry)
    queried_session_id: str  # The session ID that was originally queried (for highlighting)
    nodes: Mapping[str, LineageTreeNode]  # All nodes keyed by session ID


class LineageTreeNode(StrictModel):
    """Single node in a lineage tree.

    All operation metadata fields are None for root/native sessions
    (they have no lineage entry of their own).
    """

    # Tree structure
    session_id: str
    parent_id: str | None  # None for root
    children: Sequence[str]  # Ordered child session IDs (by cloned_at, then session ID)
    depth: int  # 0 for root, increments per level

    # Operation metadata (from LineageEntry; all None for root)
    cloned_at: JsonDatetime | None
    method: Literal['clone', 'restore'] | None
    parent_project_path: str | None
    parent_machine_id: str | None
    target_project_path: str | None
    target_machine_id: str | None
    paths_translated: bool | None
    archive_path: str | None

    # Computed
    is_cross_machine: bool | None  # True=cross-machine, False=same, None=unknown/root

    # Display (resolved from session files, None if file not found locally)
    custom_title: str | None


class LineageFile(StrictModel):
    """The lineage.json file structure.

    This model is NOT frozen to allow mutable sessions dict.
    """

    model_config = {'extra': 'forbid', 'strict': True, 'frozen': False}

    schema_version: str = '1.0'
    # Intentionally mutable - this model is frozen=False to allow dict mutation
    sessions: dict[str, LineageEntry] = pydantic.Field(default_factory=dict)  # check_schema_typing.py: mutable-type
