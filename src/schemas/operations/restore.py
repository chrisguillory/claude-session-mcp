"""
Restore operation schemas.

Models for session restoration results.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from src.schemas.base import StrictModel


class RestoreResult(StrictModel):
    """Result of a session restore operation.

    Provides explicit breakdown of all restored artifacts, matching
    the level of detail in DeleteManifest/DeleteResult.
    """

    # Core identifiers
    new_session_id: str
    original_session_id: str
    restored_at: datetime
    project_path: str

    # Operation mode
    was_in_place: bool

    # Session files (JSONL)
    main_session_file: str
    agent_files: Sequence[str]

    # Record counts (broken down)
    main_records_restored: int
    agent_records_restored: int

    # Auxiliary artifact counts
    plan_files_restored: int
    tool_results_restored: int
    todos_restored: int
    tasks_restored: int

    # Transformation info
    paths_translated: bool
    slug_mappings_applied: int
    agent_id_mappings_applied: int

    # Source metadata (from archive)
    source_machine_id: str | None
    custom_title: str | None
