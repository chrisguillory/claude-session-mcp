"""
Restore operation schemas.

Models for session restoration results.
Extracted from services/restore.py for reuse.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from src.schemas.base import StrictModel


class RestoreResult(StrictModel):
    """Result of a session restore operation."""

    new_session_id: str
    original_session_id: str
    restored_at: datetime
    project_path: str
    files_restored: int
    records_restored: int
    paths_translated: bool
    main_session_file: str
    agent_files: Sequence[str]
