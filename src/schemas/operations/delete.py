"""
Delete operation schemas.

Models for session deletion with safety features.
Extracted from services/delete.py for reuse.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from src.schemas.base import StrictModel


class ArtifactFile(StrictModel):
    """Single file in delete manifest."""

    path: str
    size_bytes: int
    artifact_type: Literal[
        'session_main',
        'session_agent',
        'plan_file',
        'tool_result',
        'todo_file',
        'session_env',
    ]


class DeleteManifest(StrictModel):
    """Discovery result for a session to delete."""

    session_id: str
    is_native: bool  # True if native session (UUIDv4), requires --force
    created_at: datetime | None  # Extracted from UUIDv7 if applicable

    files: Sequence[ArtifactFile]
    total_size_bytes: int

    session_main_file: str | None
    agent_files: Sequence[str]
    plan_files: Sequence[str]
    tool_result_files: Sequence[str]
    todo_files: Sequence[str]
    session_env_dir: str | None


class DeleteResult(StrictModel):
    """Execution result."""

    session_id: str
    was_dry_run: bool
    success: bool
    error_message: str | None

    backup_path: str | None  # Path to backup archive (if created)
    files_deleted: int
    size_freed_bytes: int
    deleted_files: Sequence[str]
    failed_deletions: Sequence[str]

    duration_ms: float
    deleted_at: datetime
