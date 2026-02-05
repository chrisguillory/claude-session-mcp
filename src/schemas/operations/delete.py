"""
Delete operation schemas.

Models for session deletion with explicit file enumeration and validation.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from src.schemas.base import StrictModel
from src.schemas.types import PathStr


class ArtifactFile(StrictModel):
    """Single file in delete manifest."""

    path: PathStr
    size_bytes: int
    artifact_type: Literal[
        'session_main',  # ~/.claude/projects/<enc>/{session_id}.jsonl
        'session_agent',  # Flat: .../agent-{id}.jsonl | Nested (2.1.2+): .../{sid}/subagents/agent-{id}.jsonl
        'plan_file',  # ~/.claude/plans/{slug}.md
        'tool_result',  # ~/.claude/projects/<enc>/{sid}/tool-results/{tool_use_id}.txt
        'todo_file',  # ~/.claude/todos/{session_id}-agent-{agent_id}.json
        'task_file',  # ~/.claude/tasks/{session_id}/{task_id}.json
    ]


class DeleteManifest(StrictModel):
    """Discovery result for a session to delete.

    All files are enumerated explicitly - no directories in the files list.
    Directories are tracked separately for cleanup after files are deleted.
    Unexpected files cause validation failure before any deletion occurs.
    """

    session_id: str
    is_native: bool  # True if native session (UUIDv4), requires --force
    created_at: datetime | None  # Extracted from UUIDv7 if applicable

    files: Sequence[ArtifactFile]
    total_size_bytes: int

    # Per-type file paths
    session_main_file: PathStr
    agent_files: Sequence[PathStr]
    plan_files: Sequence[PathStr]
    tool_result_files: Sequence[PathStr]
    todo_files: Sequence[PathStr]
    task_files: Sequence[PathStr]

    # Directories to clean up after files deleted (sorted deepest-first)
    directories_to_cleanup: Sequence[PathStr]

    # Unexpected files found during discovery (causes validation failure)
    unexpected_files: Sequence[PathStr]


class DeleteResult(StrictModel):
    """Execution result."""

    session_id: str
    was_dry_run: bool
    success: bool
    error_message: str | None

    backup_path: PathStr | None  # Path to backup archive (if created)
    files_deleted: int  # Total files deleted
    size_freed_bytes: int
    deleted_files: Sequence[PathStr]

    # Per-artifact-type breakdown
    session_files_deleted: int  # main + agents
    plan_files_deleted: int
    tool_results_deleted: int
    todos_deleted: int
    tasks_deleted: int

    # Directories that were cleaned up
    directories_removed: Sequence[PathStr]

    duration_ms: float
    deleted_at: datetime
