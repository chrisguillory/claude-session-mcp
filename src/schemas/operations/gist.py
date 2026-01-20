"""
Gist operation result models.

Models for GitHub Gist storage operations.
"""

from __future__ import annotations

from src.schemas.base import StrictModel


class GistArchiveResult(StrictModel):
    """Result of saving a session to GitHub Gist."""

    # Gist info
    gist_url: str  # https://gist.github.com/{user}/{id}
    gist_id: str  # The gist ID for future reference

    # Archive info (from ArchiveMetadata)
    session_id: str
    format: str  # Always 'json' for gist
    size_mb: float
    session_records: int  # Records in main {session_id}.jsonl file
    agent_records: int  # Records in agent-*.jsonl files
    file_count: int

    # Restore instruction
    restore_command: str  # "claude-session restore gist://{id}"
