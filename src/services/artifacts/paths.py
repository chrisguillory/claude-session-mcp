"""
Path utilities for session artifact handling.

Provides functions to extract accurate paths from session records,
avoiding the lossy path encoding used by Claude Code's directory names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from src.schemas.session import (
    ErrorAgentAssistantRecord,
    ErrorMainAssistantRecord,
    NormalAgentAssistantRecord,
    NormalMainAssistantRecord,
    SessionRecord,
    UserRecord,
)

__all__ = ['MissingCwdError', 'extract_source_project_path']


class MissingCwdError(Exception):
    """Raised when no cwd field is found in session records."""

    pass


def extract_source_project_path(
    files_data: Mapping[str, Sequence[SessionRecord]],
) -> Path:
    """
    Extract the source project path from session records.

    Claude Code encodes working directory paths for filesystem safety:
    - `/` → `-`
    - `.` → `-`
    - ` ` → `-`
    - `~` → `-`

    This encoding is **lossy** - you cannot reliably decode it back.
    For example, `-claude-session-mcp` could be:
    - `/claude/session/mcp` (three directories)
    - `/claude-session-mcp` (one directory with dashes)
    - `/claude.session.mcp` (one directory with dots)

    Instead of decoding, this function extracts the actual path from the
    'cwd' field of session records, which contains the true filesystem path.

    Args:
        files_data: Mapping of filename -> sequence of SessionRecord

    Returns:
        Source project path extracted from record cwd fields

    Raises:
        MissingCwdError: If no cwd field found in any record (corrupt session)
    """
    for records in files_data.values():
        for record in records:
            if (
                isinstance(
                    record,
                    (
                        UserRecord,
                        NormalMainAssistantRecord,
                        ErrorMainAssistantRecord,
                        NormalAgentAssistantRecord,
                        ErrorAgentAssistantRecord,
                    ),
                )
                and record.cwd
            ):
                return Path(record.cwd)

    raise MissingCwdError(
        'No cwd field found in session records. '
        'Session data may be corrupted or from an incompatible Claude Code version.'
    )
