"""
Claude Code version detection utilities.

Provides functions to extract Claude Code version from:
1. Running process (via PID and executable path)
2. Session JSONL records (version field in records)

The process-based approach is preferred as it reflects the CURRENT version
performing an operation, not historical versions from when records were written.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import packaging.version
import psutil

from src.schemas.session import (
    ApiErrorSystemRecord,
    CompactBoundarySystemRecord,
    ErrorAgentAssistantRecord,
    ErrorMainAssistantRecord,
    InformationalSystemRecord,
    LocalCommandSystemRecord,
    NormalAgentAssistantRecord,
    NormalMainAssistantRecord,
    SessionRecord,
    TurnDurationSystemRecord,
    UserRecord,
)

# All record types that have a version field
VERSION_RECORD_TYPES = (
    UserRecord,
    NormalMainAssistantRecord,
    ErrorMainAssistantRecord,
    NormalAgentAssistantRecord,
    ErrorAgentAssistantRecord,
    LocalCommandSystemRecord,
    CompactBoundarySystemRecord,
    ApiErrorSystemRecord,
    InformationalSystemRecord,
    TurnDurationSystemRecord,
)


def get_version(
    claude_pid: int | None = None,
    records: Sequence[SessionRecord] | None = None,
) -> str | None:
    """Get Claude Code version using best available method.

    Fallback chain:
    1. Process-based detection (if PID provided and process exists)
    2. Session records (if records provided and contain version)
    3. None (if version cannot be determined)

    Args:
        claude_pid: Optional PID of running Claude process
        records: Optional sequence of session records

    Returns:
        Version string, or None if version cannot be determined
    """
    # Try process-based first (most accurate for current operations)
    if claude_pid is not None:
        version = get_version_from_process(claude_pid)
        if version:
            return version

    # Fall back to records
    if records is not None:
        version = get_version_from_records(records)
        if version:
            return version

    return None


def get_version_from_process(claude_pid: int) -> str | None:
    """Extract Claude Code version from the running process's executable path.

    Uses psutil to get the actual executable path of the Claude process,
    which contains the version (e.g., ~/.local/share/claude/versions/2.1.12).

    This is more accurate than extracting from session records because it reflects
    the CURRENT version of the tool performing the operation, not historical versions
    from when records were written.

    Args:
        claude_pid: PID of the running Claude process

    Returns:
        Version string (e.g., "2.1.12"), or None if version cannot be determined
    """
    try:
        exe_path = Path(psutil.Process(claude_pid).exe())
        version = packaging.version.Version(exe_path.name)
        return str(version)
    except (psutil.NoSuchProcess, psutil.AccessDenied, packaging.version.InvalidVersion):
        return None


def get_version_from_records(records: Sequence[SessionRecord]) -> str | None:
    """Extract Claude Code version from session records.

    Searches through records to find the first one with a version field.
    This is used as a fallback when process-based detection is not available
    (e.g., when archiving a session that is no longer running).

    Args:
        records: Sequence of parsed session records

    Returns:
        Version string (e.g., "2.0.76"), or None if no version found
    """
    for record in records:
        if isinstance(record, VERSION_RECORD_TYPES) and record.version:
            return record.version
    return None
