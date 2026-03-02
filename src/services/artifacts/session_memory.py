"""
Session memory artifact handling.

Collects and writes per-session memory summaries stored at
<project-dir>/<session-id>/session-memory/summary.md.
"""

from __future__ import annotations

from pathlib import Path

SUMMARY_FILENAME = 'summary.md'
SESSION_MEMORY_DIRNAME = 'session-memory'


def collect_session_memory(session_dir: Path, session_id: str) -> str | None:
    """Read session-memory/summary.md if it exists.

    Args:
        session_dir: The project-encoded directory (e.g., ~/.claude/projects/<enc>/)
        session_id: The session UUID

    Returns:
        The summary content, or None if no session memory exists.
    """
    path = session_dir / session_id / SESSION_MEMORY_DIRNAME / SUMMARY_FILENAME
    if path.exists():
        return path.read_text()
    return None


def write_session_memory(session_dir: Path, session_id: str, content: str) -> Path:
    """Write session-memory/summary.md to target directory.

    Args:
        session_dir: The target project-encoded directory
        session_id: The session UUID
        content: The summary markdown content

    Returns:
        Path to the written file.
    """
    path = session_dir / session_id / SESSION_MEMORY_DIRNAME / SUMMARY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path
