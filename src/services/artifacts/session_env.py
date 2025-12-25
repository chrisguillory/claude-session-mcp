"""
Session-env validation for clone/restore operations.

Path pattern: ~/.claude/session-env/<session-id>/

Currently observed to be empty directories. This module provides
validation that fails fast if session-env contains data, ensuring
we don't silently lose data if Claude Code starts using this directory.

Future-proofing: When Claude Code starts using session-env, this module
will need to be updated to collect and restore its contents.
"""

from __future__ import annotations

from pathlib import Path

SESSION_ENV_DIR = Path.home() / '.claude' / 'session-env'


def validate_session_env_empty(session_id: str) -> None:
    """
    Validate that session-env directory is empty (or doesn't exist).

    This is a fail-fast check to ensure we don't silently skip
    session-env data if Claude Code starts using this directory.

    If the session-env directory for this session doesn't exist, that's
    fine - many sessions don't have one. If it exists but is empty,
    that's also fine.

    Args:
        session_id: Session ID to check

    Raises:
        NotImplementedError: If session-env directory contains files.
            Message includes the path and file count to help diagnose.
            This error indicates Claude Code has started using this
            directory and collection/restoration logic must be implemented.
    """
    session_env_path = SESSION_ENV_DIR / session_id

    if not session_env_path.exists():
        return

    # Check if directory contains any files (recursive)
    files = list(session_env_path.rglob('*'))
    file_count = sum(1 for f in files if f.is_file())

    if file_count > 0:
        raise NotImplementedError(
            f'Session-env directory contains {file_count} files: {session_env_path}\n'
            'Claude Code has started using session-env. This feature needs '
            'implementation for collection and restoration.'
        )


def create_session_env_dir(session_id: str) -> Path:
    """
    Create empty session-env directory for a new session.

    Creates the directory structure to match what Claude Code expects.
    Uses mkdir with exist_ok=True for idempotency.

    Args:
        session_id: New session ID

    Returns:
        Path to created (or existing) directory
    """
    session_env_path = SESSION_ENV_DIR / session_id
    session_env_path.mkdir(parents=True, exist_ok=True)
    return session_env_path
