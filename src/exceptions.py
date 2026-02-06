"""
Shared exceptions for claude-session-mcp.

Domain-specific exceptions used across services.

Exception Hierarchy:
    ClaudeSessionError (base)
    ├── SessionResolutionError (lookup/resolution failures)
    │   └── AmbiguousSessionError (prefix matches multiple sessions)
    └── SessionDeletionError (deletion policy violations)
        ├── NativeSessionDeletionError (native session without --force)
        └── RunningSessionDeletionError (running session without --terminate)
"""

from __future__ import annotations


class ClaudeSessionError(Exception):
    """Base exception for all claude-session-mcp errors."""


class SessionResolutionError(ClaudeSessionError):
    """Base exception for session lookup and resolution failures."""


class AmbiguousSessionError(SessionResolutionError):
    """Raised when a session ID prefix matches multiple sessions."""

    def __init__(self, prefix: str, matches: list[str]) -> None:
        self.prefix = prefix
        self.matches = matches
        matches_str = '\n  '.join(matches[:10])
        if len(matches) > 10:
            matches_str += f'\n  ... and {len(matches) - 10} more'
        super().__init__(
            f"Session ID prefix '{prefix}' is ambiguous. Matches {len(matches)} sessions:\n  {matches_str}\n\n"
            f'Please provide a more specific session ID prefix.'
        )


class SessionDeletionError(ClaudeSessionError):
    """Base exception for deletion policy violations."""


class NativeSessionDeletionError(SessionDeletionError):
    """Raised when attempting to delete a native session without --force."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(
            f'Session {session_id} is a native Claude session (UUIDv4). Use --force to delete native sessions.'
        )


class RunningSessionDeletionError(SessionDeletionError):
    """Raised when attempting to delete a running session without --terminate."""

    def __init__(self, session_id: str, pid: int) -> None:
        self.session_id = session_id
        self.pid = pid
        super().__init__(f'Session {session_id} is running (PID {pid}). Use --terminate to kill the process.')
