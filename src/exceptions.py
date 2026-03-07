"""
Shared exceptions for claude-session-mcp.

Domain-specific exceptions used across services.

Exception Hierarchy:
    ClaudeSessionError (base)
    ├── SessionResolutionError (lookup/resolution failures)
    │   └── AmbiguousSessionError (prefix matches multiple sessions)
    ├── SessionDeletionError (deletion policy violations)
    │   ├── NativeSessionDeletionError (native session without --force)
    │   └── RunningSessionDeletionError (running session without --terminate)
    └── SessionMoveError (move policy violations)
        ├── SameProjectMoveError (session already in target project)
        ├── NativeSessionMoveError (native session without --force)
        └── RunningSessionMoveError (running session without --terminate)
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
            f'Provide a more specific session ID prefix, or use --project to target a specific project.'
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


class SessionMoveError(ClaudeSessionError):
    """Base exception for move policy violations."""


class SameProjectMoveError(SessionMoveError):
    """Raised when attempting to move a session to its current project."""

    def __init__(self, session_id: str, project_path: str) -> None:
        self.session_id = session_id
        self.project_path = project_path
        super().__init__(f'Session {session_id} is already in project {project_path}.')


class NativeSessionMoveError(SessionMoveError):
    """Raised when attempting to move a native session without --force."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(
            f'Session {session_id} is a native Claude session (UUIDv4). Use --force to move native sessions.'
        )


class RunningSessionMoveError(SessionMoveError):
    """Raised when attempting to move a running session without --terminate."""

    def __init__(self, session_id: str, pid: int) -> None:
        self.session_id = session_id
        self.pid = pid
        super().__init__(f'Session {session_id} is running (PID {pid}). Use --terminate to kill the process.')
