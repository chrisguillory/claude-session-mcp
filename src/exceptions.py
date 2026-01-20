"""
Shared exceptions for claude-session-mcp.

Domain-specific exceptions used across services.
"""

from __future__ import annotations


class AmbiguousSessionError(Exception):
    """Raised when a session ID prefix matches multiple sessions."""

    def __init__(self, prefix: str, matches: list[str]) -> None:
        self.prefix = prefix
        self.matches = matches
        matches_str = '\n  '.join(matches[:10])
        if len(matches) > 10:
            matches_str += f'\n  ... and {len(matches) - 10} more'
        super().__init__(
            f"Session ID prefix '{prefix}' is ambiguous. Matches {len(matches)} sessions:\n  {matches_str}"
        )
