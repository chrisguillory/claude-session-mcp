"""
Session discovery service - finds sessions across all Claude Code projects.

Provides utilities to discover sessions by ID and list all available sessions.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.schemas.operations.discovery import SessionInfo


class SessionDiscoveryService:
    """
    Service for discovering Claude Code sessions across all projects.

    Searches ~/.claude/projects/ for session files and provides utilities
    to find sessions by ID or list all available sessions.
    """

    def __init__(self) -> None:
        """Initialize discovery service."""
        self.claude_sessions_dir = Path.home() / '.claude' / 'projects'

    async def find_session_by_id(self, session_id: str) -> SessionInfo | None:
        """
        Find a session by ID across all projects using rg.

        Args:
            session_id: Session ID to find

        Returns:
            SessionInfo with session_id and session_folder, None if not found
        """
        if not self.claude_sessions_dir.exists():
            return None

        # Use rg to quickly find the session file
        result = subprocess.run(
            ['rg', '--files', '--glob', f'{session_id}.jsonl', str(self.claude_sessions_dir)],
            capture_output=True,
            text=True,
        )

        if not result.stdout.strip():
            return None

        # Get first match (should only be one)
        session_file = Path(result.stdout.strip().split('\n')[0])
        session_folder = session_file.parent

        return SessionInfo(
            session_id=session_id,
            session_folder=session_folder,
        )
