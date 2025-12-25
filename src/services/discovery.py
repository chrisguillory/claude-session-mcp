"""
Session discovery service - finds sessions across all Claude Code projects.

Provides utilities to discover sessions by ID, list all sessions, and decode
Claude's filesystem path encoding scheme.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.base_model import StrictModel


class SessionInfo(StrictModel):
    """Information about a discovered Claude Code session."""

    session_id: str
    project_path: Path


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
            SessionInfo with session_id and project_path, None if not found
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
        project_dir = session_file.parent

        # Decode project path from directory name
        project_path = self._decode_path(project_dir.name)

        return SessionInfo(
            session_id=session_id,
            project_path=project_path,
        )

    def _decode_path(self, encoded: str) -> Path:
        """
        Decode Claude's filesystem path encoding.

        Claude encodes paths by replacing '/' with '-':
        /Users/chris/project -> -Users-chris-project

        Args:
            encoded: Encoded path string

        Returns:
            Decoded Path object
        """
        # Remove leading hyphen and replace remaining hyphens with slashes
        if encoded.startswith('-'):
            decoded = encoded[1:].replace('-', '/')
            return Path(f'/{decoded}')
        else:
            # Shouldn't happen, but handle gracefully
            return Path(encoded.replace('-', '/'))
