"""
Session discovery service - finds sessions across all Claude Code projects.

Provides utilities to discover sessions by ID and list all available sessions.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.exceptions import AmbiguousSessionError
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

    async def find_session_by_id(self, session_id_or_prefix: str) -> SessionInfo | None:
        """
        Find a session by ID or prefix across all projects using rg.

        Supports both full session IDs and unique prefixes (e.g., 'fd0fe7fa').

        Args:
            session_id_or_prefix: Full session ID or unique prefix

        Returns:
            SessionInfo with session_id and session_folder, None if not found

        Raises:
            AmbiguousSessionError: If prefix matches multiple sessions
        """
        if not self.claude_sessions_dir.exists():
            return None

        # Use rg to find session files matching the ID/prefix
        # Use wildcard pattern to support prefix matching
        result = subprocess.run(
            ['rg', '--files', '--glob', f'{session_id_or_prefix}*.jsonl', str(self.claude_sessions_dir)],
            capture_output=True,
            text=True,
        )

        if not result.stdout.strip():
            return None

        # Parse matches
        matches = [Path(p) for p in result.stdout.strip().split('\n') if p]

        # Filter out agent files (we only want main session files)
        session_files = [m for m in matches if not m.name.startswith('agent-')]

        if not session_files:
            return None

        if len(session_files) > 1:
            # Ambiguous prefix - multiple sessions match
            matching_ids = [f.stem for f in session_files]
            raise AmbiguousSessionError(session_id_or_prefix, matching_ids)

        # Single match - extract full session ID from filename
        session_file = session_files[0]
        full_session_id = session_file.stem  # filename without .jsonl
        session_folder = session_file.parent

        return SessionInfo(
            session_id=full_session_id,
            session_folder=session_folder,
        )
