"""
Session info service - retrieves comprehensive context about sessions.

Aggregates data from multiple sources:
- Session discovery (session files in ~/.claude/projects/)
- Claude-workspace tracking (~/.claude-workspace/sessions.json)
- Lineage tracking (~/.claude-session-mcp/lineage.json)
- MCP server state (for current session context)
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pydantic

from src.schemas.claude_workspace import Session, SessionDatabase
from src.schemas.operations.context import SessionContext
from src.schemas.operations.discovery import SessionInfo
from src.services.artifacts import extract_custom_title_from_file
from src.services.clone import AmbiguousSessionError
from src.services.delete import get_restoration_timestamp, is_native_session
from src.services.discovery import SessionDiscoveryService
from src.services.lineage import LineageService, get_machine_id

# Claude workspace sessions.json location
CLAUDE_WORKSPACE_SESSIONS = Path.home() / '.claude-workspace' / 'sessions.json'

# Claude debug files location
CLAUDE_DEBUG_DIR = Path.home() / '.claude' / 'debug'


@dataclass
class CurrentSessionContext:
    """
    Context provided by MCP server for the current session.

    These fields are only available when querying the current session
    via the MCP server, not when querying other sessions via CLI.
    """

    session_id: str
    project_path: Path
    claude_pid: int
    temp_dir: str


class SessionInfoService:
    """
    Service for retrieving comprehensive session context.

    Aggregates data from:
    - Session files (~/.claude/projects/)
    - Claude-workspace sessions.json (if available)
    - Lineage tracking (~/.claude-session-mcp/lineage.json)
    - Current session MCP state (if provided)
    """

    def __init__(self) -> None:
        """Initialize the service."""
        self.discovery = SessionDiscoveryService()
        self.lineage = LineageService()

    async def get_info(
        self,
        session_id: str,
        current_context: CurrentSessionContext | None = None,
    ) -> SessionContext:
        """
        Get comprehensive context for a session.

        Args:
            session_id: Session ID (full or prefix) to look up
            current_context: MCP server context for current session (optional)

        Returns:
            SessionContext with all available information

        Raises:
            FileNotFoundError: If session cannot be found
            AmbiguousSessionError: If prefix matches multiple sessions
        """
        # Resolve session ID (supports both full ID and prefix)
        session_info = await self._resolve_session(session_id)

        full_session_id = session_info.session_id
        session_folder = session_info.session_folder

        # Construct paths
        session_file = session_folder / f'{full_session_id}.jsonl'
        debug_file = CLAUDE_DEBUG_DIR / f'{full_session_id}.txt'

        # Get project path - need to extract from session file or use current context
        project_path = await self._get_project_path(session_file, full_session_id, current_context)

        # Load claude-workspace session data if available
        workspace_session = self._load_workspace_session(full_session_id)

        # Check lineage
        has_lineage = self.lineage.get_entry(full_session_id) is not None

        # Compute characteristics
        native = is_native_session(full_session_id)
        created_at = None if native else get_restoration_timestamp(full_session_id)

        # Extract custom title from session file
        custom_title = extract_custom_title_from_file(session_file)

        # Determine if this is the current session
        is_current = current_context is not None and current_context.session_id == full_session_id

        # Environment fields:
        # - temp_dir: Only available for current session (MCP server runtime)
        # - claude_pid: Current session has live PID; other sessions have historical PID from sessions.json
        # - machine_id: If session is in sessions.json, it's on this machine, so we can compute it
        if is_current and current_context is not None:
            claude_pid: int | None = current_context.claude_pid
            temp_dir: str | None = current_context.temp_dir
            machine_id: str | None = get_machine_id()
        else:
            # For non-current sessions, get historical PID from sessions.json if available
            claude_pid = workspace_session.metadata.claude_pid if workspace_session else None
            temp_dir = None  # Only available for current session
            # If session is in sessions.json, it was created on this machine
            machine_id = get_machine_id() if workspace_session else None

        return SessionContext(
            # Identity
            session_id=full_session_id,
            custom_title=custom_title,
            # Temporal
            started_at=workspace_session.metadata.started_at if workspace_session else None,
            ended_at=workspace_session.metadata.ended_at if workspace_session else None,
            created_at=created_at,
            # Paths
            project_path=str(project_path),
            session_file=str(session_file),
            debug_file=str(debug_file),
            # Environment
            machine_id=machine_id,
            claude_pid=claude_pid,
            claude_version=workspace_session.metadata.claude_version if workspace_session else None,
            temp_dir=temp_dir,
            # Origin
            source=workspace_session.source if workspace_session else 'unknown',
            state=workspace_session.state if workspace_session else 'unknown',
            parent_id=workspace_session.metadata.parent_id if workspace_session else None,
            # Characteristics
            is_native=native,
            has_lineage=has_lineage,
        )

    async def _get_project_path(
        self,
        session_file: Path,
        session_id: str,
        current_context: CurrentSessionContext | None,
    ) -> Path:
        """
        Get project path for a session.

        Priority:
        1. Current context (if matching session)
        2. Extract from first record's cwd field
        3. Fall back to encoded folder name (lossy)
        """
        # Use current context if available and matching
        if current_context and current_context.session_id == session_id:
            return current_context.project_path

        # Try to extract from session file's first record
        if session_file.exists():
            try:
                with session_file.open() as f:
                    first_line = f.readline()
                    if first_line:
                        record = json.loads(first_line)
                        cwd = record.get('cwd')
                        if cwd:
                            return Path(cwd)
            except (json.JSONDecodeError, OSError):
                pass

        # Fall back to session folder path (not ideal - lossy encoding)
        return session_file.parent

    def _load_workspace_session(self, session_id: str) -> Session | None:
        """
        Load session data from claude-workspace sessions.json.

        Uses the Session Pydantic model from src.schemas.claude_workspace
        which mirrors the authoritative definitions in claude-workspace.

        Returns:
            Session model if found, None otherwise.
        """
        if not CLAUDE_WORKSPACE_SESSIONS.exists():
            return None

        with CLAUDE_WORKSPACE_SESSIONS.open() as f:
            data = json.load(f)

        adapter = pydantic.TypeAdapter(SessionDatabase)
        db = adapter.validate_python(data)

        for session in db.sessions:
            if session.session_id == session_id:
                return session

        return None

    async def _resolve_session(self, session_id_or_prefix: str) -> SessionInfo:
        """
        Resolve a session ID or prefix to a full session.

        Args:
            session_id_or_prefix: Full session ID or prefix

        Returns:
            SessionInfo for the matched session

        Raises:
            FileNotFoundError: If no sessions match
            AmbiguousSessionError: If multiple sessions match the prefix
        """
        # First try exact match
        exact_match = await self.discovery.find_session_by_id(session_id_or_prefix)
        if exact_match:
            return exact_match

        # Try prefix match
        matches = await self._find_sessions_by_prefix(session_id_or_prefix)

        if not matches:
            raise FileNotFoundError(f'No session found matching: {session_id_or_prefix}')

        if len(matches) > 1:
            raise AmbiguousSessionError(session_id_or_prefix, [m.session_id for m in matches])

        return matches[0]

    async def _find_sessions_by_prefix(self, prefix: str) -> list[SessionInfo]:
        """Find all sessions matching a prefix."""
        claude_sessions_dir = self.discovery.claude_sessions_dir
        if not claude_sessions_dir.exists():
            return []

        # Use rg to find all matching session files
        result = subprocess.run(
            ['rg', '--files', '--glob', f'{prefix}*.jsonl', str(claude_sessions_dir)],
            capture_output=True,
            text=True,
        )

        if not result.stdout.strip():
            return []

        matches = []
        for line in result.stdout.strip().split('\n'):
            session_file = Path(line)
            # Skip agent files
            if session_file.name.startswith('agent-'):
                continue

            session_id = session_file.stem
            session_folder = session_file.parent

            matches.append(SessionInfo(session_id=session_id, session_folder=session_folder))

        return matches
