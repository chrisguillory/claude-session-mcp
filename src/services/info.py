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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import psutil
import pydantic

from src.schemas.claude_workspace import Session, SessionDatabase
from src.schemas.operations.context import SessionContext
from src.schemas.operations.discovery import SessionInfo
from src.services.artifacts import extract_custom_title_from_file
from src.services.delete import is_native_session
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


# noinspection PyMethodMayBeStatic
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

        # Check lineage and get cloned_at timestamp
        lineage_entry = self.lineage.get_entry(full_session_id)
        has_lineage = lineage_entry is not None
        cloned_at = lineage_entry.cloned_at if lineage_entry else None

        # Compute characteristics
        native = is_native_session(full_session_id)

        # Get authoritative timestamps
        first_message_at = self._get_first_message_timestamp(session_file)

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

        claude_version = await self._get_claude_version(
            claude_pid=claude_pid,
            session_file=session_file,
            workspace_version=workspace_session.metadata.claude_version if workspace_session else None,
        )

        # Get process_created_at with fallback
        process_created_at_fallback = workspace_session.metadata.process_created_at if workspace_session else None
        process_created_at = self._get_process_created_at(claude_pid, process_created_at_fallback)

        return SessionContext(
            # Identity
            session_id=full_session_id,
            custom_title=custom_title,
            # Temporal - Authoritative timestamps
            first_message_at=first_message_at,
            process_created_at=process_created_at,
            session_ended_at=workspace_session.metadata.session_ended_at if workspace_session else None,
            session_end_reason=workspace_session.metadata.session_end_reason if workspace_session else None,
            crash_detected_at=workspace_session.metadata.crash_detected_at if workspace_session else None,
            cloned_at=cloned_at,
            # Paths
            project_path=str(project_path),
            session_file=str(session_file),
            debug_file=str(debug_file),
            # Environment
            machine_id=machine_id,
            claude_pid=claude_pid,
            claude_version=claude_version,
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

    async def _get_claude_version(
        self,
        claude_pid: int | None,
        session_file: Path,
        workspace_version: str | None,
    ) -> str | None:
        """
        Get Claude Code version with proper fallback chain.

        Priority:
        1. Process-based (if PID available and process exists)
        2. JSONL records (from session file)
        3. Workspace sessions.json (lowest priority fallback)

        Args:
            claude_pid: PID of Claude process (if available)
            session_file: Path to session JSONL file
            workspace_version: Version from workspace sessions.json (if available)

        Returns:
            Version string, or None if not determinable
        """
        # Try process-based first (most accurate for current operations)
        if claude_pid is not None:
            from src.services.version import get_version_from_process

            version = get_version_from_process(claude_pid)
            if version:
                return version

        # Try JSONL records (read just enough to find version)
        if session_file.exists():
            with session_file.open() as f:
                for line in f:
                    record = json.loads(line)
                    version = record.get('version')
                    if isinstance(version, str):
                        return version

        # Lowest fallback: workspace sessions.json
        return workspace_version

    def _get_first_message_timestamp(self, session_file: Path) -> datetime | None:
        """
        Extract the first timestamp from a session JSONL file.

        Scans for the first record with a 'timestamp' field, skipping records
        without timestamps (like summary and custom-title records).

        Args:
            session_file: Path to the session JSONL file

        Returns:
            datetime of first timestamped record, or None if not found
        """
        if not session_file.exists():
            return None

        with session_file.open() as f:
            for line in f:
                if not (line := line.strip()):
                    continue

                # Parse JSON - fail fast on errors (no swallowing)
                record = json.loads(line)

                # Look for timestamp field (not all records have it)
                timestamp_str = record.get('timestamp')
                if timestamp_str is not None:
                    # Parse ISO 8601 timestamp
                    return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

        return None

    def _get_process_created_at(
        self,
        pid: int | None,
        fallback: datetime | None,
    ) -> datetime | None:
        """
        Get process creation time from OS (authoritative).

        Uses psutil directly when process is running.
        Falls back to sessions.json cached value when process is gone.

        Args:
            pid: Process ID to query (may be None)
            fallback: Cached value from sessions.json to use if process gone

        Returns:
            datetime of process creation (local timezone), or None if not available
        """
        if pid is None:
            return fallback

        try:
            proc = psutil.Process(pid)
            create_time = proc.create_time()
            # Note: AccessDenied not caught per fail-fast policy - handle if observed
            return datetime.fromtimestamp(create_time, UTC).astimezone()
        except psutil.NoSuchProcess:
            # Process gone - use cached fallback from sessions.json
            return fallback

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

        Delegates to SessionDiscoveryService which handles both exact matches
        and prefix matching.

        Args:
            session_id_or_prefix: Full session ID or prefix

        Returns:
            SessionInfo for the matched session

        Raises:
            FileNotFoundError: If no sessions match
            AmbiguousSessionError: If multiple sessions match (from discovery service)
        """
        match = await self.discovery.find_session_by_id(session_id_or_prefix)
        if not match:
            raise FileNotFoundError(f'No session found matching: {session_id_or_prefix}')
        return match
