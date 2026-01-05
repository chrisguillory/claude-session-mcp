"""Pydantic models for claude-workspace session tracking data.

These models represent the session data stored in ~/.claude-workspace/sessions.json,
which is written by claude-workspace hooks when Claude Code sessions start/end.

REFERENCE IMPLEMENTATION:
    https://github.com/chrisguillory/claude-workspace/blob/main/local-lib/local_lib/session_tracker.py

    These models are designed to be compatible with the Session and SessionMetadata
    models defined in claude-workspace's local-lib. The authoritative definitions
    live there; these are compatible copies for use in this repository.

FUTURE CONSIDERATION:
    As interconnections between claude-session-mcp and claude-workspace grow,
    these repositories may be merged. This is the first shared data structure
    between them. If/when merged, these models should be consolidated with
    the local-lib definitions to maintain a single source of truth.

USAGE:
    from src.schemas.workspace import Session, SessionDatabase
    from pydantic import TypeAdapter
    import json

    adapter = TypeAdapter(SessionDatabase)
    with open("~/.claude-workspace/sessions.json") as f:
        db = adapter.validate_python(json.load(f))

    for session in db.sessions:
        if session.state == "active":
            print(f"Active session: {session.session_id}, PID: {session.metadata.claude_pid}")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.schemas.types import JsonDatetime

# Type aliases matching claude-workspace definitions
SessionState = Literal['active', 'exited', 'completed', 'crashed']
SessionSource = Literal['startup', 'resume', 'compact', 'clear']


class SessionMetadata(BaseModel):
    """Derived session information from claude-workspace hooks.

    This metadata is populated when the SessionStart hook fires:
    - claude_pid: Found via process tree walking from the hook process
    - started_at: Timestamp when SessionStart hook executed
    - ended_at: Timestamp when SessionEnd hook executed (if session ended)
    - parent_id: Extracted from transcript file for compacted sessions
    - crash_detected_at: When crash detection identified an orphaned session
    """

    model_config = ConfigDict(extra='forbid', strict=True)

    claude_pid: int
    started_at: JsonDatetime
    ended_at: JsonDatetime | None = None
    parent_id: str | None = None
    crash_detected_at: JsonDatetime | None = None


class Session(BaseModel):
    """A Claude Code session tracked by claude-workspace.

    Represents a single Claude Code session from ~/.claude-workspace/sessions.json.
    Sessions are created by the SessionStart hook and updated by SessionEnd hook.
    """

    model_config = ConfigDict(extra='forbid', strict=True)

    # Identity
    session_id: str

    # Current status
    state: SessionState

    # Location
    project_dir: str  # Working directory (cwd from hook input)
    transcript_path: str  # Path to session JSONL file

    # Origin - how the session was initiated
    source: SessionSource

    # Detailed information
    metadata: SessionMetadata


class SessionDatabase(BaseModel):
    """Container for all tracked sessions in sessions.json.

    The sessions.json file contains a single SessionDatabase object
    with a list of all known sessions (active, exited, crashed).
    """

    model_config = ConfigDict(extra='forbid', strict=True)

    sessions: Sequence[Session] = ()
