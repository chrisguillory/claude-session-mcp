"""
Session context operation schemas.

Models for session context information returned by get_session_info tool.
"""

from __future__ import annotations

from typing import Literal

from src.schemas.base import StrictModel
from src.schemas.types import JsonDatetime

# Type aliases for session origin tracking (from claude-workspace sessions.json)
SessionSource = Literal['startup', 'resume', 'compact', 'clear', 'unknown']
SessionState = Literal['active', 'exited', 'completed', 'crashed', 'unknown']


class SessionContext(StrictModel):
    """
    Comprehensive information about a Claude Code session.

    Returned by get_session_info MCP tool and info CLI command.
    Combines data from multiple sources:
    - Session discovery (session files in ~/.claude/projects/)
    - Claude-workspace tracking (~/.claude-workspace/sessions.json)
    - MCP server state (for current session: claude_pid, temp_dir)
    - Lineage tracking (~/.claude-session-mcp/lineage.json)

    Field ordering:
    - Identity (who)
    - Temporal (when)
    - Paths (where)
    - Environment (runtime context, only for current session)
    - Origin (how session was created)
    - Characteristics (derived/computed flags)
    """

    # Identity
    session_id: str
    custom_title: str | None = None  # User-defined session name from /rename

    # Temporal - Authoritative timestamps (each has one source)
    first_message_at: JsonDatetime | None = None  # First record timestamp from JSONL
    process_created_at: JsonDatetime | None = None  # From psutil (live) or sessions.json (fallback)
    session_ended_at: JsonDatetime | None = None  # SessionEnd hook only (clean exit)
    session_end_reason: str | None = None  # SessionEnd hook only
    crash_detected_at: JsonDatetime | None = None  # Crash detection only
    cloned_at: JsonDatetime | None = None  # From lineage.json (cloned sessions only)

    # Paths
    project_path: str  # Working directory / project root
    session_file: str  # ~/.claude/projects/{encoded_path}/{session_id}.jsonl
    debug_file: str  # ~/.claude/debug/{session_id}.txt

    # Environment (only populated for current session via MCP, None for CLI queries)
    machine_id: str | None  # user@hostname identifier
    claude_pid: int | None  # Claude Code process ID
    claude_version: str | None  # Claude Code CLI version (e.g., "2.1.12")
    temp_dir: str | None  # MCP server's temp directory

    # Origin (from claude-workspace sessions.json, defaults to 'unknown' if unavailable)
    source: SessionSource  # How session was created: startup, resume, compact, clear
    state: SessionState  # Current session state: active, exited, completed, crashed
    parent_id: str | None  # Previous conversation's leafUuid (for resume/compact/clear)

    # Characteristics (computed)
    is_native: bool  # True = UUIDv4 (native Claude session), False = UUIDv7 (cloned/restored)
    has_lineage: bool  # True if session has entry in lineage tracking (was cloned/restored by us)
