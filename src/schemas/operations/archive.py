"""
Archive operation schemas.

Models for session archive creation and format detection.
Extracted from services/archive.py for reuse.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from types import MappingProxyType
from typing import Literal

from src.schemas.base import StrictModel
from src.schemas.session.models import SessionRecord
from src.schemas.types import JsonDatetime

# Archive format version - single source of truth for what this code creates
ARCHIVE_FORMAT_VERSION = '1.3'


class FileMetadata(StrictModel):
    """Metadata about a single JSONL file in the archive."""

    filename: str
    record_count: int


class ArchiveMetadata(StrictModel):
    """
    Metadata about created archive.

    Returned by save_current_session MCP tool.
    """

    file_path: str
    session_id: str
    format: Literal['json', 'zst']
    size_mb: float  # Size in megabytes, rounded to 2 decimal places
    archived_at: datetime
    record_count: int  # Total records across all files
    file_count: int  # Number of JSONL files included (main + agents)
    files: list[FileMetadata]  # Per-file breakdown


class SessionArchive(StrictModel):
    """
    Complete session archive structure (written to JSON).

    Version history:
    - 1.0: Initial format (session JSONL files only)
    - 1.1: Added plan_files field
    - 1.2: Added tool_results and todos fields
    - 1.3: Added machine_id field for cross-machine lineage tracking

    Cloned artifact identification patterns:
    - Session IDs: UUIDv7 (vs Claude's UUIDv4)
    - Plan slugs: {old-slug}-clone-{session-prefix}
    - Agent IDs: {old-agent-id}-clone-{session-prefix}

    Design decisions:
    - Agent files get new IDs on clone/restore for same-project fork safety
    - tool_results uses original tool_use_ids (nested under session_id dir)
    - todos filenames have primary session ID portion updated
    """

    version: str  # Required - use ARCHIVE_FORMAT_VERSION when creating
    session_id: str
    archived_at: JsonDatetime
    original_project_path: str
    claude_code_version: str  # Claude Code version at archive time
    files: Mapping[str, Sequence[SessionRecord]]  # filename -> records
    plan_files: Mapping[str, str] = MappingProxyType({})  # slug -> content (v1.1+)
    tool_results: Mapping[str, str] = MappingProxyType({})  # tool_use_id -> content (v1.2+)
    todos: Mapping[str, str] = MappingProxyType({})  # original_filename -> JSON content (v1.2+)
    machine_id: str | None = None  # Machine where archive was created (v1.3+)
