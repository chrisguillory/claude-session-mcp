"""
Archive operation schemas.

Models for session archive creation and format detection.

Version 2.0 introduces explicit artifact models with structural metadata,
replacing the implicit filename-keyed mappings of v1.x.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from types import MappingProxyType
from typing import Literal

from src.schemas.base import StrictModel
from src.schemas.session.models import SessionRecord, Task
from src.schemas.types import JsonDatetime

# ==============================================================================
# Archive Format Version
# ==============================================================================

ARCHIVE_FORMAT_VERSION = '2.0'
"""Current archive format version. Used when creating new archives."""


# ==============================================================================
# Archive Metadata (MCP Tool Response)
# ==============================================================================


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
    session_records: int  # Records in main {session_id}.jsonl file
    agent_records: int  # Records in agent-*.jsonl files
    file_count: int  # Number of JSONL files included (main + agents)
    files: Sequence[FileMetadata]  # Per-file breakdown
    custom_title: str | None = None  # User-defined session name from /rename (if any)


# ==============================================================================
# Session Archive V2 - Explicit Artifact Models (Composite-First Ordering)
# ==============================================================================
#
# Design principles:
# - Store only source-of-truth fields (filenames derivable from IDs)
# - Explicit structure (no implicit path parsing)
# - Metadata before data (record_count before records)
# - Top-down ordering: composite model first, then constituents
# ==============================================================================


class SessionArchiveV2(StrictModel):
    """
    Archive format v2.0 - explicit artifact models.

    Version history:
    - 2.0: Explicit artifact models, tasks support, agent structure preservation
    - 1.4: Added custom_title field
    - 1.3: Added machine_id field
    - 1.2: Added tool_results and todos fields
    - 1.1: Added plan_files field
    - 1.0: Initial format

    Key changes from v1:
    - files mapping replaced with explicit main_session + agent_files
    - Agent files have nested flag preserving subagents/ directory structure
    - Tasks are now archived (optional, for restore --in-place)
    - claude_code_version is required (never None)
    - All artifact collections use typed entry models

    Derived fields (not stored, computed on restore):
    - Main session filename: f"{session_id}.jsonl"
    - Agent filename: f"agent-{agent_id}.jsonl"
    - Agent relative path: "subagents/..." if nested else just filename
    """

    # Core metadata
    version: str  # "2.0"
    session_id: str
    archived_at: JsonDatetime
    original_project_path: str
    claude_code_version: str  # Required in v2 (use current or "unknown")
    machine_id: str | None = None
    custom_title: str | None = None

    # Explicit artifact collections
    main_session: MainSessionFileEntry
    agent_files: Sequence[AgentFileEntry] = ()
    plan_files: Sequence[PlanFileEntry] = ()
    tool_results: Sequence[ToolResultEntry] = ()
    todos: Sequence[TodoFileEntry] = ()
    tasks: Sequence[Task] = ()  # Canonical Task model from session/models.py

    # Statistics (for quick inspection without iterating records)
    total_session_records: int
    total_agent_records: int


# ==============================================================================
# V2 Entry Models (Constituents)
# ==============================================================================


class MainSessionFileEntry(StrictModel):
    """Main session file content.

    Derived fields:
    - filename: f"{parent.session_id}.jsonl"
    - location: Always flat in project folder (never nested)
    """

    record_count: int  # Metadata first
    records: Sequence[SessionRecord]


class AgentFileEntry(StrictModel):
    """Agent file with structural metadata.

    Derived fields (computed on restore):
    - filename: f"agent-{agent_id}.jsonl"
    - relative_path: f"subagents/agent-{agent_id}.jsonl" if nested
                     else f"agent-{agent_id}.jsonl"

    The nested flag indicates directory structure:
    - nested=False: File in <project-folder>/agent-{id}.jsonl (pre-2.1.2)
    - nested=True: File in <project-folder>/<sid>/subagents/agent-{id}.jsonl (2.1.2+)

    Agent ID patterns (Claude Code 2.1.25+):
    - Plain hex: "5271c147" (agent_type=None)
    - Typed: "aprompt_suggestion-a12dbf" (agent_type="aprompt_suggestion")
    """

    agent_id: str  # Source of truth (e.g., "aprompt_suggestion-d7f1a0")
    agent_type: str | None = None  # Parsed type prefix, or None for plain hex
    nested: bool  # True if in subagents/ directory
    record_count: int  # Metadata first
    records: Sequence[SessionRecord]


class PlanFileEntry(StrictModel):
    """Plan file content.

    Derived fields:
    - filename: f"{slug}.md"
    - location: ~/.claude/plans/{slug}.md
    """

    slug: str
    content: str


class ToolResultEntry(StrictModel):
    """Tool result file content.

    Derived fields:
    - filename: f"{tool_use_id}.txt"
    - location: projects/<enc>/<sid>/tool-results/{tool_use_id}.txt
    """

    tool_use_id: str
    content: str


class TodoFileEntry(StrictModel):
    """Todo file content.

    Derived fields:
    - filename: f"{session_id}-agent-{agent_id}.json"
    - location: ~/.claude/todos/{filename}

    Note: content is opaque JSON string - structure may vary by Claude Code version.
    """

    agent_id: str  # Which agent this todo belongs to
    content: str  # JSON string (not parsed, may vary)


# ==============================================================================
# Session Archive V1 - Legacy Format (Backward Compatibility)
# ==============================================================================


class SessionArchiveV1(StrictModel):
    """
    Legacy archive format (v1.0-v1.4).

    Uses implicit filename-keyed mappings. Agent file directory structure
    is lost (only filename stored, not full relative path).

    Kept for backward compatibility when loading old archives.
    Use SessionArchiveV2 for new archives.
    """

    version: str  # "1.0" through "1.4"
    session_id: str
    archived_at: JsonDatetime
    original_project_path: str
    claude_code_version: str | None  # Can be None in v1
    files: Mapping[str, Sequence[SessionRecord]]  # filename -> records
    plan_files: Mapping[str, str] = MappingProxyType({})  # slug -> content
    tool_results: Mapping[str, str] = MappingProxyType({})  # tool_use_id -> content
    todos: Mapping[str, str] = MappingProxyType({})  # filename -> JSON content
    machine_id: str | None = None
    custom_title: str | None = None


# ==============================================================================
# Union Type for Archive Loading
# ==============================================================================

# Used by restore/clone to handle either format
type SessionArchive = SessionArchiveV1 | SessionArchiveV2


# ==============================================================================
# Migration Functions
# ==============================================================================


def migrate_v1_to_v2(v1: SessionArchiveV1) -> SessionArchiveV2:
    """
    Migrate v1.x archive to v2.0 format in memory.

    Conservative approach:
    - Agent files assumed flat (v1 loses nested info)
    - Tasks empty (v1 doesn't capture tasks)
    - claude_code_version set to "unknown" if None

    This is NOT persisted - migration happens on load for consumption.

    Args:
        v1: Legacy v1.x archive

    Returns:
        SessionArchiveV2 equivalent
    """
    # Parse main session
    main_filename = f'{v1.session_id}.jsonl'
    main_records = list(v1.files.get(main_filename, []))

    main_session = MainSessionFileEntry(
        record_count=len(main_records),
        records=main_records,
    )

    # Parse agent files (assume flat - v1 loses nested info)
    agent_entries: list[AgentFileEntry] = []
    agent_total_records = 0

    for filename, records in v1.files.items():
        if not filename.startswith('agent-'):
            continue

        # Parse agent ID and type from filename
        agent_id, agent_type = parse_agent_metadata(filename)
        record_list = list(records)

        agent_entries.append(
            AgentFileEntry(
                agent_id=agent_id,
                agent_type=agent_type,
                nested=False,  # v1 loses nested info - assume flat
                record_count=len(record_list),
                records=record_list,
            )
        )
        agent_total_records += len(record_list)

    # Convert plan_files
    plan_entries = [PlanFileEntry(slug=slug, content=content) for slug, content in v1.plan_files.items()]

    # Convert tool_results
    tool_result_entries = [
        ToolResultEntry(tool_use_id=tid, content=content) for tid, content in v1.tool_results.items()
    ]

    # Convert todos - parse agent_id from filename
    todo_entries: list[TodoFileEntry] = []
    for filename, content in v1.todos.items():
        agent_id = _extract_agent_id_from_todo_filename(filename, v1.session_id)
        todo_entries.append(TodoFileEntry(agent_id=agent_id, content=content))

    return SessionArchiveV2(
        version='2.0',
        session_id=v1.session_id,
        archived_at=v1.archived_at,
        original_project_path=v1.original_project_path,
        claude_code_version=v1.claude_code_version or 'unknown',
        machine_id=v1.machine_id,
        custom_title=v1.custom_title,
        main_session=main_session,
        agent_files=agent_entries,
        plan_files=plan_entries,
        tool_results=tool_result_entries,
        todos=todo_entries,
        tasks=[],  # v1 has no tasks
        total_session_records=len(main_records) + agent_total_records,
        total_agent_records=agent_total_records,
    )


def parse_agent_metadata(filename: str) -> tuple[str, str | None]:
    """
    Parse agent ID and type from filename.

    Examples:
        "agent-5271c147.jsonl" -> ("5271c147", None)
        "agent-aprompt_suggestion-a12dbf.jsonl" -> ("aprompt_suggestion-a12dbf", "aprompt_suggestion")

    Pattern detection:
        agent-<type>-<6-hex>.jsonl -> typed agent
        agent-<8-hex>.jsonl -> plain agent (no type)
        agent-<anything-else>.jsonl -> treat as plain agent ID

    Args:
        filename: Agent filename like "agent-xxx.jsonl"

    Returns:
        (agent_id, agent_type) tuple
    """
    # Remove "agent-" prefix and ".jsonl" suffix
    base = filename.removeprefix('agent-').removesuffix('.jsonl')

    # Check for typed agent pattern: <type>-<6-hex>
    if '-' in base:
        # Split from the RIGHT to handle types with dashes (e.g., "some-type-abc123")
        parts = base.rsplit('-', 1)
        if len(parts) == 2 and len(parts[1]) == 6:
            # Looks like type-hex pattern
            agent_type = parts[0]
            return (base, agent_type)

    # Plain hex or unknown pattern - no type
    return (base, None)


def _extract_agent_id_from_todo_filename(filename: str, session_id: str) -> str:
    """
    Extract agent ID from todo filename.

    Format: {session_id}-agent-{agent_id}.json

    Examples:
        "019abc-agent-5271c147.json" -> "5271c147"
        "019abc-agent-aprompt_suggestion-a12dbf.json" -> "aprompt_suggestion-a12dbf"

    Args:
        filename: Todo filename
        session_id: Session ID to strip

    Returns:
        Agent ID portion
    """
    prefix = f'{session_id}-agent-'
    if not filename.startswith(prefix):
        # Fallback: try to extract anything after "agent-"
        if '-agent-' in filename:
            return filename.split('-agent-', 1)[1].removesuffix('.json')
        # Last resort: return filename without extension
        return filename.removesuffix('.json')

    return filename.removeprefix(prefix).removesuffix('.json')
