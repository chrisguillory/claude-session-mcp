"""Session repository - typed access layer for Claude Code sessions.

Provides convenient, typed access to session data without exposing
the complexity of JSONL parsing and record navigation.

This is Layer 1 in the architecture:
    Layer 0: Storage (JSONL files)
    Layer 1: SessionRepository (this file) <- typed access
    Layer 2: Transforms (chunking, rollups)
    Layer 3: Consumers (document-search, MCP tools)
"""

from __future__ import annotations

import json
import types
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from functools import cache, cached_property
from pathlib import Path
from typing import Annotated, Literal, Union, get_args, get_origin

from src.schemas.session import models
from src.schemas.session.markers import PathMarker
from src.schemas.session.models import BuiltinToolName

# =============================================================================
# Types: Tool Names (repository-layer extensions)
# =============================================================================

# Synthetic name for MCP tools (repository-layer concept)
# Actual MCP tool names follow pattern: mcp__{server}__{operation}
MCPToolName = Literal['MCP']

# All tool names including synthetic MCP
ToolName = BuiltinToolName | MCPToolName


def _get_tool_name(tool_input: models.ToolInput) -> ToolName:
    """Get the tool name from a ToolInput using pattern matching.

    Returns the BuiltinToolName for Claude Code tools, or 'MCP' for third-party
    MCP tools. This mirrors get_tool_type() in helpers.py but returns actual
    tool names rather than interpretive types.

    Raises:
        ValueError: For unknown tool input types (catches new tools we haven't modeled).
    """
    match tool_input:
        # File operations
        case models.ReadToolInput():
            return 'Read'
        case models.WriteToolInput() | models.MalformedWriteToolInput():
            return 'Write'
        case models.EditToolInput():
            return 'Edit'
        case models.GlobToolInput():
            return 'Glob'
        case models.GrepToolInput():
            return 'Grep'
        case models.NotebookEditToolInput():
            return 'NotebookEdit'

        # Execution (current)
        case models.BashToolInput():
            return 'Bash'
        case models.TaskToolInput():
            return 'Task'
        case models.TaskOutputToolInput():
            return 'TaskOutput'

        # Execution (legacy)
        case models.AgentOutputToolInput():
            return 'AgentOutput'
        case models.BashOutputToolInput():
            return 'BashOutput'
        case models.KillShellToolInput():
            return 'KillShell'

        # Web
        case models.WebSearchToolInput():
            return 'WebSearch'
        case models.WebFetchToolInput():
            return 'WebFetch'

        # LSP
        case models.LSPToolInput():
            return 'LSP'

        # Planning
        case models.EnterPlanModeToolInput():
            return 'EnterPlanMode'
        case models.ExitPlanModeToolInput():
            return 'ExitPlanMode'

        # User interaction (current)
        case models.AskUserQuestionToolInput():
            return 'AskUserQuestion'

        # User interaction (legacy)
        case models.TodoWriteToolInput():
            return 'TodoWrite'

        # MCP management (current)
        case models.MCPSearchToolInput():
            return 'ToolSearch'  # Current name is ToolSearch, MCPSearch is legacy
        case models.ListMcpResourcesToolInput():
            return 'ListMcpResourcesTool'
        case models.ReadMcpResourceToolInput():
            return 'ReadMcpResourceTool'

        # Skills
        case models.SkillToolInput():
            return 'Skill'

        # Task management (2.1.17+)
        case models.TaskCreateToolInput():
            return 'TaskCreate'
        case models.TaskUpdateToolInput():
            return 'TaskUpdate'
        case models.TaskListToolInput():
            return 'TaskList'

        # MCP (third-party) - must be last
        case models.MCPToolInput():
            return 'MCP'

        case _:
            raise ValueError(f'Unknown tool input type: {type(tool_input).__name__}')


def _has_path_marker(annotation: type) -> bool:
    """Check if a type annotation contains PathMarker."""
    # Handle Python 3.12+ type aliases (e.g., type PathField = Annotated[str, PathMarker()])
    if hasattr(annotation, '__value__'):
        return _has_path_marker(annotation.__value__)

    origin = get_origin(annotation)

    # Direct Annotated type: Annotated[str, PathMarker()]
    if origin is Annotated:
        args = get_args(annotation)
        # args[0] is base type, args[1:] are metadata
        return any(isinstance(meta, PathMarker) for meta in args[1:])

    # Union type: PathField | None
    if origin is Union or origin is types.UnionType:
        return any(_has_path_marker(arg) for arg in get_args(annotation))

    return False


def _extract_file_path(tool_input: models.ToolInput) -> str | None:
    """Extract file path from tool input using pattern matching.

    Raises:
        ValueError: If tool input has a PathMarker field we didn't handle.
    """
    match tool_input:
        case models.ReadToolInput():
            return tool_input.file_path
        case models.WriteToolInput() | models.MalformedWriteToolInput():
            return tool_input.file_path
        case models.EditToolInput():
            return tool_input.file_path
        case models.NotebookEditToolInput():
            return tool_input.notebook_path
        case models.GlobToolInput():
            return tool_input.path
        case models.GrepToolInput():
            return tool_input.path
        case models.LSPToolInput():
            return tool_input.filePath
        case _:
            _validate_no_unhandled_path_fields(type(tool_input))
            return None


@cache
def _validate_no_unhandled_path_fields(tool_input_type: type[models.ToolInput]) -> None:
    """Validate that a ToolInput type has no PathMarker fields.

    Cached per type - only checks once, then returns immediately.

    Raises:
        ValueError: If the type has PathMarker fields that aren't handled.
    """
    path_fields = []
    for field_name, field_info in tool_input_type.model_fields.items():
        annotation = field_info.annotation
        if annotation is not None and _has_path_marker(annotation):
            path_fields.append(field_name)

    if path_fields:
        raise ValueError(
            f'{tool_input_type.__name__} has PathMarker fields {path_fields} but is not handled in _extract_file_path'
        )


# =============================================================================
# Types: Session Identity
# =============================================================================


@dataclass(frozen=True)
class SessionInfo:
    """Basic session metadata."""

    session_id: str
    session_path: Path
    project_path: str
    slug: str | None

    line_count: int
    file_size_bytes: int

    first_timestamp: str | None
    last_timestamp: str | None


# =============================================================================
# Types: Session Statistics
# =============================================================================


@dataclass(frozen=True)
class SessionStats:
    """Aggregate session statistics.

    Contains only objective/structural aggregates.
    Interpretive groupings (like tool categories) are left to consumers.
    """

    turn_count: int
    user_message_count: int
    assistant_message_count: int

    tool_invocation_count: int
    tool_error_count: int
    tool_by_name: dict[str, int]  # Counts by tool name (e.g., "Read": 50)

    thinking_block_count: int
    summary_count: int

    progress_event_count: int
    progress_by_type: dict[str, int]
    total_duration_ms: int | None

    unique_file_paths: frozenset[str]


# =============================================================================
# Types: Tool Use/Result (Linked)
# =============================================================================


@dataclass
class ToolResultInfo:
    """Tool result with optional link to invocation."""

    line_number: int
    turn_index: int
    tool_use_id: str

    content: str | Sequence[models.TextContent | models.ImageContent | models.ToolReferenceContent] | None
    content_length: int
    is_error: bool

    # Linked after construction
    invocation: ToolUseInfo | None = field(default=None, repr=False)


@dataclass
class ToolUseInfo:
    """Tool invocation with optional link to result.

    This class provides structural/objective data only.
    Interpretive concepts like 'tool_category' are left to consumers.
    """

    line_number: int
    turn_index: int
    tool_use_id: str
    name: ToolName  # Typed tool name from _get_tool_name()

    input: models.ToolInput  # The typed input object

    # Extracted from input (objective, structural)
    file_path: str | None  # For file operations

    # Parsed from name for MCP tools (objective, structural)
    mcp_server: str | None  # e.g., "perplexity"
    mcp_operation: str | None  # e.g., "ask"

    # Linked after construction
    result: ToolResultInfo | None = field(default=None, repr=False)

    @property
    def is_mcp(self) -> bool:
        """Whether this is an MCP tool invocation."""
        return self.mcp_server is not None


# =============================================================================
# Types: Turn (Conversation Unit)
# =============================================================================


@dataclass
class Turn:
    """A conversation turn: user message + assistant response(s)."""

    index: int

    user_line: int
    user_record: models.UserRecord
    user_text: str

    assistant_entries: list[tuple[int, models.MainAssistantRecord | models.AgentAssistantRecord]] = field(
        default_factory=list
    )
    assistant_text: str = ''

    tool_uses: list[ToolUseInfo] = field(default_factory=list)
    thinking_texts: list[str] = field(default_factory=list)

    duration_ms: int | None = None


# =============================================================================
# Types: Progress
# =============================================================================


@dataclass(frozen=True)
class ProgressInfo:
    """Progress event summary."""

    line_number: int
    progress_type: str

    tool_name: str | None = None
    server_name: str | None = None
    command: str | None = None


# =============================================================================
# SessionRepository
# =============================================================================


class SessionRepository:
    """Typed access layer for Claude Code session files.

    Provides convenient access to session data:
    - Random access by line number
    - Iteration over records
    - Structured queries (turns, tool uses, etc.)
    - Aggregate statistics

    Usage:
        repo = SessionRepository(Path("~/.claude/projects/.../session.jsonl"))

        # Get metadata
        info = repo.session_info
        stats = repo.session_stats

        # Random access
        record = repo.get_line(456)

        # Structured access
        for turn in repo.turns:
            print(f"Turn {turn.index}: {turn.user_text[:50]}")
            for tool in turn.tool_uses:
                print(f"  - {tool.name}: {tool.file_path}")

        # Query tool uses (already linked to results)
        for tool_use in repo.tool_uses:
            if tool_use.result and tool_use.result.is_error:
                print(f"Error in {tool_use.name} at line {tool_use.line_number}")
    """

    def __init__(self, session_path: Path) -> None:
        """Initialize repository for a session file.

        Args:
            session_path: Path to the session JSONL file.
        """
        self._path = session_path
        self._records: dict[int, models.SessionRecord] | None = None

    # =========================================================================
    # Path Utilities
    # =========================================================================

    @property
    def path(self) -> Path:
        """The session file path."""
        return self._path

    @property
    def session_id(self) -> str:
        """Session ID extracted from filename."""
        return self._path.stem

    @property
    def project_path(self) -> str:
        """Project path decoded from parent directory name."""
        normalized = self._path.parent.name
        if normalized.startswith('-'):
            return '/' + normalized[1:].replace('-', '/')
        return normalized

    # =========================================================================
    # Loading
    # =========================================================================

    def _ensure_loaded(self) -> None:
        """Lazy load all records if not already loaded."""
        if self._records is not None:
            return

        self._records = {}
        with self._path.open(encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                record = models.SessionRecordAdapter.validate_python(raw)
                self._records[line_num] = record

    @property
    def _all_records(self) -> dict[int, models.SessionRecord]:
        """All records by line number (lazy loaded)."""
        self._ensure_loaded()
        assert self._records is not None
        return self._records

    # =========================================================================
    # Random Access
    # =========================================================================

    def get_line(self, line_number: int) -> models.SessionRecord:
        """Get the record at a specific line (1-indexed).

        Args:
            line_number: The line number (1-indexed).

        Returns:
            The parsed SessionRecord.

        Raises:
            KeyError: If line number doesn't exist.
        """
        return self._all_records[line_number]

    def get_lines(self, start: int, end: int) -> list[models.SessionRecord]:
        """Get records in a line range (inclusive).

        Args:
            start: Start line number (1-indexed, inclusive).
            end: End line number (1-indexed, inclusive).

        Returns:
            List of records in the range.
        """
        return [self._all_records[ln] for ln in range(start, end + 1) if ln in self._all_records]

    # =========================================================================
    # Iteration
    # =========================================================================

    def iter_records(self) -> Iterator[tuple[int, models.SessionRecord]]:
        """Iterate over (line_number, record) pairs in order.

        Yields:
            Tuples of (line_number, record).
        """
        for line_num in sorted(self._all_records.keys()):
            yield line_num, self._all_records[line_num]

    @property
    def line_count(self) -> int:
        """Total number of non-empty lines."""
        return len(self._all_records)

    # =========================================================================
    # Session Info
    # =========================================================================

    @cached_property
    def session_info(self) -> SessionInfo:
        """Basic session metadata."""
        first_ts: str | None = None
        last_ts: str | None = None
        slug: str | None = None

        for _, record in self.iter_records():
            # Get timestamps from records that have them
            if hasattr(record, 'timestamp') and record.timestamp:
                if first_ts is None:
                    first_ts = record.timestamp
                last_ts = record.timestamp

            # Get slug from user/assistant records
            if hasattr(record, 'slug') and record.slug:
                slug = record.slug

        return SessionInfo(
            session_id=self.session_id,
            session_path=self._path,
            project_path=self.project_path,
            slug=slug,
            line_count=self.line_count,
            file_size_bytes=self._path.stat().st_size,
            first_timestamp=first_ts,
            last_timestamp=last_ts,
        )

    # =========================================================================
    # Turns
    # =========================================================================

    @cached_property
    def turns(self) -> list[Turn]:
        """All conversation turns.

        A turn starts with a UserRecord and includes all subsequent
        AssistantRecords until the next UserRecord.
        """
        turns: list[Turn] = []
        current_turn: Turn | None = None
        turn_index = 0

        for line_num, record in self.iter_records():
            if isinstance(record, models.UserRecord):
                # Start new turn
                turn_index += 1
                user_text = self._extract_user_text(record)
                current_turn = Turn(
                    index=turn_index,
                    user_line=line_num,
                    user_record=record,
                    user_text=user_text,
                )
                turns.append(current_turn)

            elif (
                isinstance(
                    record,
                    (
                        models.NormalMainAssistantRecord,
                        models.ErrorMainAssistantRecord,
                        models.NormalAgentAssistantRecord,
                        models.ErrorAgentAssistantRecord,
                    ),
                )
                and current_turn
            ):
                current_turn.assistant_entries.append((line_num, record))
                # Extract text and thinking
                if record.message and isinstance(record.message.content, list):
                    for block in record.message.content:
                        if isinstance(block, models.TextContent):
                            if current_turn.assistant_text:
                                current_turn.assistant_text += '\n\n'
                            current_turn.assistant_text += block.text
                        elif isinstance(block, models.ThinkingContent):
                            current_turn.thinking_texts.append(block.thinking)
                        elif isinstance(block, models.ToolUseContent):
                            tool_info = self._make_tool_use_info(block, line_num, turn_index)
                            current_turn.tool_uses.append(tool_info)

            elif isinstance(record, models.TurnDurationSystemRecord) and current_turn:
                current_turn.duration_ms = record.durationMs

        # Link tool results
        self._link_tool_results(turns)

        return turns

    def get_turn(self, index: int) -> Turn:
        """Get a specific turn by index (1-indexed).

        Args:
            index: Turn index (1-indexed).

        Returns:
            The Turn object.

        Raises:
            IndexError: If turn doesn't exist.
        """
        if index < 1 or index > len(self.turns):
            raise IndexError(f'Turn {index} not found (have {len(self.turns)} turns)')
        return self.turns[index - 1]

    # =========================================================================
    # Tool Uses
    # =========================================================================

    @cached_property
    def tool_uses(self) -> list[ToolUseInfo]:
        """All tool invocations (already linked to results)."""
        uses: list[ToolUseInfo] = []
        for turn in self.turns:
            uses.extend(turn.tool_uses)
        return uses

    def get_tool_uses_by_name(self, name: str) -> list[ToolUseInfo]:
        """Get tool invocations filtered by tool name.

        Args:
            name: Tool name like "Read", "Bash", "mcp__perplexity__ask".

        Returns:
            Filtered list of ToolUseInfo.
        """
        return [u for u in self.tool_uses if u.name == name]

    def get_mcp_tool_uses(self, server: str | None = None) -> list[ToolUseInfo]:
        """Get MCP tool invocations, optionally filtered by server.

        Args:
            server: Optional MCP server name to filter by.

        Returns:
            Filtered list of ToolUseInfo.
        """
        mcp_uses = [u for u in self.tool_uses if u.is_mcp]
        if server:
            mcp_uses = [u for u in mcp_uses if u.mcp_server == server]
        return mcp_uses

    # =========================================================================
    # Progress Events
    # =========================================================================

    @cached_property
    def progress_events(self) -> list[ProgressInfo]:
        """All progress events."""
        events: list[ProgressInfo] = []

        for line_num, record in self.iter_records():
            if isinstance(record, models.ProgressRecord) and record.data:
                data = record.data
                progress_type = getattr(data, 'type', 'unknown')

                events.append(
                    ProgressInfo(
                        line_number=line_num,
                        progress_type=progress_type,
                        tool_name=getattr(data, 'toolName', None),
                        server_name=getattr(data, 'serverName', None),
                        command=getattr(data, 'command', None),
                    )
                )

        return events

    # =========================================================================
    # Statistics
    # =========================================================================

    @cached_property
    def session_stats(self) -> SessionStats:
        """Aggregate session statistics."""
        user_count = 0
        assistant_count = 0
        thinking_count = 0
        summary_count = 0
        progress_by_type: dict[str, int] = {}
        total_duration_ms = 0
        has_duration = False

        for _, record in self.iter_records():
            if isinstance(record, models.UserRecord):
                user_count += 1
            elif isinstance(
                record,
                (
                    models.NormalMainAssistantRecord,
                    models.ErrorMainAssistantRecord,
                    models.NormalAgentAssistantRecord,
                    models.ErrorAgentAssistantRecord,
                ),
            ):
                assistant_count += 1
                if record.message and isinstance(record.message.content, list):
                    for block in record.message.content:
                        if isinstance(block, models.ThinkingContent):
                            thinking_count += 1
            elif isinstance(record, models.SummaryRecord):
                summary_count += 1
            elif isinstance(record, models.TurnDurationSystemRecord):
                total_duration_ms += record.durationMs
                has_duration = True

        # Progress stats
        for event in self.progress_events:
            progress_by_type[event.progress_type] = progress_by_type.get(event.progress_type, 0) + 1

        # Tool stats
        tool_by_name: dict[str, int] = {}
        error_count = 0
        file_paths: set[str] = set()

        for use in self.tool_uses:
            tool_by_name[use.name] = tool_by_name.get(use.name, 0) + 1
            if use.file_path:
                file_paths.add(use.file_path)
            if use.result and use.result.is_error:
                error_count += 1

        return SessionStats(
            turn_count=len(self.turns),
            user_message_count=user_count,
            assistant_message_count=assistant_count,
            tool_invocation_count=len(self.tool_uses),
            tool_error_count=error_count,
            tool_by_name=tool_by_name,
            thinking_block_count=thinking_count,
            summary_count=summary_count,
            progress_event_count=len(self.progress_events),
            progress_by_type=progress_by_type,
            total_duration_ms=total_duration_ms if has_duration else None,
            unique_file_paths=frozenset(file_paths),
        )

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _extract_user_text(self, record: models.UserRecord) -> str:
        """Extract text content from a user record."""
        if not record.message:
            return ''

        content = record.message.content
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            texts = [block.text for block in content if isinstance(block, models.TextContent)]
            return '\n\n'.join(texts)

        return ''

    def _make_tool_use_info(
        self,
        block: models.ToolUseContent,
        line_number: int,
        turn_index: int,
    ) -> ToolUseInfo:
        """Create ToolUseInfo from a ToolUseContent block."""
        tool_input = block.input

        # Get typed tool name from the model (no string manipulation)
        tool_name = _get_tool_name(tool_input)

        # Extract file_path using pattern matching
        file_path = _extract_file_path(tool_input)

        # For MCP tools, parse server/operation from the block
        if not isinstance(tool_input, models.MCPToolInput):
            mcp_server: str | None = None
            mcp_operation: str | None = None
        else:
            mcp_server, mcp_operation = block.mcp_info

        return ToolUseInfo(
            line_number=line_number,
            turn_index=turn_index,
            tool_use_id=block.id,
            name=tool_name,
            input=tool_input,
            file_path=file_path,
            mcp_server=mcp_server,
            mcp_operation=mcp_operation,
        )

    def _link_tool_results(self, turns: list[Turn]) -> None:
        """Link tool uses to their results."""
        # Build lookup of tool uses by ID
        uses_by_id: dict[str, ToolUseInfo] = {}
        for turn in turns:
            for use in turn.tool_uses:
                uses_by_id[use.tool_use_id] = use

        # Find results and link them
        for line_num, record in self.iter_records():
            if not isinstance(record, models.UserRecord):
                continue
            if not record.message or not isinstance(record.message.content, list):
                continue

            # Determine turn index for this user record
            turn_index = 0
            for turn in turns:
                if turn.user_line == line_num:
                    turn_index = turn.index
                    break

            for block in record.message.content:
                if not isinstance(block, models.ToolResultContent):
                    continue

                use_or_none = uses_by_id.get(block.tool_use_id)
                if not use_or_none:
                    continue
                use = use_or_none

                # Calculate content length
                content = block.content
                if content is None:
                    content_length = 0
                elif isinstance(content, str):
                    content_length = len(content)
                else:
                    content_length = sum(len(getattr(b, 'text', '')) for b in content)

                result = ToolResultInfo(
                    line_number=line_num,
                    turn_index=turn_index,
                    tool_use_id=block.tool_use_id,
                    content=content,
                    content_length=content_length,
                    is_error=block.is_error or False,
                    invocation=use,
                )

                # Bidirectional link
                use.result = result
