"""Type definitions for session chunking.

Pure data definitions - no logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel

if TYPE_CHECKING:
    from src.schemas.session.models import ToolUseContent


# =============================================================================
# Literals
# =============================================================================

ToolType = Literal[
    # MCP (third-party)
    'mcp',
    # File operations
    'Read',
    'Write',
    'Edit',
    'Glob',
    'Grep',
    'NotebookEdit',
    # Execution
    'Bash',
    'Task',
    'TaskOutput',
    'AgentOutput',
    'BashOutput',
    'KillShell',
    # Web
    'WebSearch',
    'WebFetch',
    # LSP
    'LSP',
    # Planning
    'EnterPlanMode',
    'ExitPlanMode',
    # User interaction
    'AskUserQuestion',
    'TodoWrite',
    # MCP management
    'MCPSearch',
    'ListMcpResources',
    'ReadMcpResource',
    # Skills
    'Skill',
    # Task management
    'TaskCreate',
    'TaskUpdate',
    'TaskList',
]

ToolCategory = Literal[
    'mcp',
    'file_operation',
    'execution',
    'web',
    'lsp',
    'planning',
    'interaction',
    'mcp_management',
    'skill',
    'task_management',
]

RecordType = Literal[
    'user',
    'assistant',
    'summary',
    # System record subtypes
    'local_command',
    'compact_boundary',
    'microcompact_boundary',
    'api_error',
    'informational',
    'turn_duration',
    'stop_hook_summary',
    'system',  # Generic fallback
    # Other records
    'file_history_snapshot',
    'queue_operation',
    'custom_title',
    'progress',
]

ContentType = Literal['text', 'thinking', 'tool_use', 'tool_result']

ToolEvent = Literal['invocation', 'result']

LSPOperation = Literal[
    'goToDefinition',
    'findReferences',
    'hover',
    'documentSymbol',
    'workspaceSymbol',
    'goToImplementation',
    'prepareCallHierarchy',
    'incomingCalls',
    'outgoingCalls',
]


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """Configuration for session chunking.

    Default: Include everything. Use exclude_record_types to filter.
    """

    include_thinking: bool = True
    include_tool_results: bool = True
    min_chunk_chars: int = 0  # No minimum by default - include everything
    max_chunk_chars: int = 2500

    # Record type exclusions (empty = include all)
    # Use RecordType values: 'progress', 'turn_duration', etc.
    exclude_record_types: frozenset[str] = frozenset()


# =============================================================================
# Chunk Context (mutable state during chunking)
# =============================================================================


@dataclass
class ChunkContext:
    """Mutable state tracked during session chunking."""

    session_path: Path
    session_id: str
    project_path: str
    slug: str | None = None
    turn_index: int = 0
    chunk_index: int = 0
    line_number: int = 0
    timestamp: str = ''
    pending_tool_uses: dict[str, ToolUseContent] = field(default_factory=dict)

    def next_chunk_index(self) -> int:
        """Get next chunk index and increment counter."""
        idx = self.chunk_index
        self.chunk_index += 1
        return idx

    @classmethod
    def from_session_path(cls, session_path: Path) -> Self:
        """Create context from session file path."""
        # Extract session_id from filename (e.g., "abc123.jsonl" -> "abc123")
        session_id = session_path.stem

        # Extract project_path from parent directory name
        # e.g., "-Users-chris-project" -> "/Users/chris/project"
        normalized = session_path.parent.name
        if normalized.startswith('-'):
            project_path = '/' + normalized[1:].replace('-', '/')
        else:
            project_path = normalized

        return cls(
            session_path=session_path,
            session_id=session_id,
            project_path=project_path,
        )


# =============================================================================
# Chunk Metadata
# =============================================================================


class SessionChunkMetadata(BaseModel):
    """Metadata for a session chunk, optimized for Qdrant filtering."""

    model_config = {'frozen': True, 'extra': 'forbid'}

    # Position
    line_number: int
    turn_index: int
    chunk_index: int
    timestamp: str

    # Record structure
    record_type: RecordType
    content_type: ContentType

    # Tool identity (when content_type is 'tool_use' or 'tool_result')
    tool_type: ToolType | None = None
    tool_category: ToolCategory | None = None

    # Tool event (invocation vs result)
    tool_event: ToolEvent | None = None
    tool_use_id: str | None = None  # Links invocation <-> result
    tool_result_error: bool | None = None  # For results only

    # MCP-specific (indexed in Qdrant)
    mcp_server: str | None = None
    mcp_operation: str | None = None

    # LSP-specific (indexed in Qdrant)
    lsp_operation: LSPOperation | None = None

    # File-specific (NOT indexed - high cardinality)
    file_path: str | None = None

    # Session identity
    session_id: str
    slug: str | None = None
    project_path: str

    # Model (for assistant records)
    model: str | None = None

    @classmethod
    def for_text(
        cls,
        record_type: RecordType,
        ctx: ChunkContext,
        model: str | None = None,
    ) -> Self:
        """Create metadata for a text chunk."""
        return cls(
            record_type=record_type,
            content_type='text',
            line_number=ctx.line_number,
            turn_index=ctx.turn_index,
            chunk_index=ctx.next_chunk_index(),
            timestamp=ctx.timestamp,
            session_id=ctx.session_id,
            slug=ctx.slug,
            project_path=ctx.project_path,
            model=model,
        )

    @classmethod
    def for_thinking(
        cls,
        ctx: ChunkContext,
        model: str | None = None,
    ) -> Self:
        """Create metadata for a thinking chunk."""
        return cls(
            record_type='assistant',
            content_type='thinking',
            line_number=ctx.line_number,
            turn_index=ctx.turn_index,
            chunk_index=ctx.next_chunk_index(),
            timestamp=ctx.timestamp,
            session_id=ctx.session_id,
            slug=ctx.slug,
            project_path=ctx.project_path,
            model=model,
        )


# =============================================================================
# Session Chunk
# =============================================================================


class SessionChunk(BaseModel):
    """A searchable chunk from a Claude Code session."""

    model_config = {'frozen': True, 'extra': 'forbid'}

    text: str  # The searchable content
    source_path: str  # Path to JSONL file
    metadata: SessionChunkMetadata
