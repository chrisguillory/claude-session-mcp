"""
Pydantic models for Claude Code session JSONL records.

This module defines strict types for all record types found in Claude Code session files.
Uses discriminated unions for type-safe parsing of heterogeneous JSONL data.

CLAUDE CODE VERSION COMPATIBILITY:
- Validated against: Claude Code 2.0.35 - 2.0.65
- Last validated: 2025-12-15
- Validation coverage: 157,640 records across 1,039 session files
- Schema v0.1.1: Added todos field for Claude Code 2.0.47+
- Schema v0.1.2: Added error field to AssistantRecord for Claude Code 2.0.49+
- Schema v0.1.3: Added slug, DocumentContent, ContextManagement, SkillToolInput,
                 image/jpeg support, ECONNRESET error code for Claude Code 2.0.51+
- Schema v0.1.4: Added EnterPlanMode tool, slug to CompactBoundarySystemRecord for 2.0.62+
- Schema v0.1.5: Added AgentOutputTool input model for Claude Code 2.0.64+
- Schema v0.1.6: Added TaskOutput tool input model for Claude Code 2.0.65+
- Schema v0.1.7: Added model_context_window_exceeded stop_reason for context overflow handling
- Schema v0.1.8: Added sourceToolUseID, EmptyError, BeforeValidator for ultrathink case normalization (2.0.76+)
- Schema v0.1.9: Added CustomTitleRecord for user-defined session names
- If validation fails, Claude Code schema may have changed - update models accordingly

NEW FIELDS IN CLAUDE CODE 2.0.51+ (Schema v0.1.3):

slug field:
  Human-readable conversation identifier using adjective-verb-animal format
  (e.g., "jiggly-churning-rabbit", "floofy-singing-crab").
  - NOT a permanent session identifier - can change within same session file
  - Regenerated when session is resumed after extended inactivity (~hours)
  - Appears on: UserRecord, AssistantRecord, LocalCommandSystemRecord, ApiErrorSystemRecord
  - Some records have slug: null during transitions
  - Purpose: Makes sessions easier to reference than UUIDs
  - Related GitHub issues: #2112, #10943, #11408 (custom session naming requests)

context_management field:
  Tracks server-side context optimization operations on Message objects.
  Structure: {"applied_edits": []} - records which context clearing operations were executed.
  - Part of Claude API's context editing feature (Nov 2025)
  - Automatically clears older tool results when context exceeds thresholds
  - Helps maintain long-running sessions without context window exhaustion
  - Previously always null, now populated during context optimization

document content type:
  New MessageContent type for PDF uploads with structure:
  {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "..."}}
  - Part of Claude API's PDF support, not Claude Code specific
  - Enables visual + textual understanding of PDF documents
  - Max 32MB, 100 pages per request

Skill tool:
  New tool for invoking Agent Skills - auto-discovered capabilities.
  Input: {"command": "canvas-design"} or {"skill": "..."}
  - Skills differ from slash commands: auto-discovered vs explicit invocation
  - Skills are folders with SKILL.md + resources (templates, scripts, examples)
  - Claude examines available skills and uses them when relevant to task
  - Built-in skills: Excel, PowerPoint, Word, PDF form-filling

Key findings from analyzing real session files:
- Assistant records from agents/subprocesses have nested API response structure
- file-history-snapshot records don't inherit from BaseRecord (different schema)
- summary records don't have uuid/timestamp/sessionId (minimal schema)
- User records can have optional toolUseResult field with tool execution metadata
- Message structure varies: can be string or structured content list

Important pattern for unused/reserved fields:
- Fields that are present in JSON but ALWAYS null are typed as None (not Any | None)
- This includes: AssistantRecord.stopReason, Message.container
- These are reserved/future fields in the Claude API schema but currently unpopulated
- Using None type prevents accidental usage and makes the schema more accurate

Round-trip serialization:
- Use model_dump(exclude_unset=True, mode='json') to preserve original JSON structure
- This maintains null fields from input while excluding fields that were never set
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, TypeVar

import pydantic

from src.schemas.session.markers import PathField, PathListField
from src.schemas.types import BaseStrictModel, ModelId

# ==============================================================================
# Schema Version
# ==============================================================================

SCHEMA_VERSION = '0.1.9'
CLAUDE_CODE_MIN_VERSION = '2.0.35'
CLAUDE_CODE_MAX_VERSION = '2.0.76'
LAST_VALIDATED = '2025-12-29'
VALIDATION_RECORD_COUNT = 98_971


# ==============================================================================
# Base Configuration
# ==============================================================================


class StrictModel(BaseStrictModel):
    """Session-layer strict model.

    Inherits from BaseStrictModel (extra='forbid', strict=True, frozen=True).
    Domain-specific customization can be added here if needed.
    """

    pass


# Type variable for validated_copy function
T = TypeVar('T', bound=pydantic.BaseModel)


def validated_copy[T: pydantic.BaseModel](model: T, update: Mapping[str, Any]) -> T:
    """
    Create a validated copy of a model with updates.

    This is the type-safe way to modify frozen Pydantic models.
    The returned instance is validated through the model's __init__.

    Args:
        model: The source model instance
        update: Fields to update (keys must be valid field names)

    Returns:
        A new validated instance of the same type with updates applied

    Example:
        updated_record = validated_copy(record, {'sessionId': new_session_id})
    """
    model_class = type(model)
    new_data = model.model_dump()
    new_data.update(update)
    return model_class(**new_data)


# ==============================================================================
# Message Content Types (Discriminated Union)
# ==============================================================================


class ThinkingContent(StrictModel):
    """Thinking content block from assistant messages."""

    type: Literal['thinking']
    thinking: str
    signature: str  # Always non-null in observed data (31,105/31,105)


class TextContent(StrictModel):
    """Text content block from user or assistant messages."""

    type: Literal['text']
    text: str


# ==============================================================================
# Tool Use Input Types (for tools that use file paths)
# ==============================================================================


class ReadToolInput(StrictModel):
    """Input for Read tool."""

    file_path: PathField


class WriteToolInput(StrictModel):
    """Input for Write tool."""

    file_path: PathField
    content: str


class EditToolInput(StrictModel):
    """Input for Edit tool."""

    file_path: PathField
    old_string: str
    new_string: str
    replace_all: bool | None = None


class SkillToolInput(StrictModel):
    """Input for Skill tool."""

    command: str  # Skill name to invoke (e.g., 'canvas-design')


class EnterPlanModeToolInput(StrictModel):
    """Input for EnterPlanMode tool (no parameters)."""

    pass


class AgentOutputToolInput(StrictModel):
    """Input for AgentOutputTool - retrieves output from background agents."""

    agentId: str  # Agent ID to retrieve output from
    block: bool | None = None  # Whether to block waiting for agent completion
    wait_up_to: int | None = None  # Optional timeout in seconds


class TaskOutputToolInput(StrictModel):
    """Input for TaskOutput tool - retrieves output from running/completed tasks."""

    task_id: str  # Task ID to retrieve output from
    block: bool | None = None  # Whether to block waiting for task completion
    timeout: int | None = None  # Optional timeout in milliseconds


# Union of tool inputs (typed models first, dict fallback for MCP tools)
# NOTE: Order matters! More specific (more required fields) should come first.
# EnterPlanModeToolInput is empty (no fields), so it must come last before dict fallback.
ToolInput = Annotated[
    ReadToolInput  # file_path required
    | WriteToolInput  # file_path, content required
    | EditToolInput  # file_path, old_string, new_string required
    | AgentOutputToolInput  # agentId required
    | TaskOutputToolInput  # task_id required
    | SkillToolInput  # command required
    | EnterPlanModeToolInput  # No required fields - must be last before dict!
    | dict[str, Any],  # Fallback for MCP tools
    pydantic.Field(union_mode='left_to_right'),
]


# ==============================================================================
# Image Source (must be defined before ImageContent)
# ==============================================================================


class ImageSource(StrictModel):
    """Image source data for image content."""

    type: Literal['base64']
    media_type: Literal['image/jpeg', 'image/png']  # Only value observed across all sessions
    data: str  # Base64 encoded image data


class ImageContent(StrictModel):
    """Image content block from user messages."""

    type: Literal['image']
    source: ImageSource


class DocumentSource(StrictModel):
    """Document source data for document content (PDFs, etc.)."""

    type: Literal['base64']
    media_type: str  # e.g., 'application/pdf'
    data: str  # Base64 encoded document data


class DocumentContent(StrictModel):
    """Document content block from user messages (PDF uploads, etc.)."""

    type: Literal['document']
    source: DocumentSource


class ToolUseContent(StrictModel):
    """Tool use content block from assistant messages."""

    type: Literal['tool_use']
    id: str
    name: str
    input: ToolInput  # Typed for Read/Write/Edit, dict for MCP tools only

    @pydantic.field_validator('input', mode='after')
    @classmethod
    def validate_mcp_tool_fallback(cls, v: ToolInput, info: pydantic.ValidationInfo) -> ToolInput:
        """
        Enforce that only MCP tools (starting with 'mcp__') can use dict fallback.
        All Claude Code built-in tools must be explicitly modeled.

        Uses ALLOWED_CLAUDE_TOOL_NAMES which is derived from MODELED_CLAUDE_TOOLS,
        keeping this validator coupled to the actual model classes.
        """
        # Only validate if input is a plain dict (not one of our typed model instances)
        if isinstance(v, dict):
            # Check if it's a typed model (Read/Write/Edit have file_path)
            is_typed_model = 'file_path' in v or 'old_string' in v

            if not is_typed_model:
                # It's using the dict fallback - get tool name from validation context
                tool_name = info.data.get('name', '')

                if tool_name not in ALLOWED_CLAUDE_TOOL_NAMES and not tool_name.startswith('mcp__'):
                    raise ValueError(
                        f"Unmodeled Claude Code built-in tool: '{tool_name}'. "
                        f'All Claude Code tools must be explicitly modeled (see MODELED_CLAUDE_TOOLS). '
                        f"Only MCP tools (starting with 'mcp__') may use the dict fallback."
                    )

        return v


# ToolResultContentBlock - for content inside tool_result
ToolResultContentBlock = Annotated[TextContent | ImageContent, pydantic.Field(discriminator='type')]


class ToolResultContent(StrictModel):
    """Tool result content block from user messages."""

    type: Literal['tool_result']
    tool_use_id: str
    content: str | Sequence[ToolResultContentBlock] | None = (
        None  # Can be string, sequence of content blocks, or missing
    )
    is_error: bool | None = None


# Discriminated union of all message content types
MessageContent = Annotated[
    ThinkingContent | TextContent | ToolUseContent | ToolResultContent | ImageContent | DocumentContent,
    pydantic.Field(discriminator='type'),
]


# ==============================================================================
# Context Management (Claude Code 2.0.51+)
# ==============================================================================


class ClearThinkingEdit(StrictModel):
    """Applied context edit for clearing thinking blocks."""

    type: Literal['clear_thinking_20251015']
    cleared_thinking_turns: int
    cleared_input_tokens: int


# Union of all applied edit types (add new types here as discovered)
AppliedEdit = ClearThinkingEdit


class ContextManagement(StrictModel):
    """Context management metadata for message responses (Claude Code 2.0.51+)."""

    applied_edits: Sequence[AppliedEdit]  # Can be empty or contain edit records


# ==============================================================================
# Message Structure
# ==============================================================================


class Message(StrictModel):
    """A message within a record."""

    role: Literal['user', 'assistant']
    content: Sequence[MessageContent] | str
    # Additional fields that may appear in assistant messages (nested API response)
    type: Literal['message'] | None = pydantic.Field(
        None, description='Message type indicator (present in agent/subprocess responses)'
    )
    model: ModelId | None = pydantic.Field(
        None, description='Claude model identifier (e.g., claude-sonnet-4-5-20250929)'
    )
    id: str | None = pydantic.Field(None, description='Message ID from Claude API')
    stop_reason: Literal['tool_use', 'stop_sequence', 'end_turn', 'refusal', 'model_context_window_exceeded'] | None = (
        pydantic.Field(None, description='Reason why the model stopped generating')
    )
    stop_sequence: str | None = pydantic.Field(
        None, description='The actual stop sequence string that triggered stopping'
    )
    usage: TokenUsage | None = pydantic.Field(
        None, description='Token usage information (present in nested API responses)'
    )
    container: None = pydantic.Field(
        None, description='Reserved for future use', json_schema_extra={'status': 'reserved'}
    )
    context_management: ContextManagement | None = pydantic.Field(
        None, description='Context management metadata (Claude Code 2.0.51+)'
    )


# ==============================================================================
# Token Usage
# ==============================================================================


class CacheCreation(StrictModel):
    """Cache creation token breakdown."""

    ephemeral_5m_input_tokens: int
    ephemeral_1h_input_tokens: int


class ServerToolUse(StrictModel):
    """Server-side tool use tracking."""

    web_search_requests: int
    web_fetch_requests: int  # Always present (553/553)


class TokenUsage(StrictModel):
    """Token usage information for assistant messages."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int  # Always present (115,497/115,497)
    cache_read_input_tokens: int  # Always present (115,497/115,497)
    cache_creation: CacheCreation  # Always present (115,497/115,497)
    service_tier: Literal['standard'] | None = None  # Only value: 'standard' (19018 occurrences) - null for synthetic
    server_tool_use: ServerToolUse | None = None  # Server-side tool use tracking (0.5% present)


# ==============================================================================
# Thinking Metadata
# ==============================================================================


def _normalize_ultrathink(v: Any) -> Any:
    """Normalize any casing of 'ultrathink' to lowercase."""
    if isinstance(v, str) and v.lower() == 'ultrathink':
        return 'ultrathink'
    return v  # Let Literal validation fail with original value


class ThinkingTrigger(StrictModel):
    """A thinking trigger with position information."""

    start: int
    end: int
    text: Annotated[
        Literal['ultrathink'], pydantic.BeforeValidator(_normalize_ultrathink)
    ]  # Any casing normalized to lowercase


class ThinkingMetadata(StrictModel):
    """Thinking configuration metadata."""

    level: Literal['none', 'low', 'medium', 'high']  # Strict validation
    disabled: bool
    triggers: Sequence[str | ThinkingTrigger]  # Can be strings or trigger objects


# ==============================================================================
# Todo Item
# ==============================================================================


class TodoItem(StrictModel):
    """A single todo item from TodoWrite tool."""

    content: str
    status: Literal['pending', 'in_progress', 'completed']
    activeForm: str


# ==============================================================================
# Compact Metadata
# ==============================================================================


class CompactMetadata(StrictModel):
    """Metadata for conversation compaction."""

    trigger: Literal['auto', 'manual']  # auto=24, manual=18 across all sessions
    preTokens: int


# ==============================================================================
# API Error
# ==============================================================================


class ApiErrorDetail(StrictModel):
    """Nested API error details."""

    type: Literal['overloaded_error']  # Only value observed across all sessions
    message: str


class ApiErrorResponse(StrictModel):
    """API error response structure."""

    type: Literal['error']
    error: ApiErrorDetail
    request_id: str | None = None


class ApiError(StrictModel):
    """Complete API error information."""

    status: int
    headers: Mapping[str, str]
    requestID: str | None = None  # Can be null for some errors
    error: ApiErrorResponse | None = None  # Can be missing for some errors (e.g., 503)


class ConnectionError(StrictModel):
    """Connection error details for network failures."""

    code: Literal['ConnectionRefused', 'ECONNRESET', 'FailedToOpenSocket']
    path: str  # URL that failed (e.g., "https://api.anthropic.com/v1/messages?beta=true")
    errno: int


class NetworkError(StrictModel):
    """Network error wrapper (for connection failures)."""

    cause: ConnectionError


class EmptyError(StrictModel):
    """Empty error object for unknown/unspecified API errors (Claude Code 2.0.76+)."""

    pass


# ==============================================================================
# File Info
# ==============================================================================


class FileInfo(StrictModel):
    """File information from Read tool."""

    filePath: PathField
    content: str
    numLines: int
    startLine: int
    totalLines: int


# ==============================================================================
# Structured Patch
# ==============================================================================


class PatchHunk(StrictModel):
    """A single hunk in a git-style patch."""

    oldStart: int
    oldLines: int
    newStart: int
    newLines: int
    lines: Sequence[str]


# ==============================================================================
# Tool Use Result Structures
# ==============================================================================


class BashToolResult(StrictModel):
    """Result from Bash tool execution."""

    stdout: str
    stderr: str
    interrupted: bool
    isImage: bool
    returnCodeInterpretation: (
        Literal['No matches found', 'Some directories were inaccessible', 'Files differ'] | None
    ) = None
    backgroundTaskId: str | None = None
    shellId: str | None = None
    command: str | None = None
    exitCode: int | None = None
    stdoutLines: int | None = None
    stderrLines: int | None = None
    timestamp: str | None = None
    status: Literal['running', 'completed', 'failed'] | None = None
    filterPattern: str | None = None


class ReadToolResult(StrictModel):
    """Result from Read tool execution."""

    type: Literal['text']
    file: FileInfo


class GlobToolResult(StrictModel):
    """Result from Glob/file search tool execution."""

    mode: Literal['files_with_matches'] | None = None
    filenames: Sequence[str]
    numFiles: int
    durationMs: int | None = None
    truncated: bool | None = None


class GrepToolResult(StrictModel):
    """Result from Grep/content search tool execution."""

    mode: Literal['content', 'count']
    numFiles: int
    filenames: Sequence[str]
    content: str | None = None
    numLines: int | None = None
    numMatches: int | None = None
    appliedLimit: int | None = None


class EditToolResult(StrictModel):
    """Result from Edit tool execution."""

    filePath: PathField
    oldString: str
    newString: str
    originalFile: str
    userModified: bool
    replaceAll: bool
    structuredPatch: Sequence[PatchHunk]


class WriteToolResult(StrictModel):
    """Result from Write tool execution."""

    type: Literal['create', 'update']
    filePath: PathField
    content: str
    structuredPatch: Sequence[PatchHunk] | None = None


class TodoToolResult(StrictModel):
    """Result from TodoWrite tool execution."""

    oldTodos: Sequence[TodoItem]
    newTodos: Sequence[TodoItem]


class TaskToolResult(StrictModel):
    """Result from Task/agent tool execution."""

    status: Literal['completed']
    prompt: str
    content: Sequence[MessageContent]
    totalDurationMs: int
    totalTokens: int
    totalToolUseCount: int
    usage: TokenUsage
    agentId: str  # Always present (621/621)


# ==============================================================================
# AskUserQuestion Structures
# ==============================================================================


class QuestionOption(StrictModel):
    """A single option in a user question."""

    label: str
    description: str


class UserQuestion(StrictModel):
    """A question to ask the user."""

    question: str
    header: str
    options: Sequence[QuestionOption]
    multiSelect: bool


class AskUserQuestionToolResult(StrictModel):
    """Result from AskUserQuestion tool execution."""

    questions: Sequence[UserQuestion]
    answers: Sequence[str]  # List of question text that was answered


# ==============================================================================
# WebSearch Structures
# ==============================================================================


class WebSearchResult(StrictModel):
    """A single web search result."""

    title: str
    url: str


class WebSearchToolResult(StrictModel):
    """Result from WebSearch tool execution."""

    query: str
    results: Sequence[WebSearchResult] | str  # Can be list of results or string explanation
    durationSeconds: int


class WebFetchToolResult(StrictModel):
    """Result from WebFetch tool execution."""

    url: str
    code: int
    codeText: str
    bytes: int
    result: str
    durationMs: int


class ExitPlanModeToolResult(StrictModel):
    """Result from ExitPlanMode tool execution."""

    plan: str
    isAgent: bool


class McpResource(StrictModel):
    """A single MCP resource from ListMcpResourcesTool."""

    name: str
    title: str | None = None  # Optional - some resources don't have title
    uri: str
    description: str
    mimeType: str
    server: str


# NOTE: ListMcpResourcesTool returns a bare list[McpResource] (not wrapped in a dict)
# This is handled by UserRecord.toolUseResult: list[McpResource] variant


class KillShellToolResult(StrictModel):
    """Result from KillShell tool execution."""

    success: bool
    shellId: str


# NOTE: BashOutput tool uses BashToolResult (same structure)

# Union of all tool result types (validated left-to-right, most specific first)
ToolUseResultUnion = (
    BashToolResult  # Also handles BashOutput
    | ReadToolResult
    | EditToolResult
    | WriteToolResult
    | GrepToolResult
    | GlobToolResult
    | TodoToolResult
    | TaskToolResult
    | AskUserQuestionToolResult
    | WebSearchToolResult
    | WebFetchToolResult
    | ExitPlanModeToolResult
    | KillShellToolResult
    | dict[str, Any]  # Fallback for MCP tools (64 different tools)
)

# ==============================================================================
# Base Record
# ==============================================================================


class BaseRecord(StrictModel):
    """Base class for all session record types."""

    type: str
    uuid: str
    timestamp: str
    sessionId: str


# ==============================================================================
# User Record
# ==============================================================================


class UserRecord(BaseRecord):
    """User message record."""

    type: Literal['user']
    cwd: PathField
    parentUuid: str | None
    isSidechain: bool
    userType: Literal['external']
    version: str
    gitBranch: str
    message: Message
    projectPaths: PathListField | None = pydantic.Field(
        None, description='Additional project paths beyond cwd (each path will be translated)'
    )
    budgetTokens: int | None = pydantic.Field(None, description='Token budget limit for this request')
    skills: None = pydantic.Field(None, description='Reserved for future use', json_schema_extra={'status': 'reserved'})
    mcp: None = pydantic.Field(None, description='Reserved for future use', json_schema_extra={'status': 'reserved'})
    agentId: str | None = pydantic.Field(
        None, description='Agent ID for subprocess/agent records (references agent-{agentId}.jsonl)'
    )
    isMeta: bool | None = pydantic.Field(None, description='Indicates meta messages (system-level information)')
    thinkingMetadata: ThinkingMetadata | None = pydantic.Field(
        None, description='Extended thinking configuration (Claude 3.7+)'
    )
    isVisibleInTranscriptOnly: bool | None = pydantic.Field(
        None, description='Message visible only in transcript, not in session history'
    )
    isCompactSummary: bool | None = pydantic.Field(None, description='Indicates this is a compacted session summary')
    toolUseResult: Annotated[
        Sequence[ToolResultContentBlock]  # TextContent/ImageContent with 'type' discriminator - must come first
        | Sequence[McpResource]  # MCP resources (no 'type' field, has 'name', 'uri', etc.)
        | ToolUseResultUnion
        | str
        | None,
        pydantic.Field(union_mode='left_to_right'),
    ] = None  # Tool execution metadata (validated left-to-right)
    todos: Sequence[TodoItem] | None = pydantic.Field(None, description='Todo list state (Claude Code 2.0.47+)')
    slug: str | None = pydantic.Field(None, description='Human-readable session slug (Claude Code 2.0.51+)')
    imagePasteIds: Sequence[int] | None = pydantic.Field(None, description='IDs of pasted images in this message')
    sourceToolUseID: str | None = pydantic.Field(
        None,
        description='Tool use ID that generated this message (e.g., toolu_015eZkLKZz5JVkGC1zZrnnKm) (Claude Code 2.0.76+)',
    )


# ==============================================================================
# Assistant Record
# ==============================================================================


class AssistantRecord(BaseRecord):
    """Assistant message record."""

    type: Literal['assistant']
    cwd: PathField
    parentUuid: str | None = pydantic.Field(..., description='UUID of parent record (null for root agent records)')
    message: Message
    # Note: usage/stopReason are optional for agent records (nested in message instead)
    usage: TokenUsage | None = pydantic.Field(
        None, description='Token usage for this request (null for agent records - usage in message instead)'
    )
    stopReason: None = pydantic.Field(
        None, description='Reserved for future use', json_schema_extra={'status': 'reserved'}
    )
    model: ModelId | None = pydantic.Field(
        None, description='Claude model identifier (null for agent records - model in message instead)'
    )
    requestDuration: int | None = pydantic.Field(None, description='Request duration in milliseconds')
    requestId: str | None = pydantic.Field(None, description='Claude API request ID')
    agentId: str | None = pydantic.Field(
        None, description='Agent ID for subprocess/agent records (references agent-{agentId}.jsonl)'
    )
    isSidechain: bool | None = pydantic.Field(
        None, description='Indicates sidechain/subprocess execution (present in agent records)'
    )
    userType: str | None = pydantic.Field(None, description='User type (present in agent records)')
    version: str | None = pydantic.Field(None, description='Claude Code version (present in agent records)')
    gitBranch: str | None = pydantic.Field(None, description='Git branch (present in agent records)')
    isApiErrorMessage: bool | None = pydantic.Field(None, description='Indicates this message represents an API error')
    error: Literal['rate_limit', 'unknown', 'invalid_request', 'authentication_failed'] | None = pydantic.Field(
        None, description='Error type for API error messages'
    )
    slug: str | None = pydantic.Field(None, description='Human-readable session slug (Claude Code 2.0.51+)')


# ==============================================================================
# Summary Record (does NOT inherit from BaseRecord - different schema)
# ==============================================================================


class SummaryRecord(StrictModel):
    """Session summary record (minimal schema, no uuid/timestamp)."""

    type: Literal['summary']
    summary: str
    leafUuid: str


# ==============================================================================
# System Record
# ==============================================================================


class SystemRecord(BaseRecord):
    """System message record (standard system messages)."""

    type: Literal['system']
    cwd: PathField
    parentUuid: str | None
    systemType: str
    message: str  # Always string (0 dict occurrences across all sessions)


# ==============================================================================
# System Subtype Records (discriminated by subtype field)
# ==============================================================================


class LocalCommandSystemRecord(BaseRecord):
    """System record for local command output (subtype=local_command)."""

    type: Literal['system']
    cwd: PathField
    parentUuid: str | None
    subtype: Literal['local_command']
    content: str  # Command output XML
    level: Literal['info', 'error', 'warning', 'suggestion'] | None = None
    isMeta: bool
    isSidechain: bool
    userType: str
    version: str
    gitBranch: str
    slug: str | None = pydantic.Field(None, description='Human-readable session slug (Claude Code 2.0.51+)')


class CompactBoundarySystemRecord(BaseRecord):
    """System record for conversation compaction (subtype=compact_boundary)."""

    type: Literal['system']
    cwd: PathField
    parentUuid: str | None
    subtype: Literal['compact_boundary']
    content: str  # e.g., "Conversation compacted"
    level: Literal['info', 'error', 'warning', 'suggestion'] | None = None
    isMeta: bool
    isSidechain: bool
    userType: str
    version: str
    gitBranch: str
    slug: str | None = pydantic.Field(None, description='Human-readable session slug (Claude Code 2.0.51+)')
    logicalParentUuid: str | None = None
    compactMetadata: CompactMetadata | None = None


class ApiErrorSystemRecord(BaseRecord):
    """System record for API errors (subtype=api_error)."""

    type: Literal['system']
    cwd: PathField
    parentUuid: str | None
    subtype: Literal['api_error']
    level: Literal['error', 'warning'] | None = None
    isSidechain: bool | None = None  # Optional for api_error
    userType: str | None = None  # Optional for api_error
    version: str | None = None  # Optional for api_error
    gitBranch: str | None = None  # Optional for api_error
    slug: str | None = pydantic.Field(None, description='Human-readable session slug (Claude Code 2.0.51+)')
    cause: ConnectionError | None = None  # Connection error details (for network failures)
    error: (
        ApiError | NetworkError | EmptyError
    )  # API error, network error, or empty error (EmptyError must be last - no required fields)
    retryInMs: float
    retryAttempt: int
    maxRetries: int


class InformationalSystemRecord(BaseRecord):
    """System record for informational messages (subtype=informational)."""

    type: Literal['system']
    cwd: PathField
    parentUuid: str | None
    subtype: Literal['informational']
    content: str | None = None
    level: Literal['info', 'error', 'warning', 'suggestion'] | None = None
    isMeta: bool | None = None
    isSidechain: bool | None = None
    userType: str | None = None
    version: str | None = None
    gitBranch: str | None = None


# Union of system subtype records
SystemSubtypeRecord = Annotated[
    LocalCommandSystemRecord | CompactBoundarySystemRecord | ApiErrorSystemRecord | InformationalSystemRecord,
    pydantic.Field(discriminator='subtype'),
]


# ==============================================================================
# File History Snapshot Record (does NOT inherit from BaseRecord - different schema)
# ==============================================================================


class FileBackupInfo(StrictModel):
    """Backup information for a tracked file."""

    backupFileName: str | None  # Can be null for first version
    version: int
    backupTime: str


class FileHistorySnapshot(StrictModel):
    """Snapshot data for file history tracking."""

    messageId: str
    trackedFileBackups: Mapping[str, FileBackupInfo]  # Map of file paths to backup data
    timestamp: str


class FileHistorySnapshotRecord(StrictModel):
    """File history snapshot record (different schema from regular records)."""

    type: Literal['file-history-snapshot']
    messageId: str
    snapshot: FileHistorySnapshot
    isSnapshotUpdate: bool


# ==============================================================================
# Queue Operation Record (does NOT inherit from BaseRecord - no uuid field)
# ==============================================================================


class QueueOperationRecord(StrictModel):
    """Queue operation record (minimal schema, no uuid)."""

    type: Literal['queue-operation']
    operation: Literal['enqueue', 'dequeue', 'remove', 'popAll']  # Queue operation type
    timestamp: str
    sessionId: str
    content: str | Sequence[MessageContent] | None = pydantic.Field(
        None, description='User input content for the queued operation (string or structured message)'
    )
    data: None = pydantic.Field(None, description='Reserved for future use', json_schema_extra={'status': 'reserved'})


# ==============================================================================
# Custom Title Record (does NOT inherit from BaseRecord - minimal schema)
# ==============================================================================


class CustomTitleRecord(StrictModel):
    """Custom title record for user-defined session names (minimal schema, no uuid)."""

    type: Literal['custom-title']
    customTitle: str  # User-defined session title
    sessionId: str


# ==============================================================================
# Session Record (Discriminated Union)
# ==============================================================================

# Union of all record types (validated left-to-right)
# NOTE: Cannot use discriminator='type' because multiple records have type='system'
# System subtype records must come before SystemRecord so they match first
SessionRecord = Annotated[
    UserRecord
    | AssistantRecord
    | SummaryRecord
    | LocalCommandSystemRecord  # Must be before SystemRecord!
    | CompactBoundarySystemRecord  # Must be before SystemRecord!
    | ApiErrorSystemRecord  # Must be before SystemRecord!
    | InformationalSystemRecord  # Must be before SystemRecord!
    | SystemRecord
    | FileHistorySnapshotRecord
    | QueueOperationRecord
    | CustomTitleRecord,
    pydantic.Field(union_mode='left_to_right'),
]

# Type adapter for validating session records (required for union types)
SessionRecordAdapter: pydantic.TypeAdapter[SessionRecord] = pydantic.TypeAdapter(SessionRecord)


# ==============================================================================
# Session Metadata
# ==============================================================================


class SessionMetadata(StrictModel):
    """Metadata extracted from a session."""

    session_id: str
    record_count: int
    first_timestamp: str
    last_timestamp: str
    unique_cwds: Sequence[str]
    unique_project_paths: Sequence[str]
    user_message_count: int
    assistant_message_count: int
    summary_count: int
    system_message_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_cache_creation_tokens: int
    total_cache_read_tokens: int
    models_used: Sequence[str]
    tools_used: Sequence[str]
    files_touched: Sequence[str]
    git_branches: Sequence[str]


# ==============================================================================
# Analysis Results
# ==============================================================================


class SessionAnalysis(StrictModel):
    """Complete analysis of a session."""

    metadata: SessionMetadata
    summary_text: str | None = None
    cost_estimate_usd: float | None = None
    duration_seconds: float | None = None


# ==============================================================================
# Allowed Claude Tool Names (defined after all classes)
# ==============================================================================

# Mapping of modeled tool result classes to their tool names
# This keeps the validator coupled to the actual model classes
# NOTE: Using list of tuples because some classes map to multiple tool names
MODELED_CLAUDE_TOOLS = [
    (BashToolResult, 'Bash'),
    (BashToolResult, 'BashOutput'),  # Same result structure as Bash
    (ReadToolResult, 'Read'),
    (EditToolResult, 'Edit'),
    (WriteToolResult, 'Write'),
    (GrepToolResult, 'Grep'),
    (GlobToolResult, 'Glob'),
    (TodoToolResult, 'TodoWrite'),
    (TaskToolResult, 'Task'),
    (AskUserQuestionToolResult, 'AskUserQuestion'),
    (WebSearchToolResult, 'WebSearch'),
    (WebFetchToolResult, 'WebFetch'),
    (ExitPlanModeToolResult, 'ExitPlanMode'),
    (EnterPlanModeToolInput, 'EnterPlanMode'),  # No-param tool input
    (KillShellToolResult, 'KillShell'),
    (McpResource, 'ListMcpResourcesTool'),  # Returns list[McpResource], not dict result
    (SkillToolInput, 'Skill'),  # Input typed
    (AgentOutputToolInput, 'AgentOutputTool'),  # Input typed
    (TaskOutputToolInput, 'TaskOutput'),  # Input typed - uses task_id, timeout (ms)
]

# Allowed tool names (extracted from mapping)
ALLOWED_CLAUDE_TOOL_NAMES = {tool_name for _, tool_name in MODELED_CLAUDE_TOOLS}
