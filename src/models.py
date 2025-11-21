"""
Pydantic models for Claude Code session JSONL records.

This module defines strict types for all record types found in Claude Code session files.
Uses discriminated unions for type-safe parsing of heterogeneous JSONL data.

CLAUDE CODE VERSION COMPATIBILITY:
- Validated against: Claude Code 2.0.35 - 2.0.47
- Last validated: 2025-11-20
- Validation coverage: 74,744 records across 981 session files
- Added todos field support in schema v0.1.1 for Claude Code 2.0.47+
- If validation fails, Claude Code schema may have changed - update models accordingly

Key findings from analyzing real session files:
- Assistant records from agents/subprocesses have nested API response structure
- file-history-snapshot records don't inherit from BaseRecord (different schema)
- summary records don't have uuid/timestamp/sessionId (minimal schema)
- User records can have optional toolUseResult field with tool execution metadata
- Message structure varies: can be string or structured content list

Important pattern for unused/reserved fields:
- Fields that are present in JSON but ALWAYS null are typed as None (not Any | None)
- This includes: AssistantRecord.stopReason, Message.container, Message.context_management
- These are reserved/future fields in the Claude API schema but currently unpopulated
- Using None type prevents accidental usage and makes the schema more accurate

Round-trip serialization:
- Use model_dump(exclude_unset=True, mode='json') to preserve original JSON structure
- This maintains null fields from input while excluding fields that were never set
"""

from __future__ import annotations

import functools
from typing import Any, Literal, Annotated, Union, get_args
from pydantic import BaseModel, Field, ConfigDict, TypeAdapter, field_validator, ValidationInfo

from src.markers import PathField, PathListField


# ==============================================================================
# Schema Version
# ==============================================================================

SCHEMA_VERSION = '0.1.1'
CLAUDE_CODE_MIN_VERSION = '2.0.35'
CLAUDE_CODE_MAX_VERSION = '2.0.47'
LAST_VALIDATED = '2025-11-20'
VALIDATION_RECORD_COUNT = 74_744


# ==============================================================================
# Base Configuration
# ==============================================================================


class StrictModel(BaseModel):
    """Base model with strict validation settings."""

    model_config = ConfigDict(
        extra='forbid',  # Raise error on unexpected fields
        strict=True,  # Strict type validation
        frozen=False,  # Allow modification for path translation
    )


# ==============================================================================
# Message Content Types (Discriminated Union)
# ==============================================================================


class ThinkingContent(StrictModel):
    """Thinking content block from assistant messages."""

    type: Literal['thinking']
    thinking: str
    signature: str | None = None


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


# Union of tool inputs with file paths (others remain dict[str, Any])
ToolInput = Annotated[
    Union[
        ReadToolInput, WriteToolInput, EditToolInput, dict[str, Any]  # Fallback for all other tools
    ],
    Field(union_mode='left_to_right'),
]


# ==============================================================================
# Image Source (must be defined before ImageContent)
# ==============================================================================


class ImageSource(StrictModel):
    """Image source data for image content."""

    type: Literal['base64']
    media_type: Literal['image/png']  # Only value observed across all sessions
    data: str  # Base64 encoded image data


class ImageContent(StrictModel):
    """Image content block from user messages."""

    type: Literal['image']
    source: ImageSource


class ToolUseContent(StrictModel):
    """Tool use content block from assistant messages."""

    type: Literal['tool_use']
    id: str
    name: str
    input: ToolInput  # Typed for Read/Write/Edit, dict for MCP tools only

    @field_validator('input', mode='after')
    @classmethod
    def validate_mcp_tool_fallback(cls, v: Any, info: ValidationInfo) -> Any:
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
ToolResultContentBlock = Annotated[Union[TextContent, ImageContent], Field(discriminator='type')]


class ToolResultContent(StrictModel):
    """Tool result content block from user messages."""

    type: Literal['tool_result']
    tool_use_id: str
    content: str | list[ToolResultContentBlock] | None = None  # Can be string, list of content blocks, or missing
    is_error: bool | None = None


# Discriminated union of all message content types
MessageContent = Annotated[
    Union[ThinkingContent, TextContent, ToolUseContent, ToolResultContent, ImageContent], Field(discriminator='type')
]


# ==============================================================================
# Message Structure
# ==============================================================================


class Message(StrictModel):
    """A message within a record."""

    role: Literal['user', 'assistant']
    content: list[MessageContent] | str
    # Additional fields that may appear in assistant messages (nested API response)
    type: Literal['message'] | None = Field(
        None, description='Message type indicator (present in agent/subprocess responses)'
    )
    model: str | None = Field(None, description='Claude model identifier (e.g., claude-sonnet-4-5-20250929)')
    id: str | None = Field(None, description='Message ID from Claude API')
    stop_reason: Literal['tool_use', 'stop_sequence', 'end_turn', 'refusal'] | None = Field(
        None, description='Reason why the model stopped generating'
    )
    stop_sequence: str | None = Field(None, description='The actual stop sequence string that triggered stopping')
    usage: TokenUsage | None = Field(None, description='Token usage information (present in nested API responses)')
    container: None = Field(None, description='Reserved for future use', json_schema_extra={'status': 'reserved'})
    context_management: None = Field(
        None, description='Reserved for future use', json_schema_extra={'status': 'reserved'}
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
    web_fetch_requests: int | None = None


class TokenUsage(StrictModel):
    """Token usage information for assistant messages."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation: CacheCreation | None = None
    service_tier: Literal['standard'] | None = None  # Only value: 'standard' (19018 occurrences)
    server_tool_use: ServerToolUse | None = None  # Server-side tool use tracking


# ==============================================================================
# Thinking Metadata
# ==============================================================================


class ThinkingMetadata(StrictModel):
    """Thinking configuration metadata."""

    level: Literal['none', 'low', 'medium', 'high']  # Strict validation
    disabled: bool
    triggers: list[str]


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
    headers: dict[str, str]
    requestID: str | None = None  # Can be null for some errors
    error: ApiErrorResponse | None = None  # Can be missing for some errors (e.g., 503)


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
    lines: list[str]


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
    filenames: list[str]
    numFiles: int
    durationMs: int | None = None
    truncated: bool | None = None


class GrepToolResult(StrictModel):
    """Result from Grep/content search tool execution."""

    mode: Literal['content', 'count']
    numFiles: int
    filenames: list[str]
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
    structuredPatch: list[PatchHunk]


class WriteToolResult(StrictModel):
    """Result from Write tool execution."""

    type: Literal['create', 'update']
    filePath: PathField
    content: str
    structuredPatch: list[PatchHunk] | None = None


class TodoToolResult(StrictModel):
    """Result from TodoWrite tool execution."""

    oldTodos: list[TodoItem]
    newTodos: list[TodoItem]


class TaskToolResult(StrictModel):
    """Result from Task/agent tool execution."""

    status: Literal['completed']
    prompt: str
    content: list[MessageContent]
    totalDurationMs: int
    totalTokens: int
    totalToolUseCount: int
    usage: TokenUsage
    agentId: str | None = None


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
    options: list[QuestionOption]
    multiSelect: bool


class AskUserQuestionToolResult(StrictModel):
    """Result from AskUserQuestion tool execution."""

    questions: list[UserQuestion]
    answers: list[str]  # List of question text that was answered


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
    results: list[WebSearchResult] | str  # Can be list of results or string explanation
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
    projectPaths: PathListField | None = Field(
        None, description='Additional project paths beyond cwd (each path will be translated)'
    )
    budgetTokens: int | None = Field(None, description='Token budget limit for this request')
    skills: None = Field(None, description='Reserved for future use', json_schema_extra={'status': 'reserved'})
    mcp: None = Field(None, description='Reserved for future use', json_schema_extra={'status': 'reserved'})
    agentId: str | None = Field(
        None, description='Agent ID for subprocess/agent records (references agent-{agentId}.jsonl)'
    )
    isMeta: bool | None = Field(None, description='Indicates meta messages (system-level information)')
    thinkingMetadata: ThinkingMetadata | None = Field(None, description='Extended thinking configuration (Claude 3.7+)')
    isVisibleInTranscriptOnly: bool | None = Field(
        None, description='Message visible only in transcript, not in session history'
    )
    isCompactSummary: bool | None = Field(None, description='Indicates this is a compacted session summary')
    toolUseResult: Annotated[
        list[McpResource]
        | list[TextContent]
        | ToolUseResultUnion
        | str
        | None,  # Ordered left-to-right: MCP resources first (no 'type' field), then TextContent, then structured, then string
        Field(union_mode='left_to_right'),
    ] = None  # Tool execution metadata (validated left-to-right)
    todos: list[TodoItem] | None = Field(None, description='Todo list state (Claude Code 2.0.47+)')


# ==============================================================================
# Assistant Record
# ==============================================================================


class AssistantRecord(BaseRecord):
    """Assistant message record."""

    type: Literal['assistant']
    cwd: PathField
    parentUuid: str | None = Field(..., description='UUID of parent record (null for root agent records)')
    message: Message
    # Note: usage/stopReason are optional for agent records (nested in message instead)
    usage: TokenUsage | None = Field(
        None, description='Token usage for this request (null for agent records - usage in message instead)'
    )
    stopReason: None = Field(None, description='Reserved for future use', json_schema_extra={'status': 'reserved'})
    model: str | None = Field(
        None, description='Claude model identifier (null for agent records - model in message instead)'
    )
    requestDuration: int | None = Field(None, description='Request duration in milliseconds')
    requestId: str | None = Field(None, description='Claude API request ID')
    agentId: str | None = Field(
        None, description='Agent ID for subprocess/agent records (references agent-{agentId}.jsonl)'
    )
    isSidechain: bool | None = Field(
        None, description='Indicates sidechain/subprocess execution (present in agent records)'
    )
    userType: str | None = Field(None, description='User type (present in agent records)')
    version: str | None = Field(None, description='Claude Code version (present in agent records)')
    gitBranch: str | None = Field(None, description='Git branch (present in agent records)')
    isApiErrorMessage: bool | None = Field(None, description='Indicates this message represents an API error')


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
    error: ApiError  # Error details with status, headers, requestID
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
    Union[LocalCommandSystemRecord, CompactBoundarySystemRecord, ApiErrorSystemRecord, InformationalSystemRecord],
    Field(discriminator='subtype'),
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
    trackedFileBackups: dict[str, FileBackupInfo]  # Map of file paths to backup data
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
    content: str | list[MessageContent] | None = Field(
        None, description='User input content for the queued operation (string or structured message)'
    )
    data: None = Field(None, description='Reserved for future use', json_schema_extra={'status': 'reserved'})


# ==============================================================================
# Session Record (Discriminated Union)
# ==============================================================================

# Union of all record types (validated left-to-right)
# NOTE: Cannot use discriminator='type' because multiple records have type='system'
# System subtype records must come before SystemRecord so they match first
SessionRecord = Annotated[
    Union[
        UserRecord,
        AssistantRecord,
        SummaryRecord,
        LocalCommandSystemRecord,  # Must be before SystemRecord!
        CompactBoundarySystemRecord,  # Must be before SystemRecord!
        ApiErrorSystemRecord,  # Must be before SystemRecord!
        InformationalSystemRecord,  # Must be before SystemRecord!
        SystemRecord,
        FileHistorySnapshotRecord,
        QueueOperationRecord,
    ],
    Field(union_mode='left_to_right'),
]

# Type adapter for validating session records (required for union types)
SessionRecordAdapter = TypeAdapter(SessionRecord)


# ==============================================================================
# Session Metadata
# ==============================================================================


class SessionMetadata(StrictModel):
    """Metadata extracted from a session."""

    session_id: str
    record_count: int
    first_timestamp: str
    last_timestamp: str
    unique_cwds: list[str]
    unique_project_paths: list[str]
    user_message_count: int
    assistant_message_count: int
    summary_count: int
    system_message_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_cache_creation_tokens: int
    total_cache_read_tokens: int
    models_used: list[str]
    tools_used: list[str]
    files_touched: list[str]
    git_branches: list[str]


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
    (KillShellToolResult, 'KillShell'),
    (McpResource, 'ListMcpResourcesTool'),  # Returns list[McpResource], not dict result
]

# Allowed tool names (extracted from mapping)
ALLOWED_CLAUDE_TOOL_NAMES = {tool_name for _, tool_name in MODELED_CLAUDE_TOOLS}
