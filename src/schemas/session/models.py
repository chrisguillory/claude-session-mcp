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
- Schema v0.2.0: Added sourceToolAssistantUUID to UserRecord, TurnDurationSystemRecord for Claude Code 2.1.1+
- Schema v0.2.4: Added ProgressRecord (hook/mcp/bash/task progress), MicrocompactBoundarySystemRecord,
                 allowedPrompts to ExitPlanModeToolInput, fixed MCPSearchToolInput max_results type (2.1.9+)
- Schema v0.2.5: Added permissionMode to UserRecord, apiError to AssistantRecord (2.1.15+)
- Schema v0.2.6: Added StopHookSummarySystemRecord, resume field to AgentProgressData (2.1.14+),
                 TaskCreate/TaskUpdate/TaskList tool inputs and results (2.1.17+)
- Schema v0.2.7: Added SimpleThinkingMetadata, McpMeta, MCPStructuredContent, Task tool mode field (2.1.19+)
- Schema v0.2.8: Added timeoutMs to BashProgressData (2.1.25+)
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
  Input: {"skill": "handoff", "args": "..."}
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
from typing import Annotated, Any, Literal

import pydantic

from src.schemas.session.markers import PathField, PathListField
from src.schemas.types import BaseStrictModel, EmptyDict, EmptySequence, ModelId, PermissiveModel

# ==============================================================================
# Schema Version
# ==============================================================================

SCHEMA_VERSION = '0.2.8'
CLAUDE_CODE_MIN_VERSION = '2.0.35'
CLAUDE_CODE_MAX_VERSION = '2.1.25'
LAST_VALIDATED = '2026-01-30'
VALIDATION_RECORD_COUNT = 255_102


# ==============================================================================
# Base Configuration
# ==============================================================================


class StrictModel(BaseStrictModel):
    """Session-layer strict model.

    Inherits from BaseStrictModel (extra='forbid', strict=True, frozen=True).
    Domain-specific customization can be added here if needed.
    """

    pass


# noinspection PyNewStyleGenericSyntax
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
    """Input for Read tool.

    Fields:
        file_path: Absolute path to the file to read
        limit: Maximum number of lines to read (for large files)
        offset: Line number to start reading from (1-indexed)
    """

    file_path: PathField
    limit: int | None = None  # Max lines to read
    offset: int | str | None = None  # Start line (1-indexed), can be malformed string like "\\248"


class WriteToolInput(StrictModel):
    """Input for Write tool."""

    file_path: PathField
    content: str


class MalformedWriteToolInput(StrictModel):
    """Malformed Write tool input from historical JSON serialization bug.

    CONTEXT:
    This model exists to handle exactly 2 session records from December 24, 2025
    (Claude Code version 2.0.76) where a JSON serialization bug caused malformed
    tool_use inputs to be recorded in session files.

    WHAT HAPPENED:
    The model was writing a markdown file documenting Claude-in-Chrome MCP tool
    invocation patterns. The markdown content contained example XML showing how
    to invoke tools, including syntax like:

        <parameter name="param1">value1","param2":"value2"}}]

    Something in the serialization chain incorrectly parsed this content. The
    `param2` substring within the content string was extracted as an actual
    tool parameter, producing a malformed tool_use input like:

        {"file_path": "...", "content": "...<truncated>...", "param2": "value2"}}]..."}

    OUTCOME:
    Claude Code's input validation correctly rejected these tool calls with:
        InputValidationError: An unexpected parameter `param2` was provided

    The writes never executed - the session files contain records of rejected
    attempts, not successful operations.

    WHY WE MODEL THIS:
    Our Pydantic models validate session records (what was attempted), not tool
    execution success. The malformed tool_use attempts exist in the session files
    and need a matching model to pass validation.

    EVIDENCE:
    See fixtures/edge_cases/malformed_write_param2.jsonl for sanitized reproduction
    records. This fixture contains the actual malformed records (paths sanitized)
    and is validated by tests/test_fixtures.py.

    FUTURE:
    If Anthropic fixes the serialization bug and cleans up historical session
    files, this model can be removed. The bug appears fixed in later versions
    (no occurrences after December 2025).
    """

    file_path: PathField
    content: str
    param2: str  # The malformed field - always present in affected records


class EditToolInput(StrictModel):
    """Input for Edit tool."""

    file_path: PathField
    old_string: str
    new_string: str
    replace_all: bool | None = None


class SkillToolInput(StrictModel):
    """Input for Skill tool.

    Fields:
        skill: Skill name to invoke (e.g., 'handoff', 'commit')
        args: Optional arguments for the skill
    """

    skill: str
    args: str | None = None


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


# ==============================================================================
# Bash Tool Input (23,236x occurrences)
# ==============================================================================


class BashToolInput(StrictModel):
    """Input for Bash tool - executes shell commands.

    Fields:
        command: The shell command to execute (required)
        description: Human-readable description in 5-10 words (usually present, but optional)
        timeout: Timeout in milliseconds (max 600000, default 120000)
        run_in_background: Run command asynchronously
        dangerouslyDisableSandbox: Bypass sandbox protection (requires policy permission)
    """

    command: str
    description: str | None = None  # Usually present (1078/1080) but some old records lack it
    timeout: int | None = None
    run_in_background: bool | None = None
    dangerouslyDisableSandbox: bool | None = None


# ==============================================================================
# Grep Tool Input (2,909x occurrences)
# ==============================================================================


class GrepToolInput(StrictModel):
    """Input for Grep tool - searches file contents using ripgrep.

    Fields:
        pattern: Regex pattern to search for (required)
        path: Directory or file to search in (defaults to cwd)
        output_mode: content | files_with_matches | count
        glob: Glob pattern to filter files (e.g., "*.py")
        type: File type filter (e.g., "py", "js")
        multiline: Enable multiline regex mode
        head_limit: Limit output to first N results
        offset: Skip first N results
        context_lines: Number of context lines (alternative to -C)
        flags: String-format flags (e.g., "-i")
        grep: Alternative flag format (e.g., "-n")
        Hyphenated flags: -n (line numbers), -A/-B/-C (context), -i (case insensitive)
    """

    model_config = pydantic.ConfigDict(
        extra='forbid',
        strict=True,
        frozen=True,
        populate_by_name=True,  # Allow both alias and field name
    )

    pattern: str
    path: PathField | None = None
    output_mode: Literal['content', 'files_with_matches', 'count'] | None = None
    glob: str | None = None
    type: str | None = None  # noqa: A003 - matches ripgrep's --type flag
    multiline: bool | None = None
    head_limit: int | None = None
    offset: int | None = None
    context: int | str | None = None  # Context lines - can be int or flag reference string like "-A"
    context_lines: int | None = None  # Explicit context lines count (alternative to -C)
    flags: str | None = None  # String-format flags (e.g., "-i")
    grep: str | None = None  # Alternative flag format (e.g., "-n") - legacy/variant usage
    # Hyphenated ripgrep flags (use Field alias for JSON compatibility)
    dash_n: bool | None = pydantic.Field(None, alias='-n')
    dash_A: int | None = pydantic.Field(None, alias='-A')
    dash_B: int | None = pydantic.Field(None, alias='-B')
    dash_C: int | None = pydantic.Field(None, alias='-C')
    dash_i: bool | None = pydantic.Field(None, alias='-i')


# ==============================================================================
# Glob Tool Input (2,507x occurrences)
# ==============================================================================


class GlobToolInput(StrictModel):
    """Input for Glob tool - file pattern matching.

    Fields:
        pattern: Glob pattern to match files (e.g., "**/*.py")
        path: Directory to search in (defaults to cwd)
    """

    pattern: str
    path: PathField | None = None


# ==============================================================================
# Task Tool Input (872x occurrences)
# ==============================================================================


class TaskToolInput(StrictModel):
    """Input for Task tool - launches subagent for autonomous tasks.

    Fields:
        prompt: Task instructions for the subagent (required)
        subagent_type: Agent type - "Explore", "general-purpose", etc. (required)
        description: Human-readable task description for UI (usually present but optional)
        allowed_tools: Tools to grant the subagent (e.g., ["Write", "Bash"])
        run_in_background: Run asynchronously, check with TaskOutput
        model: Override model (e.g., "haiku", "sonnet")
        resume: Agent ID to resume from previous execution
        mode: Permission mode for the agent (Claude Code 2.1.19+)
    """

    description: str | None = None  # Usually present but some early records lack it
    prompt: str
    subagent_type: str
    allowed_tools: Sequence[str] | None = None  # Tools to grant the subagent
    run_in_background: bool | None = None
    model: str | None = None
    resume: str | None = None
    mode: Literal['default', 'bypassPermissions'] | None = None  # Permission mode (2.1.19+)


# ==============================================================================
# TaskCreate Tool Input (Claude Code 2.1.17+)
# ==============================================================================


class TaskCreateToolInput(StrictModel):
    """Input for TaskCreate tool - creates a new task in the task list.

    Fields:
        subject: Brief title for the task (imperative form, e.g., "Run tests")
        description: Detailed description of what needs to be done
        activeForm: Present continuous form shown in spinner (e.g., "Running tests")
    """

    subject: str
    description: str
    activeForm: str


# ==============================================================================
# TaskUpdate Tool Input (Claude Code 2.1.17+)
# ==============================================================================


class TaskUpdateToolInput(StrictModel):
    """Input for TaskUpdate tool - updates an existing task.

    Fields:
        taskId: ID of the task to update (required)
        status: New status (pending, in_progress, completed)
        description: New description for the task
        owner: Agent/worker name to assign the task to
        addBlockedBy: Task IDs that block this task
    """

    taskId: str
    status: Literal['pending', 'in_progress', 'completed', 'deleted'] | None = None
    description: str | None = None  # Updated task description
    owner: str | None = None
    addBlockedBy: Sequence[str] | None = None


# ==============================================================================
# TaskList Tool Input (Claude Code 2.1.17+)
# ==============================================================================


class TaskListToolInput(StrictModel):
    """Input for TaskList tool - lists all tasks (no parameters)."""

    pass


# ==============================================================================
# TodoWrite Tool Input (3,186x occurrences)
# ==============================================================================


class TodoWriteToolInput(StrictModel):
    """Input for TodoWrite tool - tracks task progress.

    Fields:
        todos: List of todo items, each with content/status/activeForm
    """

    todos: Sequence[TodoItem]


# ==============================================================================
# WebSearch Tool Input (284x occurrences)
# ==============================================================================


class WebSearchToolInput(StrictModel):
    """Input for WebSearch tool - web search.

    Fields:
        query: Search query string (required)
        allowed_domains: Only include results from these domains
        blocked_domains: Exclude results from these domains
    """

    query: str
    allowed_domains: Sequence[str] | None = None
    blocked_domains: Sequence[str] | None = None


# ==============================================================================
# WebFetch Tool Input (177x occurrences)
# ==============================================================================


class WebFetchToolInput(StrictModel):
    """Input for WebFetch tool - fetches URL content.

    Fields:
        url: URL to fetch (required)
        prompt: Prompt to run on fetched content (required)
    """

    url: str
    prompt: str


# ==============================================================================
# BashOutput Tool Input (192x occurrences)
# ==============================================================================


class BashOutputToolInput(StrictModel):
    """Input for BashOutput tool - retrieves background bash output.

    Fields:
        bash_id: Background task ID (required)
        block: Whether to wait for completion
        filter: Regex pattern to filter output
        wait_up_to: Max seconds to wait (for blocking mode)
    """

    bash_id: str
    block: bool | None = None
    filter: str | None = None  # noqa: A003 - matches tool's field name
    wait_up_to: int | None = None


# ==============================================================================
# AskUserQuestion Tool Input (159x occurrences)
# ==============================================================================


class AskUserQuestionToolInput(StrictModel):
    """Input for AskUserQuestion tool - asks user multiple choice questions.

    Fields:
        questions: List of questions (1-4), each with question/header/options/multiSelect
        answers: User answers (populated by permission component)
    """

    questions: Sequence[UserQuestion]
    answers: Mapping[str, str] | None = None


# ==============================================================================
# ExitPlanMode Tool Input (150x occurrences)
# ==============================================================================


class PromptPermission(StrictModel):
    """A prompt-based permission request in ExitPlanMode."""

    tool: str
    prompt: str


class ExitPlanModeToolInput(StrictModel):
    """Input for ExitPlanMode tool - exits planning mode.

    Note: This tool can be invoked with empty input {} or with plan/launchSwarm.
    The empty variant signals plan approval request.

    Fields:
        plan: Plan content in markdown (optional)
        launchSwarm: Whether to launch swarm agents (optional)
        allowedPrompts: Prompt-based permission requests (tool + prompt pairs)
        pushToRemote: Whether to push plan to remote Claude.ai session
    """

    plan: str | None = None
    launchSwarm: bool | None = None
    allowedPrompts: Sequence[PromptPermission] | None = None
    pushToRemote: bool | None = None


# ==============================================================================
# KillShell Tool Input (58x occurrences)
# ==============================================================================


class KillShellToolInput(StrictModel):
    """Input for KillShell tool - terminates a running shell.

    Fields:
        shell_id: ID of the shell to kill (required)
    """

    shell_id: str


# ==============================================================================
# ListMcpResourcesTool Input (5x occurrences)
# ==============================================================================


class ListMcpResourcesToolInput(StrictModel):
    """Input for ListMcpResourcesTool - lists available MCP resources.

    Fields:
        server: Filter by specific MCP server name (optional)
    """

    server: str | None = None


# ==============================================================================
# NotebookEdit Tool Input
# ==============================================================================


class NotebookEditToolInput(StrictModel):
    """Input for NotebookEdit tool - edits Jupyter notebook cells.

    Fields:
        notebook_path: Absolute path to the .ipynb file (required)
        new_source: New source content for the cell (required)
        cell_id: ID of cell to edit (optional)
        cell_type: Type of cell - code or markdown
        edit_mode: replace | insert | delete
    """

    notebook_path: PathField
    new_source: str
    cell_id: str | None = None
    cell_type: Literal['code', 'markdown'] | None = None
    edit_mode: Literal['replace', 'insert', 'delete'] | None = None


# ==============================================================================
# ReadMcpResource Tool Input
# ==============================================================================


class ReadMcpResourceToolInput(StrictModel):
    """Input for ReadMcpResourceTool - reads a specific MCP resource.

    Fields:
        server: MCP server name (required)
        uri: Resource URI to read (required)
    """

    server: str
    uri: str


# ==============================================================================
# LSP Tool Input (Language Server Protocol operations)
# ==============================================================================


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


class LSPToolInput(StrictModel):
    """Input for LSP tool - Language Server Protocol operations.

    Fields:
        operation: LSP operation to perform
        filePath: Path to the file to operate on
        line: Line number (1-based as shown in editors)
        character: Character offset (1-based as shown in editors)
    """

    operation: LSPOperation
    filePath: PathField
    line: int
    character: int


# ==============================================================================
# MCPSearch Tool Input
# ==============================================================================


class MCPSearchToolInput(StrictModel):
    """Input for MCPSearch tool - searches for MCP tools.

    Fields:
        query: Search query string, can use 'select:' prefix for exact tool selection
        max_results: Maximum number of results to return (optional, can be int or string)
    """

    query: str
    max_results: int | str | None = None  # Can be "1" string or 1 int depending on serialization


# ==============================================================================
# MCP Tool Input (Third-Party Tools)
# ==============================================================================


class MCPToolInput(PermissiveModel):
    """Permissive model for MCP (Model Context Protocol) tool inputs.

    MCP tools are third-party integrations with varying schemas. We intentionally
    do not model their specific fields because:
    1. There are 64+ different MCP tools with different schemas
    2. They are user-configured and can change without Claude Code updates
    3. Their schemas are defined by external MCP servers

    The validator on ToolUseContent enforces that ONLY tools with names starting
    with 'mcp__' can use this model. Claude Code built-in tools MUST have typed
    models - if one falls through to MCPToolInput, the validator raises an error.

    Uses PermissiveModel to accept any fields while maintaining type observability.
    """

    pass


# Union of tool inputs (typed models first, PermissiveModel fallback for MCP tools)
# NOTE: Order matters! More specific (more required fields) should come first.
# Models with no required fields must come last before fallback.
ToolInput = Annotated[
    # Path-based tools (most specific - file_path + other required fields)
    MalformedWriteToolInput  # file_path, content, param2 required - must be before WriteToolInput!
    | WriteToolInput  # file_path, content required
    | EditToolInput  # file_path, old_string, new_string required
    | NotebookEditToolInput  # notebook_path, new_source required
    | ReadToolInput  # file_path required
    # Multi-field tools
    | TaskToolInput  # prompt, description, subagent_type required
    | TaskCreateToolInput  # subject, description required (2.1.17+)
    | BashToolInput  # command required (description optional since some old records lack it)
    | GrepToolInput  # pattern required, has custom model_config
    | GlobToolInput  # pattern required
    | WebFetchToolInput  # url, prompt required
    | ReadMcpResourceToolInput  # server, uri required
    | AskUserQuestionToolInput  # questions required
    | TodoWriteToolInput  # todos required
    | WebSearchToolInput  # query required
    | MCPSearchToolInput  # query required (new in 2.1.4)
    | LSPToolInput  # operation, filePath, line, character required (ENABLE_LSP_TOOL=1)
    # Single-field tools
    | AgentOutputToolInput  # agentId required
    | TaskOutputToolInput  # task_id required
    | TaskUpdateToolInput  # taskId required (2.1.17+)
    | BashOutputToolInput  # bash_id required
    | KillShellToolInput  # shell_id required
    | SkillToolInput  # skill required
    # Optional-only fields (must be near end)
    | ExitPlanModeToolInput  # plan optional, launchSwarm optional
    | ListMcpResourcesToolInput  # server optional
    | TaskListToolInput  # No fields (2.1.17+)
    | EnterPlanModeToolInput  # No fields - must be last before fallback!
    | MCPToolInput,  # Fallback for MCP tools (PermissiveModel for observability)
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


class ToolUseCaller(StrictModel):
    """Caller metadata for tool use (Claude Code 2.1.4+)."""

    type: Literal['direct']  # Only observed value so far


class ToolUseContent(StrictModel):
    """Tool use content block from assistant messages."""

    type: Literal['tool_use']
    id: str
    name: str
    input: ToolInput  # Typed for Claude Code tools, MCPToolInput for MCP tools
    caller: ToolUseCaller | None = None  # Caller metadata (Claude Code 2.1.4+)

    @pydantic.field_validator('input', mode='after')
    @classmethod
    def validate_mcp_tool_fallback(cls, v: ToolInput, info: pydantic.ValidationInfo) -> ToolInput:
        """
        Enforce that only MCP tools (starting with 'mcp__') can use MCPToolInput fallback.
        All Claude Code built-in tools must have typed models that successfully validate.

        This catches both:
        1. New Claude Code tools we haven't modeled yet
        2. Bugs in existing models (missing/wrong fields causing fallthrough)
        """
        if isinstance(v, MCPToolInput):
            tool_name = info.data.get('name', '')

            # MCP tools are expected to use the fallback - they're third-party
            if tool_name.startswith('mcp__'):
                return v

            # ANY Claude Code tool using fallback is a bug - either:
            # - New tool needs a model, OR
            # - Existing model has missing/wrong fields
            raise ValueError(
                f"Claude Code tool '{tool_name}' fell through to MCPToolInput. "
                f'This means either: (1) no typed model exists for this tool, or '
                f'(2) the typed model has missing/incorrect fields. '
                f'Extra fields captured: {list(v.get_extra_fields().keys())}'
            )

        return v


class ToolReferenceContent(StrictModel):
    """Tool reference content block from MCPSearch tool results (Claude Code 2.1.4+).

    Appears in tool_result content when MCPSearch returns matched MCP tools.
    """

    type: Literal['tool_reference']
    tool_name: str  # Full MCP tool name (e.g., "mcp__perplexity__perplexity_research")


# ToolResultContentBlock - for content inside tool_result
ToolResultContentBlock = Annotated[
    TextContent | ImageContent | ToolReferenceContent, pydantic.Field(discriminator='type')
]


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
    """Thinking configuration metadata (full format, pre-2.1.19).

    This is the original format with level/disabled/triggers configuration.
    See also SimpleThinkingMetadata for the newer simplified format.
    """

    level: Literal['none', 'low', 'medium', 'high']  # Strict validation
    disabled: bool
    triggers: Sequence[str | ThinkingTrigger]  # Can be strings or trigger objects


class SimpleThinkingMetadata(StrictModel):
    """Simplified thinking metadata (Claude Code 2.1.19+).

    New format that only specifies max thinking tokens without
    the full level/disabled/triggers configuration.
    See also ThinkingMetadata for the original full format.
    """

    maxThinkingTokens: int


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


class MicrocompactMetadata(StrictModel):
    """Metadata for micro-compaction (Claude Code 2.1.9+)."""

    trigger: Literal['auto']  # Only 'auto' observed so far
    preTokens: int
    tokensSaved: int
    compactedToolIds: Sequence[str]
    clearedAttachmentUUIDs: EmptySequence  # Only empty arrays observed


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


# noinspection PyShadowingBuiltins
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
# MCP Metadata (Claude Code 2.1.19+)
# ==============================================================================


class MCPStructuredContent(PermissiveModel):
    """Permissive model for MCP tool structured content (in mcpMeta.structuredContent).

    MCP tools are third-party integrations with varying result schemas. We
    intentionally do not model their specific fields because:
    1. There are 64+ different MCP tools with different result structures
    2. They are user-configured and can change without Claude Code updates
    3. Their schemas are defined by external MCP servers

    NOTE: This is the NEW location for structured MCP content (Claude Code 2.1.19+).
    The same data also appears in tool_result.content as a JSON string. See also
    MCPToolResult which handles the toolUseResult field.

    Uses PermissiveModel to accept any fields while maintaining type observability.
    """

    pass


class McpMeta(StrictModel):
    """MCP tool metadata wrapper (Claude Code 2.1.19+).

    New in 2.1.19: Claude Code extracts structured content from MCP tool
    results that return valid JSON and surfaces it here for easier access.

    DUPLICATION: The same data appears both here (as parsed dict) AND in
    tool_result.content (as JSON string). This is a convenience feature
    for programmatic access without parsing.

    NOT BACKFILLED: Old sessions retain the old format (mcpMeta=None).
    When you resume an old session, new tool calls get mcpMeta but old
    records are not updated. Both patterns can coexist in the same file.

    Only populated for MCP tools returning JSON - tools returning plain text
    (like Perplexity) do not populate this field.
    """

    model_config = pydantic.ConfigDict(
        extra='forbid',
        strict=True,
        frozen=True,
        populate_by_name=True,  # Allow field access by both name and alias
    )

    meta: EmptyDict | None = pydantic.Field(None, alias='_meta')  # Always {} when present
    structuredContent: MCPStructuredContent | None = None


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
    appliedLimit: int | None = None  # Limit that was applied to results


class GrepToolResult(StrictModel):
    """Result from Grep/content search tool execution."""

    mode: Literal['content', 'count', 'files_with_matches']
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
    originalFile: str | None = None  # Present but always None for 'create' type


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
# TaskCreate/TaskUpdate/TaskList Result Structures (Claude Code 2.1.17+)
# ==============================================================================


class TaskListItem(StrictModel):
    """A task item in TaskList results."""

    id: str
    subject: str
    status: Literal['pending', 'in_progress', 'completed']
    blockedBy: Sequence[str]
    owner: str | None = None  # Only present when task is assigned


class TaskListToolResult(StrictModel):
    """Result from TaskList tool - list of all tasks."""

    tasks: Sequence[TaskListItem]


class TaskSingleItem(StrictModel):
    """A task item in TaskCreate/TaskUpdate results (minimal fields)."""

    id: str
    subject: str


class TaskSingleToolResult(StrictModel):
    """Result from TaskCreate/TaskUpdate - single task confirmation."""

    task: TaskSingleItem


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
    answers: Mapping[str, str]  # Mapping of question text to user's answer


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
    durationSeconds: int | float  # Can be int or float depending on timing


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
    filePath: PathField | None = None  # Plan file path (present in newer versions)


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


# ==============================================================================
# TaskOutput Polling Results
# ==============================================================================


class BackgroundTask(StrictModel):
    """Background task state from TaskOutput tool polling."""

    task_id: str
    task_type: Literal['local_bash', 'local_agent']
    status: Literal['running', 'completed', 'failed']
    description: str
    output: str
    exitCode: int | None = None  # Null when running


class TaskOutputPollingResult(StrictModel):
    """Result from TaskOutput tool - polling background task state."""

    retrieval_status: Literal['not_ready', 'success']
    task: BackgroundTask


# ==============================================================================
# Async Task Launch Results
# ==============================================================================


class AsyncTaskLaunchResult(StrictModel):
    """Result from launching async Task (with or without output file tracking)."""

    isAsync: Literal[True]
    status: Literal['async_launched']
    agentId: str
    description: str
    prompt: str
    outputFile: str | None = None  # Path to output file (sometimes missing)


# ==============================================================================
# Multi-Agent Retrieval Results
# ==============================================================================


class AgentCompletedState(StrictModel):
    """State of a completed background agent."""

    status: Literal['completed']
    description: str
    prompt: str
    result: str


class AgentsRetrievalResult(StrictModel):
    """Result from retrieving multiple agent states."""

    retrieval_status: Literal['not_ready', 'success']
    agents: Mapping[str, AgentCompletedState]  # Empty dict when not_ready


# ==============================================================================
# KillShell Message Variant
# ==============================================================================


class KillShellMessageResult(StrictModel):
    """Alternative KillShell result with message format (snake_case shell_id)."""

    message: str  # e.g., "Successfully killed shell: b18fae0 (...)"
    shell_id: str  # Note: uses snake_case, not camelCase


# ==============================================================================
# WebSearch Nested Structure Variant
# ==============================================================================


class WebSearchResultWrapper(StrictModel):
    """Wrapper for web search results with tool use ID (nested structure variant)."""

    tool_use_id: str
    content: Sequence[WebSearchResult]


class WebSearchNestedResult(StrictModel):
    """Result from WebSearch tool with nested structure variant."""

    query: str
    results: Sequence[WebSearchResultWrapper | str]  # Can be wrapper or text
    durationSeconds: float


# ==============================================================================
# Handoff Command Result
# ==============================================================================


class HandoffCommandResult(StrictModel):
    """Result from handoff command execution."""

    success: Literal[True]
    commandName: Literal['handoff']
    allowedTools: Sequence[str]


# ==============================================================================
# EnterPlanMode Tool Result
# ==============================================================================


class EnterPlanModeToolResult(StrictModel):
    """Result from EnterPlanMode tool execution."""

    message: str  # Plan mode entry confirmation


# ==============================================================================
# MCP Tool Result (Third-Party Tools)
# ==============================================================================


class MCPToolResult(PermissiveModel):
    """Permissive model for MCP (Model Context Protocol) tool results.

    MCP tools are third-party integrations with varying result schemas. We
    intentionally do not model their specific fields because:
    1. There are 64+ different MCP tools with different result structures
    2. They are user-configured and can change without Claude Code updates
    3. Their schemas are defined by external MCP servers

    NOTE: Unlike MCPToolInput, we cannot validate that only MCP tools use this
    fallback. Tool results don't carry the tool name (it's in the previous
    assistant message's ToolUseContent). Observability is provided by
    find_fallbacks() in validate_models.py instead.

    Uses PermissiveModel to accept any fields while maintaining type observability.
    """

    pass


# Union of all tool result types (validated left-to-right, most specific first)
# NOTE: Order matters! More specific models (more required fields) should come first.
# NOTE: Unlike ToolInput, there's no validator enforcing MCP-only fallback here.
# This is intentional: tool results don't carry the tool name (it's in the previous
# assistant message's ToolUseContent), so we can't easily distinguish MCP vs Claude Code.
# Observability is provided by find_fallbacks() in validate_models.py instead.
ToolResult = Annotated[
    # Core tool results (most specific first)
    BashToolResult  # Also handles BashOutput
    | ReadToolResult
    | EditToolResult
    | WriteToolResult
    | GrepToolResult
    | GlobToolResult
    | TodoToolResult
    | TaskToolResult  # Completed sync task
    | TaskOutputPollingResult  # Polling async task
    | AsyncTaskLaunchResult  # Launched async task
    | AgentsRetrievalResult  # Multi-agent polling
    | TaskListToolResult  # TaskList result (2.1.17+)
    | TaskSingleToolResult  # TaskCreate/TaskUpdate result (2.1.17+)
    | AskUserQuestionToolResult
    | WebSearchNestedResult  # Nested structure variant (more specific - has tool_use_id in results)
    | WebSearchToolResult  # Simple structure
    | WebFetchToolResult
    | ExitPlanModeToolResult
    | EnterPlanModeToolResult  # Plan mode entry
    | KillShellMessageResult  # Message variant (has message + shell_id)
    | KillShellToolResult  # Original variant (has success + shellId)
    | HandoffCommandResult  # Handoff command
    | MCPToolResult,  # Fallback for MCP tools (PermissiveModel for observability)
    pydantic.Field(union_mode='left_to_right'),
]

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
    thinkingMetadata: ThinkingMetadata | SimpleThinkingMetadata | None = pydantic.Field(
        None, description='Extended thinking configuration (Claude 3.7+, simplified format in 2.1.19+)'
    )
    isVisibleInTranscriptOnly: bool | None = pydantic.Field(
        None, description='Message visible only in transcript, not in session history'
    )
    isCompactSummary: bool | None = pydantic.Field(None, description='Indicates this is a compacted session summary')
    toolUseResult: Annotated[
        Sequence[ToolResultContentBlock]  # TextContent/ImageContent with 'type' discriminator - must come first
        | Sequence[McpResource]  # MCP resources (no 'type' field, has 'name', 'uri', etc.)
        | ToolResult
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
    sourceToolAssistantUUID: str | None = pydantic.Field(
        None,
        description='UUID of the assistant message that created the tool use this record responds to',
    )
    permissionMode: Literal['default', 'acceptEdits', 'plan', 'bypassPermissions'] | None = pydantic.Field(
        None, description='Permission mode for the request (Claude Code 2.1.15+)'
    )
    mcpMeta: McpMeta | None = pydantic.Field(
        None, description='MCP tool structured content metadata (Claude Code 2.1.19+)'
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
    apiError: Literal['max_output_tokens'] | None = pydantic.Field(
        None, description='API error code (Claude Code 2.1.15+)'
    )
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


class MicrocompactBoundarySystemRecord(BaseRecord):
    """System record for micro-compaction (subtype=microcompact_boundary, Claude Code 2.1.9+)."""

    type: Literal['system']
    cwd: PathField
    parentUuid: str | None
    subtype: Literal['microcompact_boundary']
    content: Literal['Context microcompacted']
    level: Literal['info', 'error', 'warning', 'suggestion'] | None = None
    isMeta: bool
    isSidechain: bool
    userType: str
    version: str
    gitBranch: str
    slug: str | None = None
    microcompactMetadata: MicrocompactMetadata


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


class TurnDurationSystemRecord(BaseRecord):
    """System record for turn duration tracking (subtype=turn_duration, Claude Code 2.1.1+)."""

    type: Literal['system']
    cwd: PathField
    parentUuid: str | None
    subtype: Literal['turn_duration']
    durationMs: int  # Duration of the turn in milliseconds
    isMeta: bool
    isSidechain: bool
    userType: str
    version: str
    gitBranch: str
    slug: str | None = None


class HookInfo(StrictModel):
    """Information about a hook execution."""

    command: str


class StopHookSummarySystemRecord(BaseRecord):
    """System record for stop hook summary (subtype=stop_hook_summary, Claude Code 2.1.14+)."""

    type: Literal['system']
    cwd: PathField
    parentUuid: str | None
    subtype: Literal['stop_hook_summary']
    hookCount: int
    hookInfos: Sequence[HookInfo]
    hookErrors: Sequence[str]  # Empty sequence observed so far
    preventedContinuation: bool
    stopReason: str  # Can be empty string
    hasOutput: bool
    level: Literal['info', 'error', 'warning', 'suggestion'] | None = None
    isSidechain: bool | None = None
    userType: str | None = None
    version: str | None = None
    gitBranch: str | None = None
    toolUseID: str | None = None  # Tool use ID if triggered by tool
    slug: str | None = None  # Human-readable session slug


# Union of system subtype records
SystemSubtypeRecord = Annotated[
    LocalCommandSystemRecord
    | CompactBoundarySystemRecord
    | MicrocompactBoundarySystemRecord
    | ApiErrorSystemRecord
    | InformationalSystemRecord
    | TurnDurationSystemRecord
    | StopHookSummarySystemRecord,
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
# Progress Record (Claude Code 2.1.9+)
# ==============================================================================


class HookProgressData(StrictModel):
    """Progress data for hook execution."""

    type: Literal['hook_progress']
    hookEvent: str  # e.g., "SessionStart"
    hookName: str  # e.g., "SessionStart:startup"
    command: PathField


class McpProgressStartedData(StrictModel):
    """Progress data for MCP tool start."""

    type: Literal['mcp_progress']
    status: Literal['started']
    serverName: str
    toolName: str


class McpProgressCompletedData(StrictModel):
    """Progress data for MCP tool completion."""

    type: Literal['mcp_progress']
    status: Literal['completed']
    serverName: str
    toolName: str
    elapsedTimeMs: int


class McpProgressFailedData(StrictModel):
    """Progress data for MCP tool failure."""

    type: Literal['mcp_progress']
    status: Literal['failed']
    serverName: str
    toolName: str
    elapsedTimeMs: int


class BashProgressData(StrictModel):
    """Progress data for bash command execution."""

    type: Literal['bash_progress']
    output: str
    fullOutput: str
    elapsedTimeSeconds: int
    totalLines: int
    timeoutMs: int | None = None  # Present when command has explicit timeout


class WaitingForTaskData(StrictModel):
    """Progress data for waiting on a task."""

    type: Literal['waiting_for_task']
    taskDescription: str
    taskType: Literal['local_bash', 'local_agent']


class SearchResultsReceivedData(StrictModel):
    """Progress data for web search results received (Claude Code 2.1.9+)."""

    type: Literal['search_results_received']
    resultCount: int
    query: str


class QueryUpdateData(StrictModel):
    """Progress data for search query update (Claude Code 2.1.9+)."""

    type: Literal['query_update']
    query: str


class AgentProgressData(StrictModel):
    """Progress data for agent/subagent execution."""

    type: Literal['agent_progress']
    agentId: str
    prompt: str
    # TODO: Remove "loose" typing below
    message: Mapping[str, Any]  # check_schema_typing.py: loose-typing
    # TODO: Remove "loose" typing below
    normalizedMessages: Sequence[Mapping[str, Any]]  # check_schema_typing.py: loose-typing
    resume: str | None = None  # Agent resume ID (Claude Code 2.1.15+)


# Discriminated union of progress data types
# NOTE: Models with more required fields must come first (left_to_right matching)
ProgressData = Annotated[
    AgentProgressData  # Most fields (5 required)
    | HookProgressData
    | McpProgressCompletedData
    | McpProgressFailedData
    | McpProgressStartedData
    | BashProgressData
    | WaitingForTaskData
    | SearchResultsReceivedData  # Web search progress (Claude Code 2.1.9+)
    | QueryUpdateData,  # Search query update (fewest fields - must be last)
    pydantic.Field(union_mode='left_to_right'),
]


class ProgressRecord(StrictModel):
    """Progress record for tracking long-running operations (Claude Code 2.1.9+)."""

    type: Literal['progress']
    uuid: str
    timestamp: str
    sessionId: str
    cwd: PathField
    parentUuid: str | None
    isSidechain: bool
    userType: Literal['external']
    version: str
    gitBranch: str
    data: ProgressData
    parentToolUseID: str
    toolUseID: str
    slug: str | None = None  # Missing on first record before slug assigned
    agentId: str | None = None  # Present in agent subfiles (references agent-{agentId}.jsonl)


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
    | MicrocompactBoundarySystemRecord  # Must be before SystemRecord!
    | ApiErrorSystemRecord  # Must be before SystemRecord!
    | InformationalSystemRecord  # Must be before SystemRecord!
    | TurnDurationSystemRecord  # Must be before SystemRecord!
    | StopHookSummarySystemRecord  # Must be before SystemRecord!
    | SystemRecord
    | FileHistorySnapshotRecord
    | QueueOperationRecord
    | CustomTitleRecord
    | ProgressRecord,
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
