"""
Session JSONL schema models.

This package contains Pydantic models for Claude Code session JSONL records.
All models are currently in models.py - see README.md for planned structure.
"""

from __future__ import annotations

# Re-export markers for convenience
from src.schemas.session.markers import PathField, PathListField, PathMarker

# Re-export all public symbols from models
from src.schemas.session.models import (
    CLAUDE_CODE_MAX_VERSION,
    CLAUDE_CODE_MIN_VERSION,
    LAST_VALIDATED,
    # Schema version
    SCHEMA_VERSION,
    VALIDATION_RECORD_COUNT,
    # Tool results (alphabetical)
    AgentCompletedState,
    # Tool inputs (alphabetical)
    AgentOutputToolInput,
    AgentsRetrievalResult,
    # API errors
    ApiError,
    ApiErrorDetail,
    ApiErrorResponse,
    ApiErrorSystemRecord,
    AskUserQuestionToolInput,
    AskUserQuestionToolResult,
    # Records
    AssistantRecord,
    AsyncTaskLaunchResult,
    BackgroundTask,
    BaseRecord,
    BashOutputToolInput,
    BashToolInput,
    BashToolResult,
    # Token usage
    CacheCreation,
    # Thinking edit
    ClearThinkingEdit,
    # Compact
    CompactBoundarySystemRecord,
    CompactMetadata,
    ConnectionError,
    # Context management
    ContextManagement,
    CustomTitleRecord,
    # Content types
    DocumentContent,
    DocumentSource,
    EditToolInput,
    EditToolResult,
    EmptyError,
    EnterPlanModeToolInput,
    EnterPlanModeToolResult,
    ExitPlanModeToolInput,
    ExitPlanModeToolResult,
    FileBackupInfo,
    FileHistorySnapshot,
    FileHistorySnapshotRecord,
    # File info
    FileInfo,
    GlobToolInput,
    GlobToolResult,
    GrepToolInput,
    GrepToolResult,
    HandoffCommandResult,
    ImageContent,
    ImageSource,
    InformationalSystemRecord,
    KillShellMessageResult,
    KillShellToolInput,
    KillShellToolResult,
    ListMcpResourcesToolInput,
    LocalCommandSystemRecord,
    McpResource,
    # Message
    Message,
    MessageContent,
    NetworkError,
    NotebookEditToolInput,
    PatchHunk,
    QuestionOption,
    QueueOperationRecord,
    ReadMcpResourceToolInput,
    ReadToolInput,
    ReadToolResult,
    ServerToolUse,
    SessionAnalysis,
    # Metadata/Analysis
    SessionMetadata,
    # Main union
    SessionRecord,
    SessionRecordAdapter,
    SkillToolInput,
    # Base
    StrictModel,
    SummaryRecord,
    SystemRecord,
    SystemSubtypeRecord,
    TaskOutputPollingResult,
    TaskOutputToolInput,
    TaskToolInput,
    TaskToolResult,
    TextContent,
    ThinkingContent,
    ThinkingMetadata,
    # Thinking metadata
    ThinkingTrigger,
    # Todo
    TodoItem,
    TodoToolResult,
    TodoWriteToolInput,
    TokenUsage,
    ToolInput,
    ToolResult,
    ToolResultContent,
    ToolResultContentBlock,
    ToolUseContent,
    TurnDurationSystemRecord,
    UnknownToolInput,
    UnknownToolResult,
    UserQuestion,
    UserRecord,
    WebFetchToolInput,
    WebFetchToolResult,
    WebSearchNestedResult,
    WebSearchResult,
    WebSearchResultWrapper,
    WebSearchToolInput,
    WebSearchToolResult,
    WriteToolInput,
    WriteToolResult,
    validated_copy,
)

# Expose submodules for qualified access (e.g., session.models.TokenUsage)
from . import markers, models

__all__ = [
    # Submodules (for qualified access like session.models.TokenUsage)
    'markers',
    'models',
    # Schema version
    'SCHEMA_VERSION',
    'CLAUDE_CODE_MIN_VERSION',
    'CLAUDE_CODE_MAX_VERSION',
    'LAST_VALIDATED',
    'VALIDATION_RECORD_COUNT',
    # Base
    'StrictModel',
    'validated_copy',
    # Content types
    'ThinkingContent',
    'TextContent',
    'ImageSource',
    'ImageContent',
    'DocumentSource',
    'DocumentContent',
    'ToolUseContent',
    'ToolResultContent',
    'ToolResultContentBlock',
    'MessageContent',
    'ClearThinkingEdit',
    # Tool inputs (alphabetical)
    'AgentOutputToolInput',
    'AskUserQuestionToolInput',
    'BashOutputToolInput',
    'BashToolInput',
    'EditToolInput',
    'EnterPlanModeToolInput',
    'ExitPlanModeToolInput',
    'GlobToolInput',
    'GrepToolInput',
    'KillShellToolInput',
    'ListMcpResourcesToolInput',
    'NotebookEditToolInput',
    'ReadMcpResourceToolInput',
    'ReadToolInput',
    'SkillToolInput',
    'TaskOutputToolInput',
    'TaskToolInput',
    'TodoWriteToolInput',
    'WebFetchToolInput',
    'WebSearchToolInput',
    'WriteToolInput',
    'ToolInput',
    'UnknownToolInput',  # Fallback for MCP tool inputs (enables isinstance checks)
    # Context management
    'ContextManagement',
    # Message
    'Message',
    # Token usage
    'CacheCreation',
    'ServerToolUse',
    'TokenUsage',
    # Thinking metadata
    'ThinkingTrigger',
    'ThinkingMetadata',
    # Todo
    'TodoItem',
    # Compact
    'CompactMetadata',
    # API errors
    'ApiErrorDetail',
    'ApiErrorResponse',
    'ApiError',
    'ConnectionError',
    'NetworkError',
    'EmptyError',
    # File info
    'FileInfo',
    'PatchHunk',
    # Tool results (alphabetical)
    'AgentCompletedState',
    'AgentsRetrievalResult',
    'AskUserQuestionToolResult',
    'AsyncTaskLaunchResult',
    'BackgroundTask',
    'BashToolResult',
    'EditToolResult',
    'EnterPlanModeToolResult',
    'ExitPlanModeToolResult',
    'GlobToolResult',
    'GrepToolResult',
    'HandoffCommandResult',
    'KillShellMessageResult',
    'KillShellToolResult',
    'McpResource',
    'QuestionOption',
    'ReadToolResult',
    'TaskOutputPollingResult',
    'TaskToolResult',
    'TodoToolResult',
    'UserQuestion',
    'WebFetchToolResult',
    'WebSearchNestedResult',
    'WebSearchResult',
    'WebSearchResultWrapper',
    'WebSearchToolResult',
    'WriteToolResult',
    'ToolResult',
    'UnknownToolResult',  # Fallback for MCP tool results (enables isinstance checks)
    # Records
    'BaseRecord',
    'UserRecord',
    'AssistantRecord',
    'SummaryRecord',
    'SystemRecord',
    'LocalCommandSystemRecord',
    'CompactBoundarySystemRecord',
    'ApiErrorSystemRecord',
    'InformationalSystemRecord',
    'TurnDurationSystemRecord',
    'SystemSubtypeRecord',
    'FileBackupInfo',
    'FileHistorySnapshot',
    'FileHistorySnapshotRecord',
    'QueueOperationRecord',
    'CustomTitleRecord',
    # Main union
    'SessionRecord',
    'SessionRecordAdapter',
    # Metadata/Analysis
    'SessionMetadata',
    'SessionAnalysis',
    # Markers
    'PathField',
    'PathListField',
    'PathMarker',
]
