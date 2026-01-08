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
    AgentOutputToolInput,
    ApiError,
    # API errors
    ApiErrorDetail,
    ApiErrorResponse,
    ApiErrorSystemRecord,
    AskUserQuestionToolResult,
    AssistantRecord,
    # Records
    BaseRecord,
    # Tool results
    BashToolResult,
    # Token usage
    CacheCreation,
    CompactBoundarySystemRecord,
    # Compact
    CompactMetadata,
    ConnectionError,
    # Context management
    ContextManagement,
    CustomTitleRecord,
    DocumentContent,
    DocumentSource,
    EditToolInput,
    EditToolResult,
    EmptyError,
    EnterPlanModeToolInput,
    ExitPlanModeToolResult,
    FileBackupInfo,
    FileHistorySnapshot,
    FileHistorySnapshotRecord,
    # File info
    FileInfo,
    GlobToolResult,
    GrepToolResult,
    ImageContent,
    ImageSource,
    InformationalSystemRecord,
    KillShellToolResult,
    LocalCommandSystemRecord,
    McpResource,
    # Message
    Message,
    MessageContent,
    NetworkError,
    PatchHunk,
    QuestionOption,
    QueueOperationRecord,
    # Tool inputs
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
    TaskOutputToolInput,
    TaskToolResult,
    TextContent,
    # Content types
    ThinkingContent,
    ThinkingMetadata,
    # Thinking metadata
    ThinkingTrigger,
    # Todo
    TodoItem,
    TodoToolResult,
    TokenUsage,
    ToolInput,
    ToolResult,
    ToolResultContent,
    ToolResultContentBlock,
    ToolUseContent,
    UnknownToolInput,
    UnknownToolResult,
    UserQuestion,
    UserRecord,
    WebFetchToolResult,
    WebSearchResult,
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
    # Tool inputs
    'ReadToolInput',
    'WriteToolInput',
    'EditToolInput',
    'SkillToolInput',
    'EnterPlanModeToolInput',
    'AgentOutputToolInput',
    'TaskOutputToolInput',
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
    # Tool results
    'BashToolResult',
    'ReadToolResult',
    'GlobToolResult',
    'GrepToolResult',
    'EditToolResult',
    'WriteToolResult',
    'TodoToolResult',
    'TaskToolResult',
    'QuestionOption',
    'UserQuestion',
    'AskUserQuestionToolResult',
    'WebSearchResult',
    'WebSearchToolResult',
    'WebFetchToolResult',
    'ExitPlanModeToolResult',
    'McpResource',
    'KillShellToolResult',
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
