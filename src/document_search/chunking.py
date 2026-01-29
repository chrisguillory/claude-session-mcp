"""Session chunking for document search integration.

This module provides the chunking logic for indexing Claude Code sessions.
Types are in types.py, helpers are in helpers.py.

Eventually moves to document-search MCP.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path

import src.schemas.session.models as models
from src.document_search.helpers import (
    extract_file_path,
    extract_text_from_message,
    format_tool_result_text,
    format_tool_use_text,
    get_lsp_operation,
    get_tool_category,
    get_tool_type,
    split_text,
)
from src.document_search.types import (
    ChunkConfig,
    ChunkContext,
    ContentType,
    LSPOperation,
    RecordType,
    SessionChunk,
    SessionChunkMetadata,
    ToolCategory,
    ToolEvent,
    ToolType,
)

__all__ = [
    # Types
    'ChunkConfig',
    'ChunkContext',
    'ContentType',
    'LSPOperation',
    'RecordType',
    'SessionChunk',
    'SessionChunkMetadata',
    'ToolCategory',
    'ToolEvent',
    'ToolType',
    # Helpers
    'extract_file_path',
    'extract_text_from_message',
    'format_tool_result_text',
    'format_tool_use_text',
    'get_lsp_operation',
    'get_tool_category',
    'get_tool_type',
    'split_text',
    # Chunking
    'iter_session_chunks',
]


# =============================================================================
# Main Entry Point
# =============================================================================


def iter_session_chunks(
    session_path: Path,
    config: ChunkConfig | None = None,
) -> Iterator[SessionChunk]:
    """Iterate over chunks from a session file.

    Memory-efficient generator that yields chunks one at a time.

    Args:
        session_path: Path to the session JSONL file
        config: Chunking configuration (uses defaults if not provided)

    Yields:
        SessionChunk objects ready for embedding
    """
    if config is None:
        config = ChunkConfig()

    ctx = ChunkContext.from_session_path(session_path)

    with session_path.open(encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            ctx.line_number = line_num
            raw = json.loads(line)
            record = models.SessionRecordAdapter.validate_python(raw)
            yield from _chunk_record(record, ctx, config)


# =============================================================================
# Record Dispatcher
# =============================================================================


def _chunk_record(
    record: models.SessionRecord,
    ctx: ChunkContext,
    config: ChunkConfig,
) -> Iterator[SessionChunk]:
    """Dispatch to appropriate chunker based on record type.

    EXHAUSTIVE: Every record type must be explicitly handled.
    If a new record type is added, this will raise ValueError.
    """
    match record:
        # === Core conversation records ===
        case models.UserRecord():
            if 'user' not in config.exclude_record_types:
                ctx.turn_index += 1
                if record.slug:
                    ctx.slug = record.slug
                ctx.timestamp = record.timestamp
                yield from _chunk_user_record(record, ctx, config)

        case (
            models.NormalMainAssistantRecord()
            | models.ErrorMainAssistantRecord()
            | models.NormalAgentAssistantRecord()
            | models.ErrorAgentAssistantRecord()
        ):
            if 'assistant' not in config.exclude_record_types:
                if record.slug:
                    ctx.slug = record.slug
                ctx.timestamp = record.timestamp
                yield from _chunk_assistant_record(record, ctx, config)

        case models.SummaryRecord():
            if 'summary' not in config.exclude_record_types:
                chunk = _chunk_summary_record(record, ctx)
                if chunk:
                    yield chunk

        # === System record subtypes ===
        case models.LocalCommandSystemRecord():
            if 'local_command' not in config.exclude_record_types:
                ctx.timestamp = record.timestamp
                chunk = _chunk_local_command_record(record, ctx)
                if chunk:
                    yield chunk

        case models.CompactBoundarySystemRecord():
            if 'compact_boundary' not in config.exclude_record_types:
                ctx.timestamp = record.timestamp
                chunk = _chunk_compact_boundary_record(record, ctx)
                if chunk:
                    yield chunk

        case models.MicrocompactBoundarySystemRecord():
            if 'microcompact_boundary' not in config.exclude_record_types:
                ctx.timestamp = record.timestamp
                chunk = _chunk_microcompact_boundary_record(record, ctx)
                if chunk:
                    yield chunk

        case models.ApiErrorSystemRecord():
            if 'api_error' not in config.exclude_record_types:
                ctx.timestamp = record.timestamp
                chunk = _chunk_api_error_record(record, ctx)
                if chunk:
                    yield chunk

        case models.InformationalSystemRecord():
            if 'informational' not in config.exclude_record_types:
                ctx.timestamp = record.timestamp
                chunk = _chunk_informational_record(record, ctx)
                if chunk:
                    yield chunk

        case models.TurnDurationSystemRecord():
            if 'turn_duration' not in config.exclude_record_types:
                ctx.timestamp = record.timestamp
                chunk = _chunk_turn_duration_record(record, ctx)
                if chunk:
                    yield chunk

        case models.StopHookSummarySystemRecord():
            if 'stop_hook_summary' not in config.exclude_record_types:
                ctx.timestamp = record.timestamp
                chunk = _chunk_stop_hook_summary_record(record, ctx)
                if chunk:
                    yield chunk

        case models.SystemRecord():
            # Generic system record fallback
            if 'system' not in config.exclude_record_types:
                ctx.timestamp = record.timestamp
                chunk = _chunk_generic_system_record(record, ctx)
                if chunk:
                    yield chunk

        # === Other records ===
        case models.FileHistorySnapshotRecord():
            if 'file_history_snapshot' not in config.exclude_record_types:
                chunk = _chunk_file_history_record(record, ctx)
                if chunk:
                    yield chunk

        case models.QueueOperationRecord():
            if 'queue_operation' not in config.exclude_record_types:
                ctx.timestamp = record.timestamp
                chunk = _chunk_queue_operation_record(record, ctx)
                if chunk:
                    yield chunk

        case models.CustomTitleRecord():
            if 'custom_title' not in config.exclude_record_types:
                chunk = _chunk_custom_title_record(record, ctx)
                if chunk:
                    yield chunk

        case models.ProgressRecord():
            if 'progress' not in config.exclude_record_types:
                chunk = _chunk_progress_record(record, ctx)
                if chunk:
                    yield chunk

        case _:
            # FAIL FAST: Unknown record type
            raise ValueError(f'Unhandled record type: {type(record).__name__}')


# =============================================================================
# User Record Chunking
# =============================================================================


def _chunk_user_record(
    record: models.UserRecord,
    ctx: ChunkContext,
    config: ChunkConfig,
) -> Iterator[SessionChunk]:
    """Chunk a user record (user message + tool results)."""
    # Extract user text
    if record.message:
        text = extract_text_from_message(record.message, include_thinking=False)
        if text and len(text) >= config.min_chunk_chars:
            # Split if too long
            for text_chunk in split_text(text, config.max_chunk_chars):
                if len(text_chunk) >= config.min_chunk_chars:
                    yield SessionChunk(
                        text=text_chunk,
                        source_path=str(ctx.session_path),
                        metadata=SessionChunkMetadata(
                            line_number=ctx.line_number,
                            turn_index=ctx.turn_index,
                            chunk_index=ctx.next_chunk_index(),
                            timestamp=ctx.timestamp,
                            record_type='user',
                            content_type='text',
                            session_id=ctx.session_id,
                            slug=ctx.slug,
                            project_path=ctx.project_path,
                        ),
                    )

        # Process tool results if configured
        if config.include_tool_results and isinstance(record.message.content, (list, tuple)):
            for content in record.message.content:
                if isinstance(content, models.ToolResultContent):
                    chunk = _chunk_tool_result(content, ctx, config)
                    if chunk:
                        yield chunk


# =============================================================================
# Assistant Record Chunking
# =============================================================================


def _chunk_assistant_record(
    record: models.MainAssistantRecord | models.AgentAssistantRecord,
    ctx: ChunkContext,
    config: ChunkConfig,
) -> Iterator[SessionChunk]:
    """Chunk an assistant record (text + thinking + tool invocations)."""
    if not record.message:
        return

    # Model is in the AssistantMessage
    model: str | None = None
    if isinstance(record.message, models.AssistantMessage):
        model = record.message.model
    content = record.message.content

    if isinstance(content, str):
        # Simple string content
        if len(content) >= config.min_chunk_chars:
            for text_chunk in split_text(content, config.max_chunk_chars):
                if len(text_chunk) >= config.min_chunk_chars:
                    yield SessionChunk(
                        text=text_chunk,
                        source_path=str(ctx.session_path),
                        metadata=SessionChunkMetadata(
                            line_number=ctx.line_number,
                            turn_index=ctx.turn_index,
                            chunk_index=ctx.next_chunk_index(),
                            timestamp=ctx.timestamp,
                            record_type='assistant',
                            content_type='text',
                            session_id=ctx.session_id,
                            slug=ctx.slug,
                            project_path=ctx.project_path,
                            model=model,
                        ),
                    )
        return

    if not isinstance(content, (list, tuple)):
        return

    # Process content blocks
    for block in content:
        match block:
            case models.TextContent():
                if len(block.text) >= config.min_chunk_chars:
                    for text_chunk in split_text(block.text, config.max_chunk_chars):
                        if len(text_chunk) >= config.min_chunk_chars:
                            yield SessionChunk(
                                text=text_chunk,
                                source_path=str(ctx.session_path),
                                metadata=SessionChunkMetadata(
                                    line_number=ctx.line_number,
                                    turn_index=ctx.turn_index,
                                    chunk_index=ctx.next_chunk_index(),
                                    timestamp=ctx.timestamp,
                                    record_type='assistant',
                                    content_type='text',
                                    session_id=ctx.session_id,
                                    slug=ctx.slug,
                                    project_path=ctx.project_path,
                                    model=model,
                                ),
                            )

            case models.ThinkingContent() if config.include_thinking:
                if len(block.thinking) >= config.min_chunk_chars:
                    for text_chunk in split_text(block.thinking, config.max_chunk_chars):
                        if len(text_chunk) >= config.min_chunk_chars:
                            yield SessionChunk(
                                text=text_chunk,
                                source_path=str(ctx.session_path),
                                metadata=SessionChunkMetadata(
                                    line_number=ctx.line_number,
                                    turn_index=ctx.turn_index,
                                    chunk_index=ctx.next_chunk_index(),
                                    timestamp=ctx.timestamp,
                                    record_type='assistant',
                                    content_type='thinking',
                                    session_id=ctx.session_id,
                                    slug=ctx.slug,
                                    project_path=ctx.project_path,
                                    model=model,
                                ),
                            )

            case models.ToolUseContent():
                chunk = _chunk_tool_use(block, ctx, model)
                if chunk:
                    yield chunk
                # Track for tool result correlation
                ctx.pending_tool_uses[block.id] = block


# =============================================================================
# Summary Record Chunking
# =============================================================================


def _chunk_summary_record(
    record: models.SummaryRecord,
    ctx: ChunkContext,
) -> SessionChunk | None:
    """Chunk a summary record (already condensed, high value)."""
    if not record.summary:
        return None

    return SessionChunk(
        text=record.summary,
        source_path=str(ctx.session_path),
        metadata=SessionChunkMetadata(
            line_number=ctx.line_number,
            turn_index=ctx.turn_index,
            chunk_index=ctx.next_chunk_index(),
            timestamp=ctx.timestamp,
            record_type='summary',
            content_type='text',
            session_id=ctx.session_id,
            slug=ctx.slug,
            project_path=ctx.project_path,
        ),
    )


# =============================================================================
# Tool Use Chunking
# =============================================================================


def _chunk_tool_use(
    tool_use: models.ToolUseContent,
    ctx: ChunkContext,
    model: str | None,
) -> SessionChunk | None:
    """Chunk a tool invocation."""
    # Format the tool use as searchable text
    text = format_tool_use_text(tool_use)
    if not text:
        return None

    # Get tool metadata
    tool_type = get_tool_type(tool_use.input)
    tool_category = get_tool_category(tool_use.input)
    file_path = extract_file_path(tool_use.input)
    lsp_operation = get_lsp_operation(tool_use.input)

    # Get MCP info if applicable
    mcp_server: str | None = None
    mcp_operation: str | None = None
    if tool_type == 'mcp':
        with contextlib.suppress(ValueError):
            mcp_server, mcp_operation = tool_use.mcp_info

    return SessionChunk(
        text=text,
        source_path=str(ctx.session_path),
        metadata=SessionChunkMetadata(
            line_number=ctx.line_number,
            turn_index=ctx.turn_index,
            chunk_index=ctx.next_chunk_index(),
            timestamp=ctx.timestamp,
            record_type='assistant',
            content_type='tool_use',
            tool_type=tool_type,
            tool_category=tool_category,
            tool_event='invocation',
            tool_use_id=tool_use.id,
            mcp_server=mcp_server,
            mcp_operation=mcp_operation,
            lsp_operation=lsp_operation,
            file_path=file_path,
            session_id=ctx.session_id,
            slug=ctx.slug,
            project_path=ctx.project_path,
            model=model,
        ),
    )


# =============================================================================
# Tool Result Chunking
# =============================================================================


def _chunk_tool_result(
    tool_result: models.ToolResultContent,
    ctx: ChunkContext,
    config: ChunkConfig,
) -> SessionChunk | None:
    """Chunk a tool result."""
    # Look up the original tool use for context
    tool_use = ctx.pending_tool_uses.get(tool_result.tool_use_id)

    # Format the result as searchable text
    text = format_tool_result_text(tool_result, tool_use)
    if len(text) < config.min_chunk_chars:
        return None

    # Get tool metadata from the original invocation
    tool_type: ToolType | None = None
    tool_category: ToolCategory | None = None
    file_path: str | None = None
    mcp_server: str | None = None
    mcp_operation: str | None = None
    lsp_operation: LSPOperation | None = None

    if tool_use:
        tool_type = get_tool_type(tool_use.input)
        tool_category = get_tool_category(tool_use.input)
        file_path = extract_file_path(tool_use.input)
        lsp_operation = get_lsp_operation(tool_use.input)

        if tool_type == 'mcp':
            with contextlib.suppress(ValueError):
                mcp_server, mcp_operation = tool_use.mcp_info

    return SessionChunk(
        text=text,
        source_path=str(ctx.session_path),
        metadata=SessionChunkMetadata(
            line_number=ctx.line_number,
            turn_index=ctx.turn_index,
            chunk_index=ctx.next_chunk_index(),
            timestamp=ctx.timestamp,
            record_type='user',
            content_type='tool_result',
            tool_type=tool_type,
            tool_category=tool_category,
            tool_event='result',
            tool_use_id=tool_result.tool_use_id,
            tool_result_error=tool_result.is_error,
            mcp_server=mcp_server,
            mcp_operation=mcp_operation,
            lsp_operation=lsp_operation,
            file_path=file_path,
            session_id=ctx.session_id,
            slug=ctx.slug,
            project_path=ctx.project_path,
        ),
    )


# =============================================================================
# System Record Chunking
# =============================================================================


def _chunk_local_command_record(
    record: models.LocalCommandSystemRecord,
    ctx: ChunkContext,
) -> SessionChunk | None:
    """Chunk a local command system record (e.g., /help output)."""
    if not record.content:
        return None

    text = f'Local command output:\n{record.content}'

    return SessionChunk(
        text=text,
        source_path=str(ctx.session_path),
        metadata=SessionChunkMetadata(
            line_number=ctx.line_number,
            turn_index=ctx.turn_index,
            chunk_index=ctx.next_chunk_index(),
            timestamp=ctx.timestamp,
            record_type='local_command',
            content_type='text',
            session_id=ctx.session_id,
            slug=ctx.slug,
            project_path=ctx.project_path,
        ),
    )


def _chunk_compact_boundary_record(
    record: models.CompactBoundarySystemRecord,
    ctx: ChunkContext,
) -> SessionChunk | None:
    """Chunk a compact boundary record."""
    parts = ['Conversation compacted']

    if record.content:
        parts.append(record.content)

    if record.compactMetadata:
        parts.append(f'Trigger: {record.compactMetadata.trigger}')
        if record.compactMetadata.preTokens:
            parts.append(f'Tokens before: {record.compactMetadata.preTokens}')

    text = '\n'.join(parts)

    return SessionChunk(
        text=text,
        source_path=str(ctx.session_path),
        metadata=SessionChunkMetadata(
            line_number=ctx.line_number,
            turn_index=ctx.turn_index,
            chunk_index=ctx.next_chunk_index(),
            timestamp=ctx.timestamp,
            record_type='compact_boundary',
            content_type='text',
            session_id=ctx.session_id,
            slug=ctx.slug,
            project_path=ctx.project_path,
        ),
    )


def _chunk_microcompact_boundary_record(
    record: models.MicrocompactBoundarySystemRecord,
    ctx: ChunkContext,
) -> SessionChunk | None:
    """Chunk a microcompact boundary record."""
    parts = ['Context microcompacted']

    if record.content:
        parts.append(record.content)

    if record.microcompactMetadata:
        if record.microcompactMetadata.tokensSaved:
            parts.append(f'Token savings: {record.microcompactMetadata.tokensSaved}')

    text = '\n'.join(parts)

    return SessionChunk(
        text=text,
        source_path=str(ctx.session_path),
        metadata=SessionChunkMetadata(
            line_number=ctx.line_number,
            turn_index=ctx.turn_index,
            chunk_index=ctx.next_chunk_index(),
            timestamp=ctx.timestamp,
            record_type='microcompact_boundary',
            content_type='text',
            session_id=ctx.session_id,
            slug=ctx.slug,
            project_path=ctx.project_path,
        ),
    )


def _chunk_api_error_record(
    record: models.ApiErrorSystemRecord,
    ctx: ChunkContext,
) -> SessionChunk | None:
    """Chunk an API error record."""
    parts = [f'API Error: {record.error}']

    if record.cause:
        parts.append(f'Cause: {record.cause}')
    if record.retryAttempt:
        parts.append(f'Retry attempt: {record.retryAttempt}/{record.maxRetries or "?"}')

    text = '\n'.join(parts)

    return SessionChunk(
        text=text,
        source_path=str(ctx.session_path),
        metadata=SessionChunkMetadata(
            line_number=ctx.line_number,
            turn_index=ctx.turn_index,
            chunk_index=ctx.next_chunk_index(),
            timestamp=ctx.timestamp,
            record_type='api_error',
            content_type='text',
            session_id=ctx.session_id,
            slug=ctx.slug,
            project_path=ctx.project_path,
        ),
    )


def _chunk_informational_record(
    record: models.InformationalSystemRecord,
    ctx: ChunkContext,
) -> SessionChunk | None:
    """Chunk an informational system record."""
    if not record.content:
        return None

    text = f'System info: {record.content}'

    return SessionChunk(
        text=text,
        source_path=str(ctx.session_path),
        metadata=SessionChunkMetadata(
            line_number=ctx.line_number,
            turn_index=ctx.turn_index,
            chunk_index=ctx.next_chunk_index(),
            timestamp=ctx.timestamp,
            record_type='informational',
            content_type='text',
            session_id=ctx.session_id,
            slug=ctx.slug,
            project_path=ctx.project_path,
        ),
    )


def _chunk_turn_duration_record(
    record: models.TurnDurationSystemRecord,
    ctx: ChunkContext,
) -> SessionChunk | None:
    """Chunk a turn duration record."""
    text = f'Turn duration: {record.durationMs}ms'

    return SessionChunk(
        text=text,
        source_path=str(ctx.session_path),
        metadata=SessionChunkMetadata(
            line_number=ctx.line_number,
            turn_index=ctx.turn_index,
            chunk_index=ctx.next_chunk_index(),
            timestamp=ctx.timestamp,
            record_type='turn_duration',
            content_type='text',
            session_id=ctx.session_id,
            slug=ctx.slug,
            project_path=ctx.project_path,
        ),
    )


def _chunk_stop_hook_summary_record(
    record: models.StopHookSummarySystemRecord,
    ctx: ChunkContext,
) -> SessionChunk | None:
    """Chunk a stop hook summary record."""
    parts = []

    if record.stopReason:
        parts.append(f'Stop reason: {record.stopReason}')

    if record.hookInfos:
        parts.extend(
            f'Hook command: {hook.command}' for hook in record.hookInfos if hasattr(hook, 'command') and hook.command
        )

    if record.hookErrors:
        parts.extend(f'Hook error: {err}' for err in record.hookErrors)

    text = '\n'.join(parts)
    if not text:
        return None

    return SessionChunk(
        text=text,
        source_path=str(ctx.session_path),
        metadata=SessionChunkMetadata(
            line_number=ctx.line_number,
            turn_index=ctx.turn_index,
            chunk_index=ctx.next_chunk_index(),
            timestamp=ctx.timestamp,
            record_type='stop_hook_summary',
            content_type='text',
            session_id=ctx.session_id,
            slug=ctx.slug,
            project_path=ctx.project_path,
        ),
    )


def _chunk_generic_system_record(
    record: models.SystemRecord,
    ctx: ChunkContext,
) -> SessionChunk | None:
    """Chunk a generic system record."""
    if not record.message:
        return None

    text = f'System ({record.systemType}): {record.message}'

    return SessionChunk(
        text=text,
        source_path=str(ctx.session_path),
        metadata=SessionChunkMetadata(
            line_number=ctx.line_number,
            turn_index=ctx.turn_index,
            chunk_index=ctx.next_chunk_index(),
            timestamp=ctx.timestamp,
            record_type='system',
            content_type='text',
            session_id=ctx.session_id,
            slug=ctx.slug,
            project_path=ctx.project_path,
        ),
    )


# =============================================================================
# Other Record Chunking
# =============================================================================


def _chunk_file_history_record(
    record: models.FileHistorySnapshotRecord,
    ctx: ChunkContext,
) -> SessionChunk | None:
    """Chunk a file history snapshot record."""
    if not record.snapshot or not record.snapshot.trackedFileBackups:
        return None

    files = list(record.snapshot.trackedFileBackups.keys())
    text = 'File history snapshot:\n' + '\n'.join(f'  - {f}' for f in files[:20])
    if len(files) > 20:
        text += f'\n  ... and {len(files) - 20} more files'

    return SessionChunk(
        text=text,
        source_path=str(ctx.session_path),
        metadata=SessionChunkMetadata(
            line_number=ctx.line_number,
            turn_index=ctx.turn_index,
            chunk_index=ctx.next_chunk_index(),
            timestamp='',  # No timestamp on this record
            record_type='file_history_snapshot',
            content_type='text',
            session_id=ctx.session_id,
            slug=ctx.slug,
            project_path=ctx.project_path,
        ),
    )


def _chunk_queue_operation_record(
    record: models.QueueOperationRecord,
    ctx: ChunkContext,
) -> SessionChunk | None:
    """Chunk a queue operation record."""
    parts = [f'Queue operation: {record.operation}']

    content = record.content
    if content:
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, (list, tuple)):
            parts.extend(block.text for block in content if hasattr(block, 'text'))

    text = '\n'.join(parts)

    return SessionChunk(
        text=text,
        source_path=str(ctx.session_path),
        metadata=SessionChunkMetadata(
            line_number=ctx.line_number,
            turn_index=ctx.turn_index,
            chunk_index=ctx.next_chunk_index(),
            timestamp=ctx.timestamp,
            record_type='queue_operation',
            content_type='text',
            session_id=ctx.session_id,
            slug=ctx.slug,
            project_path=ctx.project_path,
        ),
    )


def _chunk_custom_title_record(
    record: models.CustomTitleRecord,
    ctx: ChunkContext,
) -> SessionChunk | None:
    """Chunk a custom title record."""
    if not record.customTitle:
        return None

    text = f'Session title: {record.customTitle}'

    return SessionChunk(
        text=text,
        source_path=str(ctx.session_path),
        metadata=SessionChunkMetadata(
            line_number=ctx.line_number,
            turn_index=ctx.turn_index,
            chunk_index=ctx.next_chunk_index(),
            timestamp='',  # No timestamp on this record
            record_type='custom_title',
            content_type='text',
            session_id=ctx.session_id,
            slug=ctx.slug,
            project_path=ctx.project_path,
        ),
    )


def _chunk_progress_record(
    record: models.ProgressRecord,
    ctx: ChunkContext,
) -> SessionChunk | None:
    """Chunk a progress record."""
    data = record.data
    if not data:
        return None

    # Get progress type from data.type
    progress_type = getattr(data, 'type', 'unknown')
    parts = [f'Progress ({progress_type})']

    if data:
        # Extract searchable content from progress data
        if hasattr(data, 'prompt') and data.prompt:
            parts.append(f'Prompt: {data.prompt}')
        if hasattr(data, 'output') and data.output:
            output = data.output[:500] if len(data.output) > 500 else data.output
            parts.append(f'Output: {output}')
        if hasattr(data, 'query') and data.query:
            parts.append(f'Query: {data.query}')
        if hasattr(data, 'command') and data.command:
            parts.append(f'Command: {data.command}')
        if hasattr(data, 'taskDescription') and data.taskDescription:
            parts.append(f'Task: {data.taskDescription}')
        if hasattr(data, 'toolName') and data.toolName:
            parts.append(f'Tool: {data.toolName}')
        if hasattr(data, 'serverName') and data.serverName:
            parts.append(f'Server: {data.serverName}')

    text = '\n'.join(parts)

    return SessionChunk(
        text=text,
        source_path=str(ctx.session_path),
        metadata=SessionChunkMetadata(
            line_number=ctx.line_number,
            turn_index=ctx.turn_index,
            chunk_index=ctx.next_chunk_index(),
            timestamp='',  # Progress records may not have timestamp
            record_type='progress',
            content_type='text',
            session_id=ctx.session_id,
            slug=ctx.slug,
            project_path=ctx.project_path,
        ),
    )
