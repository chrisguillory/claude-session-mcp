"""Document search integration for Claude Code sessions.

This package provides chunking and indexing support for session files.
Eventually moves to document-search MCP.
"""

from __future__ import annotations

from src.document_search.types import (
    ChunkConfig,
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
    'ChunkConfig',
    'ContentType',
    'LSPOperation',
    'RecordType',
    'SessionChunk',
    'SessionChunkMetadata',
    'ToolCategory',
    'ToolEvent',
    'ToolType',
]
