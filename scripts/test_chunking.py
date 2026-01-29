#!/usr/bin/env python3
"""Test session chunking on a real session file.

Usage:
    uv run scripts/test_chunking.py [session_id]

If no session_id provided, uses the current session (requires claude-session MCP).
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.document_search.chunking import ChunkConfig, iter_session_chunks


def find_session_file(session_id: str | None = None) -> Path | None:
    """Find a session file by ID or return the most recent."""
    session_dir = Path.home() / '.claude/projects'

    if session_id:
        # Search for matching session
        for project_dir in session_dir.iterdir():
            if not project_dir.is_dir():
                continue
            for f in project_dir.glob(f'{session_id}*.jsonl'):
                if not f.name.startswith('agent-'):
                    return f
        return None

    # Find most recent session
    sessions = [
        f
        for project_dir in session_dir.iterdir()
        if project_dir.is_dir()
        for f in project_dir.glob('*.jsonl')
        if not f.name.startswith('agent-')
    ]

    if sessions:
        return max(sessions, key=lambda p: p.stat().st_mtime)
    return None


def main() -> None:
    session_id = sys.argv[1] if len(sys.argv) > 1 else None

    session_path = find_session_file(session_id)
    if not session_path:
        print(f'No session found{f" matching {session_id}" if session_id else ""}')
        sys.exit(1)

    print(f'Session: {session_path.name}')
    print(f'Project: {session_path.parent.name}')
    print(f'Size: {session_path.stat().st_size / 1024:.1f} KB')
    print()

    # Chunk with default config
    config = ChunkConfig()
    chunks = list(iter_session_chunks(session_path, config))

    print(f'Generated {len(chunks)} chunks')
    print()

    # Stats by content type
    by_type: dict[str, int] = {}
    for c in chunks:
        ct = c.metadata.content_type
        by_type[ct] = by_type.get(ct, 0) + 1
    print('By content_type:')
    for ct_str, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f'  {ct_str}: {count}')
    print()

    # Stats by tool category
    by_category: dict[str, int] = {}
    for c in chunks:
        if c.metadata.tool_category:
            cat = c.metadata.tool_category
            by_category[cat] = by_category.get(cat, 0) + 1
    if by_category:
        print('By tool_category:')
        for cat_str, count in sorted(by_category.items(), key=lambda x: -x[1]):
            print(f'  {cat_str}: {count}')
        print()

    # Stats by MCP server
    mcp_servers: dict[str, int] = {}
    for c in chunks:
        if c.metadata.mcp_server:
            srv = c.metadata.mcp_server
            mcp_servers[srv] = mcp_servers.get(srv, 0) + 1
    if mcp_servers:
        print('By MCP server:')
        for srv, count in sorted(mcp_servers.items(), key=lambda x: -x[1]):
            print(f'  {srv}: {count}')
        print()

    # Sample chunks
    print('--- Sample Chunks ---')

    # User message
    user_chunks = [c for c in chunks if c.metadata.record_type == 'user' and c.metadata.content_type == 'text']
    if user_chunks:
        sample = user_chunks[0]
        print(f'\n[User message, turn {sample.metadata.turn_index}]')
        print(f'{sample.text[:200]}{"..." if len(sample.text) > 200 else ""}')

    # Assistant text
    assistant_chunks = [
        c for c in chunks if c.metadata.record_type == 'assistant' and c.metadata.content_type == 'text'
    ]
    if assistant_chunks:
        sample = assistant_chunks[len(assistant_chunks) // 2]
        print(f'\n[Assistant text, turn {sample.metadata.turn_index}, model={sample.metadata.model}]')
        print(f'{sample.text[:200]}{"..." if len(sample.text) > 200 else ""}')

    # MCP tool
    mcp_chunks = [c for c in chunks if c.metadata.mcp_server]
    if mcp_chunks:
        sample = mcp_chunks[0]
        print(f'\n[MCP tool: {sample.metadata.mcp_server}/{sample.metadata.mcp_operation}]')
        print(f'{sample.text[:200]}{"..." if len(sample.text) > 200 else ""}')

    # File operation
    file_chunks = [c for c in chunks if c.metadata.file_path]
    if file_chunks:
        sample = file_chunks[0]
        print(f'\n[File op: {sample.metadata.tool_type} -> {sample.metadata.file_path}]')
        print(f'{sample.text[:200]}{"..." if len(sample.text) > 200 else ""}')


if __name__ == '__main__':
    main()
