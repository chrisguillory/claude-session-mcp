#!/usr/bin/env python3
"""Test SessionRepository on a real session file.

Usage:
    uv run scripts/test_repository.py [session_id]

If no session_id provided, uses the most recent session.
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.repositories import SessionRepository


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

    print(f'Session file: {session_path.name}')
    print(f'Project dir: {session_path.parent.name}')
    print()

    # Instantiate repository
    repo = SessionRepository(session_path)

    # === Session Info ===
    print('=== Session Info ===')
    info = repo.session_info
    print(f'  ID: {info.session_id}')
    print(f'  Project: {info.project_path}')
    print(f'  Slug: {info.slug}')
    print(f'  Lines: {info.line_count}')
    print(f'  Size: {info.file_size_bytes / 1024:.1f} KB')
    print(f'  First timestamp: {info.first_timestamp}')
    print(f'  Last timestamp: {info.last_timestamp}')
    print()

    # === Session Stats ===
    print('=== Session Stats ===')
    stats = repo.session_stats
    print(f'  Turns: {stats.turn_count}')
    print(f'  User messages: {stats.user_message_count}')
    print(f'  Assistant messages: {stats.assistant_message_count}')
    print(f'  Thinking blocks: {stats.thinking_block_count}')
    print(f'  Summaries: {stats.summary_count}')
    print()
    print(f'  Tool invocations: {stats.tool_invocation_count}')
    print(f'  Tool errors: {stats.tool_error_count}')
    if stats.tool_by_name:
        print('  Tool by name:')
        for name, count in sorted(stats.tool_by_name.items(), key=lambda x: -x[1]):
            print(f'    {name}: {count}')
    print()
    print(f'  Progress events: {stats.progress_event_count}')
    if stats.progress_by_type:
        print('  Progress by type:')
        for ptype, count in sorted(stats.progress_by_type.items(), key=lambda x: -x[1]):
            print(f'    {ptype}: {count}')
    print()
    if stats.total_duration_ms:
        print(f'  Total duration: {stats.total_duration_ms / 1000:.1f}s')
    print(f'  Unique files touched: {len(stats.unique_file_paths)}')
    print()

    # === Sample Turns ===
    print('=== Sample Turns ===')
    turns = repo.turns
    samples = [turns[0], turns[len(turns) // 2], turns[-1]] if len(turns) >= 3 else turns

    for turn in samples:
        print(f'\nTurn {turn.index} (line {turn.user_line}):')
        user_preview = turn.user_text[:100].replace('\n', ' ')
        print(f'  User: {user_preview}{"..." if len(turn.user_text) > 100 else ""}')

        if turn.assistant_text:
            asst_preview = turn.assistant_text[:100].replace('\n', ' ')
            print(f'  Assistant: {asst_preview}{"..." if len(turn.assistant_text) > 100 else ""}')

        if turn.tool_uses:
            print(f'  Tools: {len(turn.tool_uses)}')
            for tool in turn.tool_uses[:3]:
                result_info = ''
                if tool.result:
                    result_info = f' -> {tool.result.content_length} chars'
                    if tool.result.is_error:
                        result_info += ' [ERROR]'
                print(f'    - {tool.name}{result_info}')
            if len(turn.tool_uses) > 3:
                print(f'    ... and {len(turn.tool_uses) - 3} more')

        if turn.thinking_texts:
            print(f'  Thinking blocks: {len(turn.thinking_texts)}')

        if turn.duration_ms:
            print(f'  Duration: {turn.duration_ms / 1000:.1f}s')
    print()

    # === Tool Use/Result Linking ===
    print('=== Tool Use/Result Linking ===')
    total_uses = len(repo.tool_uses)
    linked = [u for u in repo.tool_uses if u.result]
    errors = [u for u in linked if u.result and u.result.is_error]
    print(f'  Total tool uses: {total_uses}')
    print(f'  With linked results: {len(linked)}')
    print(f'  With errors: {len(errors)}')

    if errors:
        print('\n  Sample errors:')
        for use in errors[:3]:
            content_preview = str(use.result.content)[:80] if use.result else ''
            print(f'    {use.name} (line {use.line_number}): {content_preview}...')
    print()

    # === MCP Tool Uses ===
    mcp_uses = repo.get_mcp_tool_uses()
    if mcp_uses:
        print('=== MCP Tool Uses ===')
        by_server: dict[str, int] = {}
        for use in mcp_uses:
            server = use.mcp_server or 'unknown'
            by_server[server] = by_server.get(server, 0) + 1

        for server, count in sorted(by_server.items(), key=lambda x: -x[1]):
            print(f'  {server}: {count}')

        print('\n  Sample MCP calls:')
        for use in mcp_uses[:5]:
            print(f'    {use.mcp_server}/{use.mcp_operation} (line {use.line_number})')
        print()

    # === Random Access ===
    print('=== Random Access ===')
    line_num = info.line_count // 2
    record = repo.get_line(line_num)
    print(f'  Line {line_num}: {type(record).__name__}')

    # Get a range
    start, end = 10, 15
    records = repo.get_lines(start, end)
    print(f'  Lines {start}-{end}: {[type(r).__name__ for r in records]}')
    print()

    # === Progress Events ===
    print('=== Progress Events ===')
    events = repo.progress_events
    print(f'  Total: {len(events)}')
    if events:
        print('\n  Sample events:')
        for event in events[:5]:
            extra = ''
            if event.tool_name:
                extra += f' tool={event.tool_name}'
            if event.server_name:
                extra += f' server={event.server_name}'
            if event.command:
                extra += f' cmd={event.command[:30]}...'
            print(f'    {event.progress_type} (line {event.line_number}){extra}')


if __name__ == '__main__':
    main()
