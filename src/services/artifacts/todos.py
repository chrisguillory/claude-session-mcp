"""
Todos file handling for clone/restore operations.

Handles:
- Todos file discovery by session ID
- Todos collection from ~/.claude/todos/
- Filename transformation for session ID updates
- Todos writing to new locations

Path pattern: ~/.claude/todos/<primary-sid>-agent-<agent-sid>.json

Filename anatomy:
- primary-sid: The main session ID (first UUID)
- agent-sid: The agent session ID (second UUID, after "agent-")

On clone/restore:
- Only the primary-sid portion is updated to new_session_id
- The agent-sid portion stays unchanged (preserves agent identity)

Example:
  Original: 019b4c64-5776-7f79-b97f-4c9da77b6085-agent-97e6bec4-a486-4587-8835-6829f56bc8dd.json
  Cloned:   019b5190-a18e-7888-aaa9-65bff7ed2110-agent-97e6bec4-a486-4587-8835-6829f56bc8dd.json
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

TODOS_DIR = Path.home() / '.claude' / 'todos'


def collect_todos(session_id: str) -> Mapping[str, str]:
    """
    Collect todos files for a session.

    Scans ~/.claude/todos/ for files matching the pattern:
    {session_id}-agent-*.json

    Note: Only collects files where session_id is the PRIMARY session ID
    (appears before "agent-"). Agent-originated todos that reference
    this session are NOT collected - they belong to the agent's session.

    If the todos directory doesn't exist or contains no matching files,
    returns an empty mapping. This is NOT an error condition.

    Args:
        session_id: Session ID to collect todos for

    Returns:
        Mapping of original_filename -> JSON content string
    """
    if not TODOS_DIR.exists():
        return {}

    results: dict[str, str] = {}
    pattern = f'{session_id}-agent-*.json'

    for path in TODOS_DIR.glob(pattern):
        results[path.name] = path.read_text(encoding='utf-8')

    return results


def transform_todo_filename(
    old_filename: str,
    old_session_id: str,
    new_session_id: str,
) -> str:
    """
    Transform a todo filename to use the new session ID.

    Only replaces the primary session ID portion (before "agent-").
    The agent session ID portion is preserved.

    This is critical for maintaining the relationship between todos
    and their originating agents while updating the parent session ID.

    Args:
        old_filename: Original filename (e.g., "old-sid-agent-agent-sid.json")
        old_session_id: Original session ID to replace
        new_session_id: New session ID

    Returns:
        Transformed filename

    Raises:
        ValueError: If old_session_id not found at start of filename
    """
    if not old_filename.startswith(old_session_id):
        raise ValueError(f'Expected filename to start with session ID {old_session_id}, got: {old_filename}')

    # Replace only the first occurrence (the primary session ID)
    return old_filename.replace(old_session_id, new_session_id, 1)


def write_todos(
    todos: Iterable[tuple[str, str]],
    *,
    exist_ok: bool = False,
) -> int:
    """Write todo files to ~/.claude/todos/.

    Mode-agnostic writer — caller computes target filenames.

    Examples:
        # Clone (transform_todo_filename remaps session ID):
        write_todos((transform_todo_filename(f, old_sid, new_sid), c) for f, c in todos.items())
        # Restore non-in-place (remap both session ID and agent ID):
        write_todos((f'{new_sid}-agent-{new_agent_id}.json', c) for ...)
        # In-place/rollback (original filenames):
        write_todos((f'{sid}-agent-{agent_id}.json', c) for ..., exist_ok=True)

    Args:
        todos: (target_filename, content) pairs
        exist_ok: If True, silently overwrite existing files (for rollback).
                  If False (default), raise FileExistsError on collision.

    Returns:
        Number of files written

    Raises:
        FileExistsError: If exist_ok=False and a target file already exists
    """
    count = 0
    for filename, content in todos:
        if count == 0:
            TODOS_DIR.mkdir(parents=True, exist_ok=True)
        todo_path = TODOS_DIR / filename
        if not exist_ok and todo_path.exists():
            raise FileExistsError(f'Todo file already exists: {todo_path}\nThis indicates a session ID collision.')
        todo_path.write_text(content, encoding='utf-8')
        count += 1
    return count
