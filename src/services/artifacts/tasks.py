"""
Tasks file handling for archive/restore operations.

Path pattern: ~/.claude/tasks/{session_id}/{id}.json

On restore:
- Tasks are only restored for --in-place (same session ID)
- Clone operations skip tasks (fresh start, no stale tasks)
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from src.schemas.session.models import Task

TASKS_DIR = Path.home() / '.claude' / 'tasks'


def iter_tasks(session_id: str) -> Iterator[Task]:
    """
    Iterate task files for a session.

    Scans ~/.claude/tasks/{session_id}/ for all files (except .lock).
    Parses each as JSON with strict Pydantic validation - fails if invalid.

    Yields nothing if directory doesn't exist.
    """
    tasks_dir = TASKS_DIR / session_id
    if not tasks_dir.exists():
        return

    for path in tasks_dir.iterdir():
        if path.name == '.lock':
            continue  # Claude Code convention - skip lock file

        yield Task.model_validate_json(path.read_text(encoding='utf-8'))


def write_tasks(session_id: str, tasks: Sequence[Task]) -> int:
    """
    Write task files for a session.

    Used during restore --in-place to restore archived tasks.
    Raises FileExistsError if task file already exists.
    """
    if not tasks:
        return 0

    tasks_dir = TASKS_DIR / session_id
    tasks_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        task_path = tasks_dir / f'{task.id}.json'

        if task_path.exists():
            raise FileExistsError(f'Task file already exists: {task_path}')

        task_path.write_text(task.model_dump_json(indent=2), encoding='utf-8')

    return len(tasks)
