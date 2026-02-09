"""
Tasks file handling for archive/restore/clone operations.

Path pattern: ~/.claude/tasks/{session_id}/{id}.json

On restore/clone:
- Tasks are written to the target session ID (original for in-place, new for clone)
- Task metadata (.highwatermark) is always written alongside task data
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.schemas.session.models import Task

TASKS_DIR = Path.home() / '.claude' / 'tasks'

# Known metadata files in task directories (not task data)
# .lock: process coordination file
# .highwatermark: next task ID counter (contains integer like "5")
TASK_METADATA_FILES = frozenset({'.lock', '.highwatermark'})


@dataclass(frozen=True)
class TaskDirectoryContents:
    """Classification of all files in a task session directory.

    Single-pass enumeration that separates task data files from
    metadata files and flags anything unexpected for fail-fast behavior.
    """

    task_files: list[Path]  # {id}.json data files
    metadata_files: list[Path]  # .lock, .highwatermark
    unexpected_files: list[Path]  # Anything else (signals bug in task system)


def classify_task_directory(session_id: str) -> TaskDirectoryContents | None:
    """Classify all files in a task directory in a single pass.

    Returns None if the directory doesn't exist.

    Examples:
        Directory with 1.json, 2.json, .lock, .highwatermark:
            task_files = [1.json, 2.json]
            metadata_files = [.lock, .highwatermark]
            unexpected_files = []

        Directory with task-old.json (unexpected):
            unexpected_files = [task-old.json]
    """
    tasks_dir = TASKS_DIR / session_id
    if not tasks_dir.exists():
        return None

    result = TaskDirectoryContents(task_files=[], metadata_files=[], unexpected_files=[])
    for path in tasks_dir.iterdir():
        if not path.is_file():
            continue
        if path.name in TASK_METADATA_FILES:
            result.metadata_files.append(path)
        elif path.suffix == '.json' and path.stem.isdigit():
            result.task_files.append(path)
        else:
            result.unexpected_files.append(path)
    return result


def iter_task_paths(session_id: str) -> Iterator[Path]:
    """Yield paths to task data files ({id}.json where id is positive integer)."""
    contents = classify_task_directory(session_id)
    if contents is not None:
        yield from contents.task_files


def iter_tasks(session_id: str) -> Iterator[Task]:
    """
    Iterate parsed Task objects for a session.

    Delegates to iter_task_paths() for file identification, then parses each.
    """
    for path in iter_task_paths(session_id):
        yield Task.model_validate_json(path.read_text(encoding='utf-8'))


def collect_task_metadata(session_id: str) -> Mapping[str, str]:
    """Collect archivable task metadata files (excludes .lock).

    Returns mapping of filename -> content for metadata files that
    need to survive archive/restore cycles (.highwatermark).

    .lock is ephemeral (runtime coordination) and excluded from backup.
    """
    contents = classify_task_directory(session_id)
    if contents is None:
        return {}

    result: dict[str, str] = {}
    for path in contents.metadata_files:
        if path.name == '.lock':
            continue  # Ephemeral, never archived
        result[path.name] = path.read_text(encoding='utf-8')
    return result


def write_task_metadata(
    session_id: str,
    metadata: Mapping[str, str],
    *,
    exist_ok: bool = False,
) -> int:
    """Write task metadata files (.highwatermark, etc.) for a session.

    Used during restore, clone, and delete rollback.

    Args:
        session_id: Target session ID
        metadata: Mapping of filename -> content
        exist_ok: If True, silently overwrite existing files (for rollback).
                  If False (default), raise FileExistsError on collision.
    """
    if not metadata:
        return 0

    tasks_dir = TASKS_DIR / session_id
    tasks_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in metadata.items():
        file_path = tasks_dir / filename
        if not exist_ok and file_path.exists():
            raise FileExistsError(f'Task metadata file already exists: {file_path}')
        file_path.write_text(content, encoding='utf-8')

    return len(metadata)


def write_tasks(
    session_id: str,
    tasks: Sequence[Task],
    *,
    exist_ok: bool = False,
) -> int:
    """Write task data files ({id}.json) for a session.

    Used during restore, clone, and delete rollback.

    Args:
        session_id: Target session ID
        tasks: Task objects to write
        exist_ok: If True, silently overwrite existing files (for rollback).
                  If False (default), raise FileExistsError on collision.
    """
    if not tasks:
        return 0

    tasks_dir = TASKS_DIR / session_id
    tasks_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        task_path = tasks_dir / f'{task.id}.json'
        if not exist_ok and task_path.exists():
            raise FileExistsError(f'Task file already exists: {task_path}')
        task_path.write_text(task.model_dump_json(indent=2), encoding='utf-8')

    return len(tasks)
