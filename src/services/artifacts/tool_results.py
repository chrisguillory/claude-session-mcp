"""
Tool results handling for session operations.

Handles:
- Tool result file discovery with extension validation
- Tool result collection from session directories
- Tool result writing to new session locations

Path pattern: ~/.claude/projects/<encoded-path>/<session-id>/tool-results/<tool-use-id>{.txt,.json}

Key insight: Tool results are nested under session_id, so no ID conflicts.
The tool_use_id can be preserved unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence, Set
from pathlib import Path
from typing import get_args

from src.schemas.base import StrictModel
from src.schemas.types import ToolResultExtension

TOOL_RESULT_EXTENSIONS: Set[str] = set(get_args(ToolResultExtension))


class ToolResultFile(StrictModel):
    """A tool result file with extension tracking.

    Uses Literal type for extension to fail fast on unknown file types.
    """

    tool_use_id: str
    content: str
    extension: ToolResultExtension

    @property
    def filename(self) -> str:
        return f'{self.tool_use_id}{self.extension}'


def get_tool_results_dir(project_folder: Path, session_id: str) -> Path:
    """
    Get the tool-results directory path for a session.

    Path structure: {project_folder}/{session_id}/tool-results/

    Args:
        project_folder: Path to the project folder under ~/.claude/projects/
        session_id: Session ID

    Returns:
        Path to tool-results directory (may not exist)
    """
    return project_folder / session_id / 'tool-results'


def collect_tool_results(
    project_folder: Path,
    session_id: str,
) -> Sequence[ToolResultFile]:
    """
    Collect tool result files for a session.

    Reads all files from the tool-results subdirectory, validates their
    extensions against TOOL_RESULT_EXTENSIONS, and returns typed results.

    If the tool-results directory doesn't exist or is empty, returns
    an empty sequence. This is NOT an error condition - many sessions
    don't have tool results stored (depends on tool types used).

    Raises FileNotFoundError if files with unknown extensions are found,
    preventing silent data loss in clone/move/archive operations.

    Args:
        project_folder: Path to the project folder under ~/.claude/projects/
        session_id: Session ID to collect tool results for

    Returns:
        Sequence of ToolResultFile objects

    Raises:
        FileNotFoundError: If files with unknown extensions exist in the
            tool-results directory (indicates Claude Code changed)
    """
    tool_results_dir = get_tool_results_dir(project_folder, session_id)

    if not tool_results_dir.exists():
        return ()

    results: list[ToolResultFile] = []
    unknown_files: list[Path] = []

    for path in sorted(tool_results_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix in TOOL_RESULT_EXTENSIONS:
            results.append(
                ToolResultFile.model_validate(
                    {
                        'tool_use_id': path.stem,
                        'content': path.read_text(encoding='utf-8'),
                        'extension': path.suffix,
                    }
                )
            )
        else:
            unknown_files.append(path)

    if unknown_files:
        file_list = '\n  '.join(str(p) for p in unknown_files)
        raise FileNotFoundError(
            f'Found {len(unknown_files)} tool result file(s) with unknown extensions:\n  {file_list}\n\n'
            f'Known extensions: {sorted(TOOL_RESULT_EXTENSIONS)}\n'
            f'Claude Code may have changed. Update TOOL_RESULT_EXTENSIONS to handle new file types.'
        )

    return results


def write_tool_results(
    tool_results: Sequence[ToolResultFile],
    target_dir: Path,
    new_session_id: str,
    *,
    exist_ok: bool = False,
) -> int:
    """Write tool result files to new session location.

    Creates the tool-results subdirectory structure and writes each
    tool result file with its original extension. Tool use IDs are
    preserved unchanged since they are nested under the session ID directory.

    Directory structure created:
        {target_dir}/{new_session_id}/tool-results/{tool_use_id}{extension}

    Args:
        tool_results: Sequence of ToolResultFile objects to write
        target_dir: Target project directory under ~/.claude/projects/
        new_session_id: New session ID for directory structure
        exist_ok: If True, silently overwrite existing files (for rollback).
                  If False (default), raise FileExistsError on collision.

    Returns:
        Number of files written

    Raises:
        FileExistsError: If exist_ok=False and a target file already exists
    """
    if not tool_results:
        return 0

    tool_results_dir = get_tool_results_dir(target_dir, new_session_id)
    tool_results_dir.mkdir(parents=True, exist_ok=True)

    for tr in tool_results:
        file_path = tool_results_dir / tr.filename
        if not exist_ok and file_path.exists():
            raise FileExistsError(
                f'Tool result file already exists: {file_path}\nThis indicates cloning into an existing session.'
            )
        file_path.write_text(tr.content, encoding='utf-8')

    return len(tool_results)
