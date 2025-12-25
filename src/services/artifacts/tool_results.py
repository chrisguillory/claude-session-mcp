"""
Tool results handling for clone/restore operations.

Handles:
- Tool result file discovery
- Tool result collection from session directories
- Tool result writing to new session locations

Path pattern: ~/.claude/projects/<encoded-path>/<session-id>/tool-results/<tool-use-id>.txt

Key insight: Tool results are nested under session_id, so no ID conflicts.
The tool_use_id can be preserved unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


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
) -> Mapping[str, str]:
    """
    Collect tool result contents for a session.

    Reads all .txt files from the tool-results subdirectory and returns
    their contents keyed by tool_use_id (filename without extension).

    If the tool-results directory doesn't exist or is empty, returns
    an empty mapping. This is NOT an error condition - many sessions
    don't have tool results stored (depends on tool types used).

    Args:
        project_folder: Path to the project folder under ~/.claude/projects/
        session_id: Session ID to collect tool results for

    Returns:
        Mapping of tool_use_id -> file content (only for existing files)
    """
    tool_results_dir = get_tool_results_dir(project_folder, session_id)

    if not tool_results_dir.exists():
        return {}

    results: dict[str, str] = {}
    for path in tool_results_dir.glob('*.txt'):
        tool_use_id = path.stem  # filename without extension
        results[tool_use_id] = path.read_text(encoding='utf-8')

    return results


def write_tool_results(
    tool_results: Mapping[str, str],
    target_dir: Path,
    new_session_id: str,
) -> int:
    """
    Write tool result files to new session location.

    Creates the tool-results subdirectory structure and writes each
    tool result file. Tool use IDs are preserved unchanged since they
    are nested under the session ID directory.

    Directory structure created:
    {target_dir}/{new_session_id}/tool-results/{tool_use_id}.txt

    Args:
        tool_results: Mapping of tool_use_id -> content from archive/source
        target_dir: Target project directory under ~/.claude/projects/
        new_session_id: New session ID for directory structure

    Returns:
        Number of files written

    Raises:
        FileExistsError: If tool-results directory already contains files
            (indicates cloning into existing session - should not happen)
    """
    if not tool_results:
        return 0

    tool_results_dir = get_tool_results_dir(target_dir, new_session_id)

    # Check for existing files
    if tool_results_dir.exists() and any(tool_results_dir.iterdir()):
        raise FileExistsError(
            f'Tool results directory already contains files: {tool_results_dir}\n'
            'This indicates cloning into an existing session.'
        )

    # Create directory structure
    tool_results_dir.mkdir(parents=True, exist_ok=True)

    # Write files
    for tool_use_id, content in tool_results.items():
        file_path = tool_results_dir / f'{tool_use_id}.txt'
        file_path.write_text(content, encoding='utf-8')

    return len(tool_results)
