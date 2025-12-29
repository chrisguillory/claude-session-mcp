"""
Path encoding utilities for Claude Code session management.

Claude Code encodes paths for directory names by replacing:
- `/` -> `-`
- `.` -> `-`
- ` ` -> `-`
- `~` -> `-`

WARNING: This encoding is LOSSY - decoding is impossible.
To get the real path, read the `cwd` field from session records
using `extract_source_project_path()` from `src.services.artifacts`.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ['encode_path']


def encode_path(path: Path | str) -> str:
    """
    Encode path for Claude's directory naming.

    This is the ONLY direction encoding can go. There is no decode function
    because the encoding is lossy (4 chars -> 1 char).

    To get the original path:
    - From a session file: Use extract_source_project_path() to read cwd field
    - From caller context: The caller should already know the project_path

    Args:
        path: Filesystem path to encode

    Returns:
        Encoded string for use as directory name in ~/.claude/projects/

    Examples:
        >>> encode_path("/Users/chris/project")
        '-Users-chris-project'

        >>> encode_path("/Users/chris/My Project.app")
        '-Users-chris-My-Project-app'
    """
    result = str(path) if isinstance(path, Path) else path
    for char in ['/', '.', ' ', '~']:
        result = result.replace(char, '-')
    return result
