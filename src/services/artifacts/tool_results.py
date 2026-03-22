"""
Tool results handling for session operations.

Handles:
- Tool result file/directory discovery with extension validation
- Tool result collection from session directories (flat files + subdirectories)
- Tool result writing to new session locations

Path patterns:
    Flat files:   ~/.claude/projects/<path>/<session-id>/tool-results/<tool-use-id>{ext}
    Directories:  ~/.claude/projects/<path>/<session-id>/tool-results/pdf-<uuid>/page-NN.jpg

Key insight: Tool results are nested under session_id, so no ID conflicts.
The tool_use_id can be preserved unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence, Set
from pathlib import Path
from typing import NamedTuple, cast, get_args

from src.schemas.base import StrictModel
from src.schemas.types import Base64JsonBytes, ToolResultExtension

logger = logging.getLogger(__name__)

TOOL_RESULT_EXTENSIONS: Set[str] = set(get_args(ToolResultExtension))

# Known directory prefixes in tool-results/. Used for logging only — unknown prefixes
# are NOT rejected (directory names are organizational, not data-format indicators).
KNOWN_DIR_PREFIXES = ('pdf-',)


# ==============================================================================
# Service-side Models (carry content bytes, used by clone/archive/move/restore)
# ==============================================================================


class ToolResultFile(StrictModel):
    """A tool result file with extension tracking.

    Uses Literal type for extension to fail fast on unknown file types.
    """

    tool_use_id: str
    content: Base64JsonBytes
    extension: ToolResultExtension

    @property
    def filename(self) -> str:
        return f'{self.tool_use_id}{self.extension}'


class ToolResultDirectoryFile(StrictModel):
    """A file within a tool result directory (e.g., page-01.jpg in pdf-<uuid>/)."""

    filename: str
    content: Base64JsonBytes
    extension: ToolResultExtension


class ToolResultDirectory(StrictModel):
    """A tool result directory (e.g., pdf-<uuid>/ containing page images)."""

    name: str
    files: Sequence[ToolResultDirectoryFile]


class ToolResultCollection(StrictModel):
    """All tool result artifacts for a session — flat files and directories."""

    files: Sequence[ToolResultFile] = ()
    directories: Sequence[ToolResultDirectory] = ()

    @property
    def total_file_count(self) -> int:
        """Total number of files (flat + inside directories)."""
        return len(self.files) + sum(len(d.files) for d in self.directories)

    def __bool__(self) -> bool:
        """True if any files or directories exist."""
        return bool(self.files) or bool(self.directories)


# ==============================================================================
# Discovery Types (lightweight, carry paths not content, never serialized)
# ==============================================================================
# NamedTuple chosen over StrictModel because these are ephemeral internal types
# that carry Path objects (not JSON-serializable) and are never persisted.


class DiscoveredFile(NamedTuple):
    """A discovered flat tool result file (path + metadata, no content)."""

    path: Path
    tool_use_id: str  # stem of filename — only meaningful for flat files
    extension: ToolResultExtension


class DiscoveredDirectory(NamedTuple):
    """A discovered tool result directory with its file paths."""

    path: Path
    name: str
    file_paths: Sequence[Path]
    file_extensions: Sequence[ToolResultExtension]  # parallel to file_paths


class DiscoveryResult(NamedTuple):
    """Result of tool result discovery — files, directories, and unknowns."""

    files: Sequence[DiscoveredFile]
    directories: Sequence[DiscoveredDirectory]
    unknown_files: Sequence[Path]  # callers decide: raise or collect


# ==============================================================================
# Functions
# ==============================================================================


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


def discover_tool_results(
    project_folder: Path,
    session_id: str,
) -> DiscoveryResult:
    """Discover and validate all tool result artifacts without reading content.

    Scans the tool-results/ directory one level deep. For flat files, validates
    extensions. For subdirectories, recurses one level and validates file
    extensions inside. Logs a warning on unrecognized directory prefixes.

    Does NOT raise on unknown extensions — returns them in the result so callers
    can decide how to handle (raise for clone/archive/move, collect for delete).

    Returns empty result if the tool-results/ directory does not exist.
    """
    tool_results_dir = get_tool_results_dir(project_folder, session_id)

    if not tool_results_dir.exists():
        return DiscoveryResult(files=(), directories=(), unknown_files=())

    files: list[DiscoveredFile] = []
    directories: list[DiscoveredDirectory] = []
    unknown_files: list[Path] = []

    for path in sorted(tool_results_dir.iterdir()):
        if path.name.startswith('.'):
            continue

        if path.is_file():
            if path.suffix in TOOL_RESULT_EXTENSIONS:
                files.append(
                    DiscoveredFile(
                        path=path,
                        tool_use_id=path.stem,
                        extension=cast(ToolResultExtension, path.suffix),
                    )
                )
            else:
                unknown_files.append(path)

        elif path.is_dir():
            if not path.name.startswith(KNOWN_DIR_PREFIXES):
                logger.warning('Unrecognized tool-result directory prefix: %s', path.name)

            dir_file_paths: list[Path] = []
            dir_file_extensions: list[ToolResultExtension] = []
            for child in sorted(path.iterdir()):
                if child.name.startswith('.'):
                    continue
                if child.is_dir():
                    unknown_files.append(child)  # nested dirs not supported
                elif child.suffix in TOOL_RESULT_EXTENSIONS:
                    dir_file_paths.append(child)
                    dir_file_extensions.append(cast(ToolResultExtension, child.suffix))
                else:
                    unknown_files.append(child)

            if dir_file_paths:  # skip empty directories
                directories.append(
                    DiscoveredDirectory(
                        path=path,
                        name=path.name,
                        file_paths=dir_file_paths,
                        file_extensions=dir_file_extensions,
                    )
                )

    return DiscoveryResult(files=files, directories=directories, unknown_files=unknown_files)


def _raise_on_unknown(discovery: DiscoveryResult) -> None:
    """Raise FileNotFoundError if any unknown files in discovery result."""
    if discovery.unknown_files:
        file_list = '\n  '.join(str(p) for p in discovery.unknown_files)
        raise FileNotFoundError(
            f'Found {len(discovery.unknown_files)} tool result file(s) with unknown extensions:\n  {file_list}\n\n'
            f'Known extensions: {sorted(TOOL_RESULT_EXTENSIONS)}\n'
            f'Claude Code may have changed. Update TOOL_RESULT_EXTENSIONS to handle new file types.'
        )


def collect_tool_results(
    project_folder: Path,
    session_id: str,
) -> ToolResultCollection:
    """
    Collect tool result files and directories for a session.

    Discovers all artifacts via discover_tool_results(), validates extensions
    (raises on unknowns), then reads file content into typed models.

    If the tool-results directory doesn't exist or is empty, returns
    an empty ToolResultCollection. This is NOT an error condition — many
    sessions don't have tool results stored.

    Args:
        project_folder: Path to the project folder under ~/.claude/projects/
        session_id: Session ID to collect tool results for

    Returns:
        ToolResultCollection with flat files and directories

    Raises:
        FileNotFoundError: If files with unknown extensions exist
    """
    discovery = discover_tool_results(project_folder, session_id)
    _raise_on_unknown(discovery)

    files = [
        ToolResultFile(
            tool_use_id=f.tool_use_id,
            content=f.path.read_bytes(),
            extension=f.extension,
        )
        for f in discovery.files
    ]
    directories = [
        ToolResultDirectory(
            name=d.name,
            files=[
                ToolResultDirectoryFile(
                    filename=p.name,
                    content=p.read_bytes(),
                    extension=ext,
                )
                for p, ext in zip(d.file_paths, d.file_extensions)
            ],
        )
        for d in discovery.directories
    ]
    return ToolResultCollection(files=files, directories=directories)


def write_tool_results(
    collection: ToolResultCollection,
    target_dir: Path,
    new_session_id: str,
    *,
    exist_ok: bool = False,
) -> int:
    """Write tool result files and directories to a new session location.

    Creates the tool-results subdirectory structure and writes each
    tool result file with its original extension. For directories (e.g.,
    pdf-<uuid>/), creates the subdirectory and writes files inside.

    Directory structure created:
        {target_dir}/{new_session_id}/tool-results/{tool_use_id}{extension}
        {target_dir}/{new_session_id}/tool-results/{dir_name}/{filename}

    Args:
        collection: ToolResultCollection with flat files and directories
        target_dir: Target project directory under ~/.claude/projects/
        new_session_id: New session ID for directory structure
        exist_ok: If True, silently overwrite existing files (for rollback).
                  If False (default), raise FileExistsError on collision.

    Returns:
        Number of files written

    Raises:
        FileExistsError: If exist_ok=False and a target file already exists
    """
    if not collection:
        return 0

    tool_results_dir = get_tool_results_dir(target_dir, new_session_id)
    tool_results_dir.mkdir(parents=True, exist_ok=True)

    for tr in collection.files:
        file_path = tool_results_dir / tr.filename
        if not exist_ok and file_path.exists():
            raise FileExistsError(
                f'Tool result file already exists: {file_path}\nThis indicates cloning into an existing session.'
            )
        file_path.write_bytes(tr.content)

    for d in collection.directories:
        subdir = tool_results_dir / d.name
        subdir.mkdir(exist_ok=True)
        for f in d.files:
            file_path = subdir / f.filename
            if not exist_ok and file_path.exists():
                raise FileExistsError(
                    f'Tool result file already exists: {file_path}\nThis indicates cloning into an existing session.'
                )
            file_path.write_bytes(f.content)

    return collection.total_file_count
