"""
Session lineage tracking service.

Records parent-child relationships when sessions are cloned or restored.
Stores lineage in ~/.claude-session-mcp/lineage.json with thread-safe access.
"""

from __future__ import annotations

import getpass
import json
import socket
from datetime import datetime
from pathlib import Path
from typing import Literal

from filelock import FileLock
from pydantic import Field

from ..base_model import StrictModel
from ..types import JsonDatetime

__all__ = [
    'LineageEntry',
    'LineageFile',
    'LineageResult',
    'LineageService',
    'get_machine_id',
]


def get_machine_id() -> str:
    """Get current machine identifier as 'user@hostname'.

    This is human-readable, cross-platform, and captures both
    user context and machine identity.
    """
    return f'{getpass.getuser()}@{socket.gethostname()}'


class LineageEntry(StrictModel):
    """Lineage entry for a cloned/restored session.

    Self-contained model with consistent 'parent_*' prefixing for source references.

    Field ordering:
    - Identity (who)
    - Temporal (when)
    - Operation (how)
    - Parent context (from where)
    - Target context (to where)
    - Operational flags (details)
    """

    # Identity
    child_session_id: str  # Session ID created by clone/restore
    parent_session_id: str  # Source session that was cloned

    # Temporal
    cloned_at: JsonDatetime  # Timestamp of operation (UTC)

    # Operation
    method: Literal['clone', 'restore']  # How the session was created

    # Parent context (source)
    parent_project_path: str  # Project path where parent session exists
    parent_machine_id: str | None  # Machine where parent was created (from archive, None for clone)

    # Target context (destination)
    target_project_path: str  # Project path where clone was created
    target_machine_id: str  # Machine where clone/restore operation executed

    # Operational flags
    paths_translated: bool  # Whether project paths were translated
    archive_path: str | None  # Archive path (restore operations only)


class LineageResult(StrictModel):
    """API response for session lineage query.

    Separate from LineageEntry (storage model) to allow independent evolution
    of the API contract and storage format. Contains computed fields that are
    not persisted to storage.

    Field ordering matches LineageEntry plus computed fields at the end.
    """

    # Identity
    child_session_id: str
    parent_session_id: str

    # Temporal
    cloned_at: JsonDatetime

    # Operation
    method: Literal['clone', 'restore']

    # Parent context (source)
    parent_project_path: str
    parent_machine_id: str | None

    # Target context (destination)
    target_project_path: str
    target_machine_id: str

    # Operational flags
    paths_translated: bool
    archive_path: str | None

    # Computed (not in storage)
    is_cross_machine: bool | None  # True=cross-machine, False=same-machine, None=unknown (clones)


class LineageFile(StrictModel):
    """The lineage.json file structure.

    This model is NOT frozen to allow mutable sessions dict.
    """

    model_config = {'extra': 'forbid', 'strict': True, 'frozen': False}

    schema_version: str = '1.0'
    sessions: dict[str, LineageEntry] = Field(default_factory=dict)


class LineageService:
    """Service for tracking session clone lineage.

    Uses filelock for cross-process safety and atomic writes.
    """

    def __init__(self, lineage_dir: Path | None = None) -> None:
        """Initialize with ~/.claude-session-mcp/ by default."""
        self.lineage_dir = lineage_dir or (Path.home() / '.claude-session-mcp')
        self.lineage_file = self.lineage_dir / 'lineage.json'
        self.lock_file = self.lineage_file.with_suffix('.lock')

    def record_clone(
        self,
        child_session_id: str,
        parent_session_id: str,
        cloned_at: datetime,
        parent_project_path: Path,
        target_project_path: Path,
        method: Literal['clone', 'restore'],
        parent_machine_id: str | None = None,
        paths_translated: bool = False,
        archive_path: str | None = None,
    ) -> None:
        """Record a clone/restore operation.

        target_machine_id is computed automatically via get_machine_id().

        Args:
            child_session_id: The new session ID created by clone/restore
            parent_session_id: The source session ID
            cloned_at: Timestamp of the operation (UTC)
            parent_project_path: Project path of the parent session
            target_project_path: Project path where clone was created
            method: Either 'clone' or 'restore'
            parent_machine_id: Machine ID from archive (restore only, None for clone)
            paths_translated: Whether path translation was applied
            archive_path: Path to archive file (restore only)
        """
        entry = LineageEntry(
            child_session_id=child_session_id,
            parent_session_id=parent_session_id,
            cloned_at=cloned_at,
            method=method,
            parent_project_path=str(parent_project_path),
            parent_machine_id=parent_machine_id,
            target_project_path=str(target_project_path),
            target_machine_id=get_machine_id(),
            paths_translated=paths_translated,
            archive_path=archive_path,
        )

        # Ensure directory exists
        self.lineage_dir.mkdir(parents=True, exist_ok=True)

        # Acquire lock, read, modify, write atomically
        with FileLock(self.lock_file):
            lineage = self._read_lineage_file()
            # Update sessions dict (LineageFile is not frozen)
            lineage.sessions[child_session_id] = entry
            self._write_lineage_file(lineage)

    def get_entry(self, session_id: str) -> LineageEntry | None:
        """Get lineage entry for a session.

        Args:
            session_id: Full or prefix of session ID

        Returns:
            LineageEntry if found, None otherwise
        """
        if not self.lineage_file.exists():
            return None

        lineage = self._read_lineage_file()

        # Try exact match first
        if session_id in lineage.sessions:
            return lineage.sessions[session_id]

        # Try prefix match
        matches = [sid for sid in lineage.sessions if sid.startswith(session_id)]
        if len(matches) == 1:
            return lineage.sessions[matches[0]]

        return None

    def get_parent(self, session_id: str) -> str | None:
        """Get parent session ID.

        Returns:
            Parent session ID if found, None otherwise
        """
        entry = self.get_entry(session_id)
        return entry.parent_session_id if entry else None

    def get_children(self, parent_id: str) -> list[str]:
        """Get all sessions cloned from this parent.

        Args:
            parent_id: Full or prefix of parent session ID

        Returns:
            List of child session IDs
        """
        if not self.lineage_file.exists():
            return []

        lineage = self._read_lineage_file()
        children = []

        for session_id, entry in lineage.sessions.items():
            # Check exact match or prefix match
            if entry.parent_session_id == parent_id or entry.parent_session_id.startswith(parent_id):
                children.append(session_id)

        return children

    def get_ancestry(self, session_id: str, max_depth: int = 10) -> list[str]:
        """Get full ancestry chain [root, ..., parent, self].

        Follows parent links recursively until:
        - Reaching a session with no parent (root)
        - Reaching max_depth (cycle detection)

        Args:
            session_id: Starting session ID
            max_depth: Maximum depth to prevent infinite loops

        Returns:
            List of session IDs from root to self
        """
        ancestry: list[str] = []
        current = session_id
        depth = 0

        while current and depth < max_depth:
            ancestry.append(current)
            entry = self.get_entry(current)
            if entry is None:
                break
            current = entry.parent_session_id
            depth += 1

        # Reverse to get root-first order
        ancestry.reverse()
        return ancestry

    def is_cross_machine(self, session_id: str) -> bool | None:
        """Check if session was restored from a different machine.

        Returns:
            True - Restore from different machine
            False - Clone or restore on same machine
            None - Session not found or parent_machine_id not available
        """
        entry = self.get_entry(session_id)
        if not entry or entry.parent_machine_id is None:
            return None
        return entry.target_machine_id != entry.parent_machine_id

    def _read_lineage_file(self) -> LineageFile:
        """Read and parse lineage.json (creates empty if missing)."""
        if not self.lineage_file.exists():
            return LineageFile()

        with self.lineage_file.open() as f:
            data = json.load(f)
        return LineageFile.model_validate(data)

    def _write_lineage_file(self, lineage: LineageFile) -> None:
        """Write lineage.json atomically using temp file + rename."""
        tmp_file = self.lineage_file.with_suffix('.tmp.json')

        # Write to temp file
        with tmp_file.open('w') as f:
            # Convert to dict for serialization
            data = lineage.model_dump(mode='json')
            json.dump(data, f, indent=2, default=str)

        # Atomic rename
        tmp_file.rename(self.lineage_file)
