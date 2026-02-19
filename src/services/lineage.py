"""
Session lineage tracking service.

Records parent-child relationships when sessions are cloned or restored.
Stores lineage in ~/.claude-session-mcp/lineage.json with thread-safe access.
"""

from __future__ import annotations

import getpass
import json
import mmap
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Literal

from filelock import FileLock

from src.exceptions import AmbiguousSessionError
from src.schemas.operations.lineage import (
    LineageEntry,
    LineageFile,
    LineageTree,
    LineageTreeNode,
)

__all__ = [
    'LineageEntry',
    'LineageFile',
    'LineageService',
    'get_machine_id',
]


def get_machine_id() -> str:
    """Get current machine identifier as 'user@hostname'.

    This is human-readable, cross-platform, and captures both
    user context and machine identity.
    """
    return f'{getpass.getuser()}@{socket.gethostname()}'


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

        Raises:
            AmbiguousSessionError: If prefix matches multiple sessions
        """
        if not self.lineage_file.exists():
            return None

        lineage = self._read_lineage_file()

        # Try exact match first
        if session_id in lineage.sessions:
            return lineage.sessions[session_id]

        # Try prefix match
        matches = [sid for sid in lineage.sessions if sid.startswith(session_id)]
        if len(matches) > 1:
            raise AmbiguousSessionError(session_id, matches)
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

    def get_full_tree(self, session_id: str) -> LineageTree | None:
        """Build the complete lineage tree containing session_id.

        Walks UP to find the root ancestor, then DOWN to collect all
        descendants. Returns None if session_id is not part of any lineage.

        Args:
            session_id: Any session ID in the tree (full or prefix).

        Returns:
            LineageTree with all nodes, or None if not part of any lineage.
        """
        lineage = self._read_lineage_file() if self.lineage_file.exists() else LineageFile()

        # Resolve session_id (may be a prefix, may be a root that only appears as parent)
        resolved = self._resolve_in_lineage(session_id, lineage)
        if resolved is None:
            return None

        # Walk up to root
        root = resolved
        visited_up: set[str] = {root}
        while root in lineage.sessions:
            parent = lineage.sessions[root].parent_session_id
            if parent in visited_up:
                break  # Cycle
            visited_up.add(parent)
            root = parent

        # Check: if root has no entry AND no children referencing it, it's not in any lineage
        children_index: dict[str, list[str]] = {}
        for child_id, entry in lineage.sessions.items():
            children_index.setdefault(entry.parent_session_id, []).append(child_id)

        if root not in lineage.sessions and root not in children_index:
            return None

        # Sort children deterministically: by cloned_at, then session ID
        for parent_id in children_index:
            children_index[parent_id].sort(key=lambda cid: (lineage.sessions[cid].cloned_at.isoformat(), cid))

        # Build session file index for title resolution
        session_files = self._find_session_files(
            [root, *children_index.get(root, [])] + [cid for kids in children_index.values() for cid in kids]
        )

        # BFS from root to build all nodes
        nodes: dict[str, LineageTreeNode] = {}
        queue: list[tuple[str, str | None, int]] = [(root, None, 0)]
        visited: set[str] = set()

        while queue:
            current_id, current_parent_id, current_depth = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            node_entry = lineage.sessions.get(current_id)
            children = children_index.get(current_id, [])

            # Compute cross-machine status
            is_cross: bool | None = None
            if node_entry and node_entry.parent_machine_id is not None:
                is_cross = node_entry.target_machine_id != node_entry.parent_machine_id

            # Resolve custom title from session file
            session_file = session_files.get(current_id)
            custom_title = self._extract_title(session_file) if session_file else None

            nodes[current_id] = LineageTreeNode(
                session_id=current_id,
                parent_id=current_parent_id,
                children=tuple(children),
                depth=current_depth,
                cloned_at=node_entry.cloned_at if node_entry else None,
                method=node_entry.method if node_entry else None,
                parent_project_path=node_entry.parent_project_path if node_entry else None,
                parent_machine_id=node_entry.parent_machine_id if node_entry else None,
                target_project_path=node_entry.target_project_path if node_entry else None,
                target_machine_id=node_entry.target_machine_id if node_entry else None,
                paths_translated=node_entry.paths_translated if node_entry else None,
                archive_path=node_entry.archive_path if node_entry else None,
                is_cross_machine=is_cross,
                custom_title=custom_title,
            )

            queue.extend((child_id, current_id, current_depth + 1) for child_id in children)

        return LineageTree(
            root_session_id=root,
            queried_session_id=resolved,
            nodes=nodes,
        )

    def _resolve_in_lineage(self, session_id: str, lineage: LineageFile) -> str | None:
        """Resolve a session ID prefix within the lineage file.

        Checks both child_session_id keys and parent_session_id values.

        Returns:
            Full session ID, or None if not found in lineage at all.

        Raises:
            AmbiguousSessionError: If prefix matches multiple sessions.
        """
        # Exact match as child
        if session_id in lineage.sessions:
            return session_id

        # Prefix match as child
        child_matches = [sid for sid in lineage.sessions if sid.startswith(session_id)]
        if len(child_matches) > 1:
            raise AmbiguousSessionError(session_id, child_matches)
        if len(child_matches) == 1:
            return child_matches[0]

        # Exact match as parent
        all_parent_ids = {entry.parent_session_id for entry in lineage.sessions.values()}
        if session_id in all_parent_ids:
            return session_id

        # Prefix match as parent
        parent_matches = [pid for pid in all_parent_ids if pid.startswith(session_id)]
        if len(parent_matches) > 1:
            raise AmbiguousSessionError(session_id, parent_matches)
        if len(parent_matches) == 1:
            return parent_matches[0]

        return None

    @staticmethod
    def _find_session_files(session_ids: list[str]) -> dict[str, Path]:
        """Find JSONL session files for the given session IDs.

        Uses a single rg call to find all files efficiently.

        Returns:
            Mapping of session_id -> Path for files found locally.
        """
        claude_projects = Path.home() / '.claude' / 'projects'
        if not claude_projects.exists():
            return {}

        # Single rg call with glob alternatives: {id1,id2,...}.jsonl
        glob_pattern = '{' + ','.join(session_ids) + '}.jsonl'
        result = subprocess.run(
            ['rg', '--files', '--glob', glob_pattern, str(claude_projects)],
            capture_output=True,
            text=True,
        )

        if not result.stdout.strip():
            return {}

        files: dict[str, Path] = {}
        for line in result.stdout.strip().split('\n'):
            path = Path(line)
            files[path.stem] = path
        return files

    @staticmethod
    def _extract_title(session_file: Path) -> str | None:
        """Extract custom title from a session file using mmap rfind.

        Searches backwards for the last custom-title record without parsing
        the entire JSONL. Only parses the single matching line.
        """
        try:
            with session_file.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                pos = mm.rfind(b'"custom-title"')
                if pos == -1:
                    return None
                line_start = mm.rfind(b'\n', 0, pos) + 1
                line_end = mm.find(b'\n', pos)
                if line_end == -1:
                    line_end = len(mm)
                return json.loads(mm[line_start:line_end]).get('customTitle')  # type: ignore[no-any-return]
        except (OSError, json.JSONDecodeError, ValueError):
            return None

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
