"""Detect Claude Code process and resolve current session."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pydantic

from src.schemas.claude_workspace import SessionDatabase


def find_ancestor_claude_pid() -> int | None:
    """Walk process tree to find an ancestor Claude Code process.

    Returns the PID if found, None if not running inside Claude Code.
    """
    current = os.getppid()

    for _ in range(20):
        result = subprocess.run(
            ['ps', '-p', str(current), '-o', 'ppid=,comm='],
            capture_output=True,
            text=True,
        )
        if not result.stdout.strip():
            break

        parts = result.stdout.strip().split(None, 1)
        ppid = int(parts[0])
        comm = parts[1] if len(parts) > 1 else ''

        if 'claude' in comm.lower():
            return current

        current = ppid

    return None


def resolve_session_id_from_pid(claude_pid: int, *, max_attempts: int) -> str | None:
    """Look up session ID for a Claude PID in sessions.json.

    Returns session ID if found, None if no matching active session.

    Raises:
        RuntimeError: If multiple active sessions match the same PID.
    """
    sessions_file = Path.home() / '.claude-workspace' / 'sessions.json'
    adapter = pydantic.TypeAdapter(SessionDatabase)

    for attempt in range(max_attempts):
        if not sessions_file.exists():
            if attempt < max_attempts - 1:
                time.sleep(0.1)
            continue

        with sessions_file.open() as f:
            db = adapter.validate_python(json.load(f))

        matching = [s for s in db.sessions if s.state == 'active' and s.metadata.claude_pid == claude_pid]

        if len(matching) == 1:
            return matching[0].session_id
        if len(matching) > 1:
            ids = [s.session_id for s in matching]
            raise RuntimeError(f'Multiple active sessions for PID {claude_pid}: {ids}')

        if attempt < max_attempts - 1:
            time.sleep(0.1)

    return None


def auto_detect_session_id() -> str | None:
    """Find Claude ancestor and resolve session ID.

    Returns session ID if running inside Claude Code, None otherwise.
    """
    pid = find_ancestor_claude_pid()
    if pid is None:
        return None
    return resolve_session_id_from_pid(pid, max_attempts=1)
