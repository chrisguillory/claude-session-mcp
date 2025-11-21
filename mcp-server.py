#!/usr/bin/env -S uv run
"""
Claude Code Session MCP Server.

Provides tools for archiving and managing Claude Code session files.

Setup:
    claude mcp add --transport stdio claude-session -- uv run "$REPO_ROOT/mcp-server.py"

Example:
    # Save current session to temp file
    save_current_session()

    # Save with compression to specific path
    save_current_session(output_path='/path/to/archive.json.zst', format='zst')
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import AsyncIterator, Literal

import attrs
from mcp.server.fastmcp import Context, FastMCP

from mcp_utils import DualLogger
from src.services.archive import ArchiveMetadata, SessionArchiveService
from src.services.parser import SessionParserService
from src.services.restore import RestoreResult, SessionRestoreService
from src.storage.local import LocalFileSystemStorage


# ==============================================================================
# Server State (immutable)
# ==============================================================================


@attrs.define(frozen=True)
class ServerState:
    """
    Immutable server state initialized at startup.

    Contains all services and configuration needed for tool execution.
    """

    session_id: str
    project_path: Path
    temp_dir: tempfile.TemporaryDirectory[str]
    parser_service: SessionParserService
    archive_service: SessionArchiveService


# ==============================================================================
# Session Discovery (from python-interpreter-mcp pattern)
# ==============================================================================


@attrs.define(frozen=True)
class ClaudeContext:
    """Context information about the Claude Code session."""

    claude_pid: int
    project_dir: pathlib.Path


def _find_claude_context() -> ClaudeContext:
    """
    Find Claude process and extract its context (PID, project directory).

    Walks up the process tree to find the Claude process, then uses lsof to
    determine its working directory.

    Returns:
        ClaudeContext with PID and project directory

    Raises:
        RuntimeError: If Claude process cannot be found or CWD cannot be determined
    """
    current = os.getppid()

    for _ in range(20):  # Depth limit
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

        # Check if this is Claude
        if 'claude' in comm.lower():
            # Get Claude's CWD using lsof
            result = subprocess.run(
                ['lsof', '-p', str(current), '-a', '-d', 'cwd'],
                capture_output=True,
                text=True,
            )

            cwd = None
            for line in result.stdout.split('\n'):
                if 'cwd' in line:
                    parts = line.split()
                    if len(parts) >= 9:
                        cwd = pathlib.Path(' '.join(parts[8:]))
                        break

            if not cwd:
                raise RuntimeError(f'Found Claude process (PID {current}) but could not determine CWD')

            # Verify by checking if Claude has .claude/ files open that match the CWD
            result = subprocess.run(['lsof', '-p', str(current)], capture_output=True, text=True)

            claude_files = []
            for line in result.stdout.split('\n'):
                if '.claude' in line:
                    # Extract the full path from lsof output
                    parts = line.split()
                    if len(parts) >= 9:
                        file_path = ' '.join(parts[8:])
                        claude_files.append(file_path)

            if not claude_files:
                raise RuntimeError(
                    f'Found Claude process (PID {current}) with CWD {cwd}, '
                    f'but no .claude/ files are open - may not be a Claude project'
                )

            # Verify at least one .claude file is parented to the CWD
            cwd_str = str(cwd)
            matching_files = [f for f in claude_files if f.startswith(cwd_str)]

            if not matching_files:
                raise RuntimeError(
                    f'Found Claude process (PID {current}) with CWD {cwd}, '
                    f'but .claude/ files open are not parented to this directory:\n'
                    f'  Open files: {claude_files}\n'
                    f'  Expected parent: {cwd_str}'
                )

            return ClaudeContext(claude_pid=current, project_dir=cwd)

        current = ppid

    raise RuntimeError('Could not find Claude process in parent tree')


def _discover_session_id(session_marker: str) -> str:
    """
    Discover Claude Code session ID via self-referential marker search.

    Workaround for lack of official session discovery API. Writes a unique marker
    to stdout, then searches Claude's debug logs for it to find the session ID.

    Args:
        session_marker: Unique marker string to search for

    Returns:
        Session ID (UUID string)

    Raises:
        RuntimeError: If debug directory doesn't exist or session ID cannot be found
    """
    # Emit marker (will appear in Claude's debug logs)
    print(session_marker, file=sys.stderr, flush=True)

    # Search debug logs for the marker
    debug_dir = pathlib.Path.home() / '.claude' / 'debug'
    if not debug_dir.exists():
        raise RuntimeError(f'Claude debug log directory not found: {debug_dir}')

    result = subprocess.run(
        [
            'rg',
            '-l',
            '--fixed-strings',
            '--glob',
            '!latest',
            session_marker,
            debug_dir.as_posix(),
        ],
        capture_output=True,
        text=True,
        timeout=2,
    )

    if not result.stdout.strip():
        raise RuntimeError(f'Could not find session marker in any debug log files in {debug_dir}')

    # rg found the marker - extract session ID from first matching file
    # Format: ~/.claude/debug/{session_id}.log
    log_file = pathlib.Path(result.stdout.strip().split('\n')[0])
    session_id = log_file.stem

    return session_id


# ==============================================================================
# Lifespan Management
# ==============================================================================


@contextlib.asynccontextmanager
async def lifespan(mcp_server: FastMCP) -> AsyncIterator[None]:
    """
    Manage server lifecycle and state initialization.

    Creates ServerState with all services at startup and cleans up on shutdown.
    """
    # Discover Claude context (PID and project directory)
    claude_context = _find_claude_context()
    project_path = claude_context.project_dir

    # Discover session ID via marker search
    session_marker = f'SESSION_MARKER_{uuid.uuid4()}'
    session_id = _discover_session_id(session_marker)

    # Create temp directory (cleaned up on exit)
    temp_dir = tempfile.TemporaryDirectory(prefix='claude-session-')
    temp_path = Path(temp_dir.name)

    try:
        # Initialize services
        parser_service = SessionParserService()
        archive_service = SessionArchiveService(
            session_id=session_id, project_path=project_path, temp_dir=temp_path, parser_service=parser_service
        )

        # Create immutable state
        state = ServerState(
            session_id=session_id,
            project_path=project_path,
            temp_dir=temp_dir,
            parser_service=parser_service,
            archive_service=archive_service,
        )

        # Register tools with closure over state
        register_tools(state)

        print(f'[MCP Server] Session ID: {session_id}')
        print(f'[MCP Server] Project: {project_path}')
        print(f'[MCP Server] Temp dir: {temp_path}')

        yield  # Setup successful; application active

    finally:
        # Cleanup temp directory
        temp_dir.cleanup()
        print('[MCP Server] Cleaned up temp directory')


# ==============================================================================
# Server Setup
# ==============================================================================

server = FastMCP('claude-session', lifespan=lifespan)


# ==============================================================================
# Tool Registration (Closure Pattern)
# ==============================================================================


def register_tools(state: ServerState) -> None:
    """
    Register MCP tools with closure over server state.

    Args:
        state: Server state containing services
    """

    @server.tool()
    async def save_current_session(
        output_path: str | None = None,
        format: Literal['json', 'zst'] | None = None,
        storage_backend: Literal['local'] = 'local',
        ctx: Context = None,
    ) -> ArchiveMetadata:
        """
        Save current Claude Code session to archive.

        Creates a complete archive of the current session including all agent
        sessions. By default, saves to a temporary file that is cleaned up on
        server shutdown.

        Args:
            output_path: Optional custom output path (default: temp file)
            format: Optional format override ('json' or 'zst')
            storage_backend: Storage backend to use (default: 'local')

        Returns:
            Archive metadata with file path, size, and record counts

        Examples:
            # Save to temp file (default)
            result = await save_current_session()
            # Returns: ArchiveMetadata(file_path='/tmp/.../session-abc123.json', ...)

            # Save to specific path with compression
            result = await save_current_session(
                output_path='/path/to/archive.json.zst',
                format='zst'
            )
        """
        logger = DualLogger(ctx)

        # Create storage backend
        if storage_backend == 'local':
            if output_path:
                # User-specified path - use parent directory
                storage = LocalFileSystemStorage(Path(output_path).parent)
            else:
                # Temp file - use temp directory
                storage = LocalFileSystemStorage(Path(state.temp_dir.name))
        else:
            raise ValueError(f'Unsupported storage backend: {storage_backend}')

        # Create archive
        metadata = await state.archive_service.create_archive(
            storage=storage, output_path=output_path, format_param=format, logger=logger
        )

        return metadata

    @server.tool()
    async def restore_session(
        archive_path: str,
        translate_paths: bool = True,
        ctx: Context = None,
    ) -> RestoreResult:
        """
        Restore a saved session archive with a new session ID.

        Works like Claude's teleport feature - creates a new session ID
        while preserving the full conversation history. Loads the session
        into ~/.claude/projects/ allowing you to continue the conversation.

        Args:
            archive_path: Path to session archive file (JSON or .zst)
            translate_paths: Translate paths to current project (default: True)

        Returns:
            RestoreResult with new session ID and restoration details

        Examples:
            # Restore from Downloads
            result = await restore_session('/Users/chris/Downloads/claude-session.json')
            # Returns: RestoreResult(new_session_id='abc123...', ...)

            # Restore without path translation
            result = await restore_session(
                archive_path='/path/to/archive.json',
                translate_paths=False
            )

        Note:
            After restoration, use `claude --resume {new_session_id}` in CLI
            to continue the conversation with the restored history.
        """
        logger = DualLogger(ctx)

        # Create restore service for current project
        restore_service = SessionRestoreService(state.project_path)

        # Restore the archive
        result = await restore_service.restore_archive(
            archive_path=archive_path,
            translate_paths=translate_paths,
            logger=logger,
        )

        await logger.info(f'Session restored with new ID: {result.new_session_id}')
        await logger.info(f'Use: claude --resume {result.new_session_id}')

        return result


# ==============================================================================
# Server Entry Point
# ==============================================================================


def main() -> None:
    """Run the MCP server."""
    server.run()


if __name__ == '__main__':
    main()
