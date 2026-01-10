"""
Claude Code Session MCP Server.

Provides tools for archiving and managing Claude Code session files.

Setup:
    claude mcp add --scope user claude-session -- uvx --refresh --from git+https://github.com/chrisguillory/claude-session-mcp mcp-server

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
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

import attrs
from mcp.server.fastmcp import Context, FastMCP

from src.mcp.utils import DualLogger
from src.schemas.operations.archive import ArchiveMetadata
from src.schemas.operations.context import SessionContext
from src.schemas.operations.delete import DeleteResult
from src.schemas.operations.lineage import LineageResult
from src.schemas.operations.restore import RestoreResult
from src.services.archive import SessionArchiveService
from src.services.clone import AmbiguousSessionError, SessionCloneService
from src.services.delete import SessionDeleteService
from src.services.info import CurrentSessionContext, SessionInfoService
from src.services.lineage import LineageService
from src.services.parser import SessionParserService
from src.services.restore import SessionRestoreService
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
    claude_pid: int
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

            # Verify by checking if Claude has .claude/ files open
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
    import time

    # Emit marker (will appear in Claude's debug logs)
    print(session_marker, file=sys.stderr, flush=True)

    # Search debug logs for the marker (with retry for timing)
    debug_dir = pathlib.Path.home() / '.claude' / 'debug'
    if not debug_dir.exists():
        raise RuntimeError(f'Claude debug log directory not found: {debug_dir}')

    # Retry loop - Claude Code may not have flushed the marker to disk yet
    max_retries = 10
    retry_delay = 0.1  # 100ms between retries

    for attempt in range(max_retries):
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
        )

        if result.stdout.strip():
            break  # Found it

        time.sleep(retry_delay)
    else:
        raise RuntimeError(
            f'Could not find session marker in any debug log files in {debug_dir} after {max_retries} attempts'
        )

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
            session_id=session_id,
            temp_dir=temp_path,
            parser_service=parser_service,
            project_path=project_path,  # Real project path from lsof
        )

        # Create immutable state
        state = ServerState(
            session_id=session_id,
            project_path=project_path,
            claude_pid=claude_context.claude_pid,
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
        ctx: Context[Any, Any, Any] | None = None,
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
        if ctx is None:
            raise RuntimeError('Context is required - must be called via FastMCP')
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
        ctx: Context[Any, Any, Any] | None = None,
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
        if ctx is None:
            raise RuntimeError('Context is required - must be called via FastMCP')
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

    @server.tool()
    async def clone_session(
        source_session_id: str | None = None,
        translate_paths: bool = True,
        ctx: Context[Any, Any, Any] | None = None,
    ) -> RestoreResult:
        """
        Clone a session directly without creating an archive file.

        Faster alternative to archive+restore when cloning sessions on the
        same machine. Creates a new session with a fresh UUIDv7 session ID,
        copying all conversation history from the source session.

        Args:
            source_session_id: Session ID to clone (full UUID or prefix).
                              If None, clones the current session.
            translate_paths: Translate paths to current project (default: True)

        Returns:
            RestoreResult with new session ID and clone details

        Examples:
            # Clone the current session (fork yourself)
            result = await clone_session()

            # Clone by full session ID
            result = await clone_session('dbae570a-8b88-43e0-a6da-71d649ec07b0')

            # Clone by prefix (must be unique)
            result = await clone_session('dbae570a')

            # Clone without path translation
            result = await clone_session(
                source_session_id='dbae570a',
                translate_paths=False
            )

        Note:
            After cloning, use `claude --resume {new_session_id}` in CLI
            to continue the conversation with the cloned history.
        """
        if ctx is None:
            raise RuntimeError('Context is required - must be called via FastMCP')
        logger = DualLogger(ctx)

        # Default to current session if not specified
        target_id = source_session_id or state.session_id

        # Create clone service for current project
        clone_service = SessionCloneService(state.project_path)

        try:
            # Clone the session
            result = await clone_service.clone(
                source_session_id=target_id,
                translate_paths=translate_paths,
                logger=logger,
            )

            await logger.info(f'Session cloned with new ID: {result.new_session_id}')
            await logger.info(f'Use: claude --resume {result.new_session_id}')

            return result

        except AmbiguousSessionError as e:
            # Re-raise with more context for MCP
            raise ValueError(str(e)) from e

    @server.tool()
    async def delete_session(
        session_id: str,
        force: bool = False,
        no_backup: bool = False,
        dry_run: bool = False,
        ctx: Context[Any, Any, Any] | None = None,
    ) -> DeleteResult:
        """
        Delete session artifacts with auto-backup.

        By default, only cloned/restored sessions (UUIDv7) can be deleted.
        Native Claude sessions (UUIDv4) require force=True.

        Before deletion, an archive is automatically created at
        ~/.claude-session-mcp/deleted/<session-id>-<timestamp>.json
        unless no_backup=True.

        Use restore --in-place on the backup to undo a delete.

        Args:
            session_id: Session ID to delete
            force: Required to delete native (UUIDv4) sessions
            no_backup: Skip auto-backup before deletion
            dry_run: If True, show what would be deleted without actually deleting

        Returns:
            DeleteResult with deletion details and backup path

        Examples:
            # Delete a cloned session (auto-backup created)
            result = await delete_session('019b5232-1234-7abc-...')
            # Returns: DeleteResult(backup_path='~/.claude-session-mcp/deleted/...', ...)

            # Preview what would be deleted
            result = await delete_session('019b5232-...', dry_run=True)

            # Delete native session (requires force)
            result = await delete_session('a1b2c3d4-...', force=True)

            # Delete without backup
            result = await delete_session('019b5232-...', no_backup=True)

        Note:
            To undo a delete, run:
            claude-session restore --in-place <backup_path>
        """
        if ctx is None:
            raise RuntimeError('Context is required - must be called via FastMCP')
        logger = DualLogger(ctx)

        # Create delete service for current project
        delete_service = SessionDeleteService(state.project_path)

        # Delete the session
        result = await delete_service.delete_session(
            session_id=session_id,
            force=force,
            no_backup=no_backup,
            dry_run=dry_run,
            logger=logger,
        )

        if result.success:
            if dry_run:
                await logger.info(
                    f'Dry run: would delete {result.files_deleted} files ({result.size_freed_bytes:,} bytes)'
                )
            else:
                await logger.info(f'Deleted {result.files_deleted} files ({result.size_freed_bytes:,} bytes)')
                if result.backup_path:
                    await logger.info(f'Backup: {result.backup_path}')
                    await logger.info(f'To undo: claude-session restore --in-place {result.backup_path}')
        else:
            await logger.error(f'Delete failed: {result.error_message}')

        return result

    @server.tool()
    async def session_lineage(
        session_id: str | None = None,
        ctx: Context[Any, Any, Any] | None = None,
    ) -> LineageResult | None:
        """
        Get lineage information for a session.

        Shows parent-child relationships, cross-machine detection, and session
        provenance. Useful for understanding where a session came from.

        Args:
            session_id: Session ID to look up. Accepts full UUID or prefix.
                       If None, uses the current session.

        Returns:
            LineageResult with session provenance, or None if session has no
            lineage (native session, never cloned/restored).

        Examples:
            # Get lineage for current session
            result = await session_lineage()

            # Get lineage for specific session
            result = await session_lineage('019b5232')
        """
        if ctx is None:
            raise RuntimeError('Context is required - must be called via FastMCP')
        logger = DualLogger(ctx)

        target_id = session_id or state.session_id
        await logger.info(f'Looking up lineage for session: {target_id[:12]}...')

        # Query storage via service
        lineage_service = LineageService()
        entry = lineage_service.get_entry(target_id)

        if entry is None:
            await logger.info('No lineage found (native session)')
            return None

        # Compute cross-machine status
        is_cross = lineage_service.is_cross_machine(target_id)

        # MCP handler creates API response model
        result = LineageResult(
            **entry.model_dump(),
            is_cross_machine=is_cross,
        )

        if result.is_cross_machine:
            await logger.info(f'Cross-machine restore: {result.parent_machine_id} -> {result.target_machine_id}')
        elif result.is_cross_machine is False:
            await logger.info('Same-machine operation')
        else:
            await logger.info(f'Method: {result.method} (no machine tracking)')

        return result

    @server.tool()
    async def get_session_info(
        session_id: str | None = None,
        ctx: Context[Any, Any, Any] | None = None,
    ) -> SessionContext:
        """
        Get comprehensive information about a session.

        Returns context about a session including session ID, project path,
        session files, origin (how it was created), state, and characteristics.

        Data sources:
        - Session files (~/.claude/projects/)
        - Claude-workspace tracking (~/.claude-workspace/sessions.json)
        - Lineage tracking (~/.claude-session-mcp/lineage.json)

        For other sessions on this machine (tracked in sessions.json), claude_pid
        and machine_id are available from historical data. temp_dir is only
        available for the current session.

        Args:
            session_id: Session ID (full UUID or prefix) to query.
                       If None, returns info for the current session.

        Returns:
            SessionContext with comprehensive session information.

        Examples:
            # Get current session info
            info = await get_session_info()
            # Returns: SessionContext(session_id='...', source='startup', ...)

            # Get info for another session
            info = await get_session_info('019b5232')
        """
        if ctx is None:
            raise RuntimeError('Context is required - must be called via FastMCP')

        # Build current session context for enrichment
        current_context = CurrentSessionContext(
            session_id=state.session_id,
            project_path=state.project_path,
            claude_pid=state.claude_pid,
            temp_dir=state.temp_dir.name,
        )

        # Determine target session
        target_id = session_id or state.session_id

        # Get session info via service
        info_service = SessionInfoService()
        return await info_service.get_info(target_id, current_context)


# ==============================================================================
# Server Entry Point
# ==============================================================================


def main() -> None:
    """Run the MCP server."""
    server.run()


if __name__ == '__main__':
    main()
