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
import json
import os
import pathlib
import subprocess
import tempfile
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

import attrs
import pydantic
from mcp.server.fastmcp import Context, FastMCP

from src.exceptions import RunningSessionDeletionError
from src.mcp.utils import DualLogger
from src.schemas.claude_workspace import SessionDatabase
from src.schemas.operations.archive import ArchiveMetadata
from src.schemas.operations.context import SessionContext
from src.schemas.operations.delete import DeleteResult
from src.schemas.operations.gist import GistArchiveResult
from src.schemas.operations.lineage import LineageResult
from src.schemas.operations.restore import RestoreResult
from src.services.archive import SessionArchiveService
from src.services.clone import SessionCloneService
from src.services.delete import SessionDeleteService
from src.services.info import CurrentSessionContext, SessionInfoService
from src.services.lineage import LineageService
from src.services.parser import SessionParserService
from src.services.restore import SessionRestoreService
from src.storage.gist import GistStorage
from src.storage.local import LocalFileSystemStorage

# ==============================================================================
# Types
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


@attrs.define(frozen=True)
class ClaudeContext:
    """Context information about the Claude Code session."""

    claude_pid: int
    project_dir: pathlib.Path


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

    # Discover session ID from claude-workspace sessions.json
    session_id = _discover_session_id(claude_context.claude_pid)

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
            claude_pid=claude_context.claude_pid,  # For process-based version detection
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

        # Clone the session
        result = await clone_service.clone(
            source_session_id=target_id,
            translate_paths=translate_paths,
            logger=logger,
        )

        await logger.info(f'Session cloned with new ID: {result.new_session_id}')
        await logger.info(f'Use: claude --resume {result.new_session_id}')

        return result

    @server.tool()
    async def delete_session(
        session_id: str,
        force: bool = False,
        terminate_running: bool = False,
        no_backup: bool = False,
        dry_run: bool = False,
        ctx: Context[Any, Any, Any] | None = None,
    ) -> DeleteResult:
        """
        Delete session artifacts with auto-backup.

        By default, only cloned/restored sessions (UUIDv7) can be deleted.
        Native Claude sessions (UUIDv4) require force=True.

        If the session is currently running (another Claude process), set
        terminate_running=True to kill it before deletion. For self-deletion
        (deleting the current session), termination is automatic.

        Deletion is atomic with rollback on failure. A backup is always
        created for rollback capability. The no_backup flag only controls
        whether the backup is kept after successful deletion (for undo).

        Use restore --in-place on the backup to undo a delete.

        Args:
            session_id: Session ID to delete
            force: Required to delete native (UUIDv4) sessions
            terminate_running: Terminate running Claude process before deletion
            no_backup: Don't keep a backup file for undo
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

            # Delete a running session (requires terminate_running)
            result = await delete_session('019b5232-...', terminate_running=True)

            # Delete without backup
            result = await delete_session('019b5232-...', no_backup=True)

        Note:
            To undo a delete, run:
            claude-session restore --in-place <backup_path>
        """
        if ctx is None:
            raise RuntimeError('Context is required - must be called via FastMCP')
        logger = DualLogger(ctx)

        # Resolve prefix to full session ID
        info_service = SessionInfoService()
        session_info = await info_service.resolve_session(session_id)
        full_session_id = session_info.session_id

        # Create delete service for current project
        delete_service = SessionDeleteService(state.project_path)

        # Determine if termination is needed
        is_self_delete = full_session_id == state.session_id

        if is_self_delete:
            # Self-deletion: auto-terminate (no flag needed)
            terminate_pid = state.claude_pid if not dry_run else None
        else:
            # Other session: check if running
            is_running, running_pid = info_service.is_session_running(full_session_id)
            if is_running:
                if dry_run:
                    await logger.info(f'Warning: Session is currently running (PID {running_pid})')
                    terminate_pid = None
                elif not terminate_running:
                    # Running without terminate_running: raise exception
                    assert running_pid is not None  # is_running=True guarantees this
                    raise RunningSessionDeletionError(full_session_id, running_pid)
                else:
                    # Running with terminate_running: will terminate
                    await logger.info(f'Session is running (PID {running_pid}), will terminate before deletion')
                    terminate_pid = running_pid
            else:
                terminate_pid = None

        # Delete the session
        result = await delete_service.delete_session(
            session_id=full_session_id,
            force=force,
            no_backup=no_backup,
            dry_run=dry_run,
            logger=logger,
            terminate_pid_before_delete=terminate_pid,
        )

        if result.success:
            if dry_run:
                await logger.info(
                    f'Dry run: would delete {result.files_deleted} files, '
                    f'{len(result.directories_removed)} directories ({result.size_freed_bytes:,} bytes)'
                )
            else:
                await logger.info(
                    f'Deleted {result.files_deleted} files, '
                    f'{len(result.directories_removed)} directories ({result.size_freed_bytes:,} bytes)'
                )
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

        target_id_or_prefix = session_id or state.session_id

        # Resolve prefix to full session ID and validate session exists
        info_service = SessionInfoService()
        session_info = await info_service.resolve_session(target_id_or_prefix)
        target_id = session_info.session_id

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

    @server.tool()
    async def save_session_to_gist(
        session_id: str | None = None,
        gist_id: str | None = None,
        visibility: Literal['public', 'secret'] = 'secret',
        description: str = 'Claude Code Session Archive',
        ctx: Context[Any, Any, Any] | None = None,
    ) -> GistArchiveResult:
        """
        Save a session to GitHub Gist.

        Creates a new gist or updates an existing one with the session archive.
        Requires GitHub authentication via GITHUB_TOKEN env var or gh CLI.

        Args:
            session_id: Session to save (full UUID or prefix). Defaults to current session.
            gist_id: Existing gist ID to update. If None, creates a new gist.
            visibility: 'public' or 'secret' (default: secret)
            description: Gist description

        Returns:
            GistArchiveResult with gist URL, ID, and archive metadata.

        Authentication:
            Token is obtained from (in order):
            1. GITHUB_TOKEN environment variable
            2. `gh auth token` (GitHub CLI)

            If neither is available, an error is raised with setup instructions.

        Examples:
            # Save current session to new secret gist
            result = await save_session_to_gist()

            # Save specific session to public gist
            result = await save_session_to_gist(
                session_id='019b5232',
                visibility='public'
            )

            # Update an existing gist
            result = await save_session_to_gist(gist_id='abc123xyz')
        """
        if ctx is None:
            raise RuntimeError('Context is required - must be called via FastMCP')
        logger = DualLogger(ctx)

        # Get GitHub token
        token = _get_github_token()
        if not token:
            raise ValueError(
                'GitHub authentication required.\n\n'
                'Set up authentication using one of:\n'
                '1. Set GITHUB_TOKEN environment variable\n'
                '2. Install and authenticate GitHub CLI: gh auth login\n\n'
                'To create a token manually:\n'
                '  1. Go to https://github.com/settings/tokens\n'
                '  2. Generate new token (classic)\n'
                "  3. Select 'gist' scope\n"
                '  4. Set as GITHUB_TOKEN in your environment'
            )

        # Determine target session (resolve prefix if needed)
        target_id_or_prefix = session_id or state.session_id

        # Resolve prefix to full session ID
        info_service = SessionInfoService()
        session_info = await info_service.resolve_session(target_id_or_prefix)
        target_id = session_info.session_id

        await logger.info(f'Saving session to Gist: {target_id[:12]}...')

        # Determine correct PID for version detection:
        # - If archiving current session: use live PID from state
        # - If archiving other session: get historical PID from sessions.json (if available)
        is_current_session = target_id == state.session_id
        if is_current_session:
            archive_claude_pid: int | None = state.claude_pid
        else:
            # Get target session's PID from workspace sessions.json
            workspace_session = info_service._load_workspace_session(target_id)
            archive_claude_pid = workspace_session.metadata.claude_pid if workspace_session else None

        # Create Gist storage backend
        storage = GistStorage(
            token=token,
            gist_id=gist_id,
            visibility=visibility,
            description=description,
        )

        archive_service = SessionArchiveService(
            session_id=target_id,
            temp_dir=Path(state.temp_dir.name),
            parser_service=state.parser_service,
            session_folder=session_info.session_folder,
            claude_pid=archive_claude_pid,  # Target session's PID (not current session's PID)
        )

        # Create archive (zst for better compression - auto base64-encoded for Gist)
        metadata = await archive_service.create_archive(
            storage=storage,
            output_path=None,  # Let storage generate path
            format_param='zst',  # Zstd compression (6x smaller than JSON)
            logger=logger,
        )

        # Get gist ID from storage (set after creation)
        final_gist_id = storage.gist_id or ''

        await logger.info(f'Session uploaded to Gist: {metadata.file_path}')
        await logger.info(f'Gist ID: {final_gist_id}')
        await logger.info(f'To restore: claude-session restore gist://{final_gist_id}')

        return GistArchiveResult(
            gist_url=metadata.file_path,  # GistStorage returns html_url as file_path
            gist_id=final_gist_id,
            session_id=target_id,
            format=metadata.format,
            size_mb=metadata.size_mb,
            session_records=metadata.session_records,
            agent_records=metadata.agent_records,
            file_count=metadata.file_count,
            restore_command=f'claude-session restore gist://{final_gist_id}',
        )


# ==============================================================================
# Private Helpers
# ==============================================================================


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


def _discover_session_id(claude_pid: int) -> str:
    """
    Discover Claude Code session ID from claude-workspace sessions.json.

    Looks up the active session matching the given Claude PID in the sessions
    database maintained by claude-workspace hooks.

    Args:
        claude_pid: PID of the Claude process (from _find_claude_context)

    Returns:
        Session ID (UUID string)

    Raises:
        RuntimeError: If sessions.json doesn't exist, no matching session found,
                      or multiple active sessions match the PID
    """
    sessions_file = pathlib.Path.home() / '.claude-workspace' / 'sessions.json'

    # Retry loop - SessionStart hook may not have finished writing yet
    max_retries = 10
    retry_delay = 0.1  # 100ms between retries

    for attempt in range(max_retries):
        if not sessions_file.exists():
            time.sleep(retry_delay)
            continue

        with sessions_file.open() as f:
            data = json.load(f)

        adapter = pydantic.TypeAdapter(SessionDatabase)
        db = adapter.validate_python(data)

        # Find active sessions matching our Claude PID
        matching = [s for s in db.sessions if s.state == 'active' and s.metadata.claude_pid == claude_pid]

        if len(matching) == 1:
            return matching[0].session_id

        if len(matching) > 1:
            # Multiple active sessions with same PID - shouldn't happen
            session_ids = [s.session_id for s in matching]
            raise RuntimeError(f'Multiple active sessions found for Claude PID {claude_pid}: {session_ids}')

        # No match yet - hook may still be writing
        time.sleep(retry_delay)

    # All retries exhausted
    raise RuntimeError(
        f'Could not find active session for Claude PID {claude_pid} in {sessions_file} '
        f'after {max_retries} attempts. Ensure claude-workspace SessionStart hook is configured.'
    )


def _get_github_token() -> str | None:
    """
    Get GitHub token from environment or gh CLI.

    Checks in order:
    1. GITHUB_TOKEN environment variable
    2. `gh auth token` command (GitHub CLI)

    Returns:
        GitHub token string, or None if not available
    """
    # 1. Check environment
    token = os.environ.get('GITHUB_TOKEN')
    if token:
        return token

    # 2. Try gh CLI
    result = subprocess.run(
        ['gh', 'auth', 'token'],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()

    return None


# ==============================================================================
# Server Entry Point
# ==============================================================================


def main() -> None:
    """Run the MCP server."""
    server.run()


if __name__ == '__main__':
    main()
