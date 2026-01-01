"""
Session delete service - removes session artifacts with safety features.

Safety features:
- Native sessions (UUIDv4) require --force flag
- Auto-backup before deletion (unless --no-backup)
- Dry-run mode to preview deletions
- Fail-fast if any file cannot be deleted

Backup location: ~/.claude-session-mcp/deleted/
This keeps our tool's data separate from Claude Code's ~/.claude/ directory.
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from src.base_model import StrictModel
from src.paths import encode_path
from src.protocols import LoggerProtocol
from src.services.archive import SessionArchiveService
from src.services.artifacts import (
    SESSION_ENV_DIR,
    TODOS_DIR,
    extract_slugs_from_records,
    get_tool_results_dir,
)
from src.services.parser import SessionParserService
from src.storage.local import LocalFileSystemStorage

# Backup location - separate from ~/.claude/
DELETED_SESSIONS_DIR = Path.home() / '.claude-session-mcp' / 'deleted'


# ==============================================================================
# Models
# ==============================================================================


class DeleteManifest(StrictModel):
    """Discovery result for a session to delete."""

    session_id: str
    is_native: bool  # True if native session (UUIDv4), requires --force
    created_at: datetime | None  # Extracted from UUIDv7 if applicable

    files: Sequence[ArtifactFile]
    total_size_bytes: int

    session_main_file: str | None
    agent_files: Sequence[str]
    plan_files: Sequence[str]
    tool_result_files: Sequence[str]
    todo_files: Sequence[str]
    session_env_dir: str | None


class ArtifactFile(StrictModel):
    """Single file in delete manifest."""

    path: str
    size_bytes: int
    artifact_type: Literal[
        'session_main',
        'session_agent',
        'plan_file',
        'tool_result',
        'todo_file',
        'session_env',
    ]


class DeleteResult(StrictModel):
    """Execution result."""

    session_id: str
    was_dry_run: bool
    success: bool
    error_message: str | None

    backup_path: str | None  # Path to backup archive (if created)
    files_deleted: int
    size_freed_bytes: int
    deleted_files: Sequence[str]
    failed_deletions: Sequence[str]

    duration_ms: float
    deleted_at: datetime


# ==============================================================================
# Utility Functions
# ==============================================================================


def is_native_session(session_id: str) -> bool:
    """
    Check if session is native (UUIDv4) vs cloned/restored (UUIDv7).

    Native Claude sessions use UUIDv4 (random).
    Cloned/restored sessions use UUIDv7 (time-ordered).

    Args:
        session_id: Session ID to check

    Returns:
        True if native (UUIDv4), False if cloned/restored (UUIDv7)
    """
    try:
        uid = uuid.UUID(session_id)
        # UUIDv7 has version 7, UUIDv4 has version 4
        return uid.version != 7
    except ValueError:
        # Invalid UUID format - treat as native (safer)
        return True


def get_restoration_timestamp(session_id: str) -> datetime | None:
    """
    Extract the restoration timestamp from a restored session ID.

    UUIDv7 embeds a Unix timestamp in the first 48 bits.

    Args:
        session_id: Session ID (must be UUIDv7)

    Returns:
        Restoration datetime or None if not a UUIDv7
    """
    try:
        uid = uuid.UUID(session_id)
        if uid.version != 7:
            return None

        # Extract timestamp from first 48 bits (in milliseconds)
        timestamp_ms = int.from_bytes(uid.bytes[:6], 'big')
        timestamp_s = timestamp_ms / 1000
        return datetime.fromtimestamp(timestamp_s, tz=UTC)
    except (ValueError, OSError):
        return None


# ==============================================================================
# Delete Service
# ==============================================================================


class SessionDeleteService:
    """
    Service for deleting session artifacts with safety features.

    Safety features:
    - Native sessions (UUIDv4) require force=True
    - Auto-backup to ~/.claude-session-mcp/deleted/ before deletion
    - Dry-run mode to preview what would be deleted
    """

    def __init__(self, project_path: Path) -> None:
        """
        Initialize delete service.

        Args:
            project_path: Project directory where session exists
        """
        self.project_path = project_path.resolve()
        self.claude_sessions_dir = Path.home() / '.claude' / 'projects'
        self.parser_service = SessionParserService()

    async def discover_artifacts(
        self,
        session_id: str,
        logger: LoggerProtocol | None = None,
    ) -> DeleteManifest:
        """
        Find all artifacts for a session.

        Discovers:
        - Session JSONL files (main + agents)
        - Plan files (by extracting slugs from records)
        - Tool results directory
        - Todo files
        - Session-env directory

        Args:
            session_id: Session ID to discover artifacts for
            logger: Optional logger

        Returns:
            DeleteManifest with all discovered artifacts

        Raises:
            FileNotFoundError: If main session file not found
        """
        if logger:
            await logger.info(f'Discovering artifacts for session: {session_id}')

        # Get session directory
        encoded_path = encode_path(self.project_path)
        session_dir = self.claude_sessions_dir / encoded_path

        # Find main session file
        main_file = session_dir / f'{session_id}.jsonl'
        if not main_file.exists():
            raise FileNotFoundError(f'Session not found: {main_file}')

        # Initialize artifact collections
        artifacts: list[ArtifactFile] = []
        agent_file_paths: list[str] = []
        plan_file_paths: list[str] = []
        tool_result_paths: list[str] = []
        todo_file_paths: list[str] = []
        session_env_path: str | None = None

        # 1. Main session file
        main_size = main_file.stat().st_size
        artifacts.append(
            ArtifactFile(
                path=str(main_file),
                size_bytes=main_size,
                artifact_type='session_main',
            )
        )

        # 2. Find agent files by loading main session and discovering
        #    (reuse the same discovery pattern as clone/archive)
        import subprocess

        result = subprocess.run(
            [
                'rg',
                '--files-with-matches',
                f'"sessionId":\\s*"{session_id}"',
                '--glob',
                'agent-*.jsonl',
                str(session_dir),
            ],
            capture_output=True,
            text=True,
        )

        if result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                agent_path = Path(line)
                if agent_path.exists():
                    size = agent_path.stat().st_size
                    artifacts.append(
                        ArtifactFile(
                            path=str(agent_path),
                            size_bytes=size,
                            artifact_type='session_agent',
                        )
                    )
                    agent_file_paths.append(str(agent_path))

        # 3. Extract slugs from session records to find plan files
        session_files = [main_file] + [Path(p) for p in agent_file_paths]
        files_data = await self.parser_service.load_session_files(session_files, logger)
        slugs = extract_slugs_from_records(files_data)

        plans_dir = Path.home() / '.claude' / 'plans'
        for slug in slugs:
            plan_path = plans_dir / f'{slug}.md'
            if plan_path.exists():
                size = plan_path.stat().st_size
                artifacts.append(
                    ArtifactFile(
                        path=str(plan_path),
                        size_bytes=size,
                        artifact_type='plan_file',
                    )
                )
                plan_file_paths.append(str(plan_path))

        if logger:
            await logger.info(f'Found {len(slugs)} slugs, {len(plan_file_paths)} plan files')

        # 4. Tool results directory
        tool_results_dir = get_tool_results_dir(session_dir, session_id)
        if tool_results_dir.exists():
            for path in tool_results_dir.glob('*.txt'):
                size = path.stat().st_size
                artifacts.append(
                    ArtifactFile(
                        path=str(path),
                        size_bytes=size,
                        artifact_type='tool_result',
                    )
                )
                tool_result_paths.append(str(path))

        if logger and tool_result_paths:
            await logger.info(f'Found {len(tool_result_paths)} tool result files')

        # 5. Todo files
        if TODOS_DIR.exists():
            for path in TODOS_DIR.glob(f'{session_id}-agent-*.json'):
                size = path.stat().st_size
                artifacts.append(
                    ArtifactFile(
                        path=str(path),
                        size_bytes=size,
                        artifact_type='todo_file',
                    )
                )
                todo_file_paths.append(str(path))

        if logger and todo_file_paths:
            await logger.info(f'Found {len(todo_file_paths)} todo files')

        # 6. Session-env directory
        session_env_dir = SESSION_ENV_DIR / session_id
        if session_env_dir.exists():
            session_env_path = str(session_env_dir)
            # Count size of all files in the directory
            env_size = sum(f.stat().st_size for f in session_env_dir.rglob('*') if f.is_file())
            if env_size > 0 or session_env_dir.exists():
                artifacts.append(
                    ArtifactFile(
                        path=session_env_path,
                        size_bytes=env_size,
                        artifact_type='session_env',
                    )
                )

        # Calculate totals
        total_size = sum(a.size_bytes for a in artifacts)

        # Check if native session
        native = is_native_session(session_id)
        created_at = None if native else get_restoration_timestamp(session_id)

        if logger:
            await logger.info(f'Session type: {"native (UUIDv4)" if native else "cloned/restored (UUIDv7)"}')
            await logger.info(f'Total artifacts: {len(artifacts)}, size: {total_size:,} bytes')

        return DeleteManifest(
            session_id=session_id,
            is_native=native,
            created_at=created_at,
            files=artifacts,
            total_size_bytes=total_size,
            session_main_file=str(main_file),
            agent_files=agent_file_paths,
            plan_files=plan_file_paths,
            tool_result_files=tool_result_paths,
            todo_files=todo_file_paths,
            session_env_dir=session_env_path,
        )

    async def delete_session(
        self,
        session_id: str,
        force: bool = False,
        no_backup: bool = False,
        dry_run: bool = False,
        logger: LoggerProtocol | None = None,
    ) -> DeleteResult:
        """
        Delete session artifacts.

        Args:
            session_id: Session to delete
            force: Required to delete native (UUIDv4) sessions
            no_backup: Skip auto-backup before deletion
            dry_run: Show what would be deleted without actually deleting

        Returns:
            DeleteResult with deletion details

        Raises:
            ValueError: If native session and force not provided
            FileNotFoundError: If session not found
        """
        start_time = datetime.now(UTC)

        # Discover all artifacts
        manifest = await self.discover_artifacts(session_id, logger)

        # Safety check for native sessions
        if manifest.is_native and not force:
            return DeleteResult(
                session_id=session_id,
                was_dry_run=dry_run,
                success=False,
                error_message=(
                    f'Session {session_id} is a native Claude session (UUIDv4). Use --force to delete native sessions.'
                ),
                backup_path=None,
                files_deleted=0,
                size_freed_bytes=0,
                deleted_files=[],
                failed_deletions=[],
                duration_ms=0,
                deleted_at=start_time,
            )

        # Dry run - just return what would be deleted
        if dry_run:
            if logger:
                await logger.info('Dry run - no files will be deleted')
                await logger.info(f'Would delete {len(manifest.files)} artifacts ({manifest.total_size_bytes:,} bytes)')

            end_time = datetime.now(UTC)
            duration_ms = (end_time - start_time).total_seconds() * 1000

            return DeleteResult(
                session_id=session_id,
                was_dry_run=True,
                success=True,
                error_message=None,
                backup_path=None,
                files_deleted=len(manifest.files),
                size_freed_bytes=manifest.total_size_bytes,
                deleted_files=[a.path for a in manifest.files],
                failed_deletions=[],
                duration_ms=duration_ms,
                deleted_at=end_time,
            )

        # Create backup unless --no-backup
        backup_path: str | None = None
        if not no_backup:
            if logger:
                await logger.info('Creating backup before deletion...')

            try:
                backup_path = await self._create_backup(session_id, logger)
                if logger:
                    await logger.info(f'Backup created: {backup_path}')
            except Exception as e:
                return DeleteResult(
                    session_id=session_id,
                    was_dry_run=False,
                    success=False,
                    error_message=f'Failed to create backup: {e}',
                    backup_path=None,
                    files_deleted=0,
                    size_freed_bytes=0,
                    deleted_files=[],
                    failed_deletions=[],
                    duration_ms=(datetime.now(UTC) - start_time).total_seconds() * 1000,
                    deleted_at=datetime.now(UTC),
                )

        # Delete all artifacts
        deleted_files: list[str] = []
        failed_deletions: list[str] = []
        size_freed = 0

        for artifact in manifest.files:
            path = Path(artifact.path)
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                deleted_files.append(artifact.path)
                size_freed += artifact.size_bytes
                if logger:
                    await logger.info(f'Deleted: {artifact.path}')
            except Exception as e:
                failed_deletions.append(f'{artifact.path}: {e}')
                if logger:
                    await logger.error(f'Failed to delete {artifact.path}: {e}')

        # Clean up empty directories
        await self._cleanup_empty_dirs(session_id, logger)

        end_time = datetime.now(UTC)
        duration_ms = (end_time - start_time).total_seconds() * 1000

        success = len(failed_deletions) == 0

        return DeleteResult(
            session_id=session_id,
            was_dry_run=False,
            success=success,
            error_message=None if success else f'{len(failed_deletions)} files failed to delete',
            backup_path=backup_path,
            files_deleted=len(deleted_files),
            size_freed_bytes=size_freed,
            deleted_files=deleted_files,
            failed_deletions=failed_deletions,
            duration_ms=duration_ms,
            deleted_at=end_time,
        )

    async def _create_backup(
        self,
        session_id: str,
        logger: LoggerProtocol | None,
    ) -> str:
        """
        Create a backup archive before deletion.

        Uses SessionArchiveService to create a consistent archive format
        that can be restored with restore --in-place.

        Args:
            session_id: Session ID to backup
            logger: Logger for progress messages

        Returns:
            Path to created backup archive
        """
        # Ensure backup directory exists
        DELETED_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

        # Generate backup filename with timestamp
        timestamp = datetime.now(UTC).strftime('%Y%m%d-%H%M%S')
        backup_filename = f'{session_id}-{timestamp}.json'
        backup_path = DELETED_SESSIONS_DIR / backup_filename

        # Create archive service
        archive_service = SessionArchiveService(
            session_id=session_id,
            temp_dir=DELETED_SESSIONS_DIR,
            parser_service=self.parser_service,
            project_path=self.project_path,  # Real project path from CLI/MCP
        )

        # Create storage backend and archive
        storage = LocalFileSystemStorage(DELETED_SESSIONS_DIR)
        metadata = await archive_service.create_archive(
            storage=storage,
            output_path=str(backup_path),
            format_param='json',
            logger=logger,
        )

        return metadata.file_path

    async def _cleanup_empty_dirs(
        self,
        session_id: str,
        logger: LoggerProtocol | None,
    ) -> None:
        """Clean up empty directories after deletion."""
        # Tool results parent directory
        encoded_path = encode_path(self.project_path)
        session_dir = self.claude_sessions_dir / encoded_path
        tool_results_parent = session_dir / session_id

        if tool_results_parent.exists() and not any(tool_results_parent.iterdir()):
            try:
                tool_results_parent.rmdir()
                if logger:
                    await logger.info(f'Removed empty directory: {tool_results_parent}')
            except OSError:
                pass  # Directory not empty or other issue

        # Session-env directory (should already be deleted if it existed)
        session_env_dir = SESSION_ENV_DIR / session_id
        if session_env_dir.exists() and not any(session_env_dir.iterdir()):
            try:
                session_env_dir.rmdir()
                if logger:
                    await logger.info(f'Removed empty directory: {session_env_dir}')
            except OSError:
                pass
