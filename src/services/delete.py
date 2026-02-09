"""
Session delete service - atomic deletion with rollback on failure.

Implements strong exception safety guarantee:
- Operations either complete successfully or leave system in original state
- No partial deletion state - atomic rollback on any failure

Error Boundary Pattern:
- Context manager catches ALL exceptions (including KeyboardInterrupt)
- Performs rollback, then re-raises
- Calling code catches only expected exceptions for graceful handling
- Unexpected exceptions propagate with full traceback (fail loudly)

Safety features:
- Native sessions (UUIDv4) require --force flag
- Backup always created for rollback capability
- --no-backup means "don't keep backup after success" (rollback still works)
- Dry-run mode to preview deletions
- Explicit file enumeration - no directory-level deletion
- Validation fails fast on unexpected files

Backup location: ~/.claude-session-mcp/deleted/
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

from src.exceptions import NativeSessionDeletionError
from src.paths import encode_path
from src.protocols import LoggerProtocol
from src.schemas.operations.archive import (
    SessionArchiveV1,
    SessionArchiveV2,
    migrate_v1_to_v2,
)
from src.schemas.operations.delete import ArtifactFile, DeleteManifest, DeleteResult
from src.schemas.session import SessionRecordAdapter
from src.services.archive import SessionArchiveService
from src.services.artifacts import (
    SESSION_ENV_DIR,
    TASKS_DIR,
    TODOS_DIR,
    classify_task_directory,
    create_session_env_dir,
    extract_slugs_from_records,
    get_tool_results_dir,
    write_jsonl,
    write_plan_files,
    write_task_metadata,
    write_tasks,
    write_todos,
    write_tool_results,
)
from src.services.parser import SessionParserService
from src.storage.local import LocalFileSystemStorage

__all__ = [
    'SessionDeleteService',
    'is_native_session',
]

logger = logging.getLogger(__name__)


class SessionDeleteService:
    """
    Service for atomic session deletion with rollback on failure.

    Implements strong exception safety guarantee:
    - Either all artifacts are deleted successfully
    - Or system is rolled back to original state (no partial deletion)

    Safety features:
    - Native sessions (UUIDv4) require force=True
    - Backup always created for rollback capability
    - no_backup=True just removes backup after success (rollback still works)
    - Explicit file enumeration with validation for unexpected files
    """

    # Backup location - separate from ~/.claude/
    DELETED_SESSIONS_DIR = Path.home() / '.claude-session-mcp' / 'deleted'

    # Expected file extensions in tool-results directory
    # If new extensions appear, Claude Code changed and we need to update
    EXPECTED_TOOL_RESULT_EXTENSIONS = frozenset({'.txt'})

    # Only permission errors are truly "expected" - environment issue, not a bug
    # Other errors indicate unexpected state and should propagate after rollback:
    # - FileNotFoundError: file disappeared = unexpected state change
    # - OSError ENOTEMPTY: directory not empty = bug in discovery
    # - IsADirectoryError: wrong operation = bug in our code
    EXPECTED_DELETION_ERRORS = (PermissionError,)

    def __init__(self, project_path: Path) -> None:
        """Initialize delete service."""
        self.project_path = project_path.resolve()
        self.claude_sessions_dir = Path.home() / '.claude' / 'projects'
        self.parser_service = SessionParserService()

    @asynccontextmanager
    async def _atomic_deletion(self, backup_path: Path) -> AsyncGenerator[None]:
        """
        Error boundary ensuring atomic deletion with rollback on any failure.

        This context manager provides strong exception safety:
        - On ANY exception within the context, rollback is performed
        - The original exception is always re-raised after cleanup
        - Calling code decides how to handle specific exception types

        The pattern separates error containment (this context manager) from
        error handling (the calling code's try/except).

        If backup cleanup fails after rollback, Python's exception chaining
        preserves the original exception in __context__, visible in traceback.

        Args:
            backup_path: Path to backup file for rollback

        Yields:
            Nothing - just provides the error boundary

        Raises:
            Re-raises any exception after performing rollback
        """
        try:
            yield
        except asyncio.CancelledError:
            # Task cancellation - still perform cleanup, then propagate
            # Separate handling for clarity and potential future customization
            logger.warning('Deletion cancelled, performing rollback...')
            try:
                await self._rollback_from_backup(backup_path)
                logger.info('Rollback completed after cancellation')
            except Exception as rollback_error:
                logger.error(f'Rollback failed during cancellation: {rollback_error}')
                # Keep backup for manual recovery
            else:
                # Rollback succeeded - remove backup
                # No try/except: if unlink fails, original exception preserved via __context__
                backup_path.unlink(missing_ok=True)
            raise

        except BaseException as original_exc:
            # All other exceptions (including KeyboardInterrupt, SystemExit)
            logger.error(f'Deletion failed: {original_exc}, performing rollback...')
            try:
                await self._rollback_from_backup(backup_path)
                logger.info('Rollback completed successfully')
            except Exception as rollback_error:
                logger.error(f'Rollback failed: {rollback_error}')
                # Keep backup for manual recovery
                original_exc.add_note(f'Rollback failed: {rollback_error}')
                original_exc.add_note(f'Backup preserved at: {backup_path}')
            else:
                # Rollback succeeded - remove backup
                # No try/except: if unlink fails, original exception preserved via __context__
                backup_path.unlink(missing_ok=True)
                original_exc.add_note('Rollback completed successfully')
            raise

    async def discover_artifacts(
        self,
        session_id: str,
        logger: LoggerProtocol | None = None,
    ) -> DeleteManifest:
        """
        Find all artifacts for a session with explicit file enumeration.

        Discovers and validates:
        - Session JSONL files (main + agents)
        - Plan files (by extracting slugs from records)
        - Tool result files (validates expected extensions)
        - Todo files
        - Session-env files (expected to be empty - any file is unexpected)

        Tracks unexpected files separately - these cause validation failure
        before any deletion occurs.

        Args:
            session_id: Session ID to discover artifacts for
            logger: Optional logger

        Returns:
            DeleteManifest with all discovered artifacts and validation info

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

        # Initialize collections
        artifacts: list[ArtifactFile] = []
        agent_file_paths: list[str] = []
        plan_file_paths: list[str] = []
        tool_result_paths: list[str] = []
        todo_file_paths: list[str] = []
        task_file_paths: list[str] = []
        directories_to_cleanup: list[str] = []
        unexpected_files: list[str] = []

        # 1. Main session file
        main_size = main_file.stat().st_size
        artifacts.append(
            ArtifactFile(
                path=str(main_file),
                size_bytes=main_size,
                artifact_type='session_main',
            )
        )

        # 2. Find agent files
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

        # Track subagents directory if it exists (2.1.2+ nested structure)
        subagents_dir = session_dir / session_id / 'subagents'
        if subagents_dir.exists():
            # Check for unexpected files in subagents/ (non-agent files)
            for path in subagents_dir.iterdir():
                if path.is_file() and not path.name.startswith('agent-'):
                    unexpected_files.append(str(path))
                elif path.is_dir():
                    # Unexpected subdirectory in subagents/
                    directories_to_cleanup.append(str(path))
            directories_to_cleanup.append(str(subagents_dir))

        # Also track parent session directory if it exists
        session_parent_dir = session_dir / session_id
        if session_parent_dir.exists() and str(session_parent_dir) not in directories_to_cleanup:
            directories_to_cleanup.append(str(session_parent_dir))

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

        # 4. Tool results - explicit enumeration with extension validation
        tool_results_dir = get_tool_results_dir(session_dir, session_id)
        tool_results_parent = session_dir / session_id

        if tool_results_dir.exists():
            for path in tool_results_dir.rglob('*'):
                if path.is_file():
                    if path.suffix in self.EXPECTED_TOOL_RESULT_EXTENSIONS:
                        size = path.stat().st_size
                        artifacts.append(
                            ArtifactFile(
                                path=str(path),
                                size_bytes=size,
                                artifact_type='tool_result',
                            )
                        )
                        tool_result_paths.append(str(path))
                    else:
                        # Unexpected file type - Claude Code may have changed
                        unexpected_files.append(str(path))
                elif path.is_dir():
                    # Track subdirectories for cleanup
                    directories_to_cleanup.append(str(path))

            # Track tool-results directory itself
            directories_to_cleanup.append(str(tool_results_dir))

        # Track parent session directory if it exists
        if tool_results_parent.exists():
            directories_to_cleanup.append(str(tool_results_parent))

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

        # 6. Task files (CLAUDE_CODE_ENABLE_TASKS mode)
        task_contents = classify_task_directory(session_id)
        if task_contents is not None:
            # Task data files - backed up
            for path in task_contents.task_files:
                size = path.stat().st_size
                artifacts.append(
                    ArtifactFile(
                        path=str(path),
                        size_bytes=size,
                        artifact_type='task_file',
                    )
                )
                task_file_paths.append(str(path))

            # Metadata files - all deleted, .lock not backed up (ephemeral)
            for path in task_contents.metadata_files:
                size = path.stat().st_size
                artifacts.append(
                    ArtifactFile(
                        path=str(path),
                        size_bytes=size,
                        artifact_type='task_metadata' if path.name != '.lock' else 'task_lock',
                    )
                )

            # Unknown files - fail fast
            unexpected_files.extend(str(path) for path in task_contents.unexpected_files)

            # Track directory for cleanup
            tasks_session_dir = TASKS_DIR / session_id
            directories_to_cleanup.append(str(tasks_session_dir))

        if logger and task_file_paths:
            await logger.info(f'Found {len(task_file_paths)} task files and metadata')

        # 7. Session-env - expected to be empty, any file is unexpected
        session_env_dir = SESSION_ENV_DIR / session_id
        if session_env_dir.exists():
            for path in session_env_dir.rglob('*'):
                if path.is_file():
                    # ANY file in session-env is unexpected
                    unexpected_files.append(str(path))
                elif path.is_dir():
                    directories_to_cleanup.append(str(path))

            # Track session-env directory itself
            directories_to_cleanup.append(str(session_env_dir))

        # Sort directories deepest-first for rmdir order
        directories_to_cleanup.sort(key=lambda p: -p.count('/'))

        # Calculate totals
        total_size = sum(a.size_bytes for a in artifacts)

        # Check if native session
        native = is_native_session(session_id)
        created_at = None if native else get_restoration_timestamp(session_id)

        if logger:
            await logger.info(f'Session type: {"native (UUIDv4)" if native else "cloned/restored (UUIDv7)"}')
            await logger.info(f'Total artifacts: {len(artifacts)} files, {len(directories_to_cleanup)} directories')
            await logger.info(f'Total size: {total_size:,} bytes')
            if unexpected_files:
                await logger.warning(f'Found {len(unexpected_files)} unexpected files')

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
            task_files=task_file_paths,
            directories_to_cleanup=directories_to_cleanup,
            unexpected_files=unexpected_files,
        )

    async def delete_session(
        self,
        session_id: str,
        force: bool = False,
        no_backup: bool = False,
        dry_run: bool = False,
        logger: LoggerProtocol | None = None,
        terminate_pid_before_delete: int | None = None,
    ) -> DeleteResult:
        """
        Delete session artifacts with atomic rollback on failure.

        Implements strong exception safety:
        - Either all artifacts are deleted successfully
        - Or system is rolled back to original state

        The --no-backup flag controls whether the backup is kept after
        successful deletion (for user undo capability). Rollback always
        works regardless of this flag - backup is created temporarily
        for atomicity, then either kept or removed based on the flag.

        Args:
            session_id: Session to delete
            force: Required to delete native (UUIDv4) sessions
            no_backup: Don't keep backup after success (rollback still works)
            dry_run: Preview what would be deleted

        Returns:
            DeleteResult with deletion details
        """
        start_time = datetime.now(UTC)

        # Discover all artifacts
        manifest = await self.discover_artifacts(session_id, logger)

        # Validation: check for unexpected files (fail fast)
        if manifest.unexpected_files:
            error_lines = [f'  - {f}' for f in manifest.unexpected_files]
            error_msg = (
                f'Found {len(manifest.unexpected_files)} unexpected files:\n'
                + '\n'.join(error_lines)
                + '\n\nClaude Code may have changed. Update discovery logic to handle these files.'
            )
            if logger:
                await logger.error(error_msg)

            return DeleteResult(
                session_id=session_id,
                was_dry_run=dry_run,
                success=False,
                error_message=error_msg,
                backup_path=None,
                files_deleted=0,
                size_freed_bytes=0,
                deleted_files=[],
                session_files_deleted=0,
                plan_files_deleted=0,
                tool_results_deleted=0,
                todos_deleted=0,
                tasks_deleted=0,
                directories_removed=[],
                duration_ms=(datetime.now(UTC) - start_time).total_seconds() * 1000,
                deleted_at=datetime.now(UTC),
            )

        # Safety check for native sessions (skip for dry_run)
        if manifest.is_native and not force and not dry_run:
            raise NativeSessionDeletionError(session_id)

        # Dry run - return what would be deleted
        if dry_run:
            if logger:
                await logger.info('Dry run - no files will be deleted')
                await logger.info(
                    f'Would delete {len(manifest.files)} files, '
                    f'{len(manifest.directories_to_cleanup)} directories '
                    f'({manifest.total_size_bytes:,} bytes)'
                )

            # Compute artifact counts for dry run
            counts = self._compute_artifact_counts(manifest)

            return DeleteResult(
                session_id=session_id,
                was_dry_run=True,
                success=True,
                error_message=None,
                backup_path=None,
                files_deleted=len(manifest.files),
                size_freed_bytes=manifest.total_size_bytes,
                deleted_files=[a.path for a in manifest.files],
                session_files_deleted=counts['session_files'],
                plan_files_deleted=counts['plan_files'],
                tool_results_deleted=counts['tool_results'],
                todos_deleted=counts['todos'],
                tasks_deleted=counts['tasks'],
                directories_removed=list(manifest.directories_to_cleanup),
                duration_ms=(datetime.now(UTC) - start_time).total_seconds() * 1000,
                deleted_at=datetime.now(UTC),
            )

        # Always create backup for atomic rollback capability
        if logger:
            await logger.info('Creating backup for atomic rollback...')

        try:
            backup_path = Path(await self._create_backup(session_id, logger))
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
                session_files_deleted=0,
                plan_files_deleted=0,
                tool_results_deleted=0,
                todos_deleted=0,
                tasks_deleted=0,
                directories_removed=[],
                duration_ms=(datetime.now(UTC) - start_time).total_seconds() * 1000,
                deleted_at=datetime.now(UTC),
            )

        # Self-deletion: terminate the calling process to prevent session file recreation
        # This happens AFTER backup (rollback safety) but BEFORE deletion (no interference)
        # Uses SIGKILL because SIGTERM allows graceful shutdown which writes to session file
        if terminate_pid_before_delete is not None:
            if logger:
                await logger.info(f'Terminating process {terminate_pid_before_delete} (SIGKILL) before deletion')
            os.kill(terminate_pid_before_delete, signal.SIGKILL)
            time.sleep(0.2)  # Brief pause to ensure process termination

        # Perform atomic deletion with rollback on any failure
        try:
            async with self._atomic_deletion(backup_path):
                # Delete all files
                deleted_files: list[str] = []
                size_freed = 0

                for artifact in manifest.files:
                    path = Path(artifact.path)
                    path.unlink()
                    deleted_files.append(artifact.path)
                    size_freed += artifact.size_bytes
                    if logger:
                        await logger.info(f'Deleted: {artifact.path}')

                # Delete all directories (sorted deepest-first)
                directories_removed: list[str] = []

                for dir_path in manifest.directories_to_cleanup:
                    path = Path(dir_path)
                    if path.exists():
                        path.rmdir()  # Fails if not empty - intentional!
                        directories_removed.append(dir_path)
                        if logger:
                            await logger.info(f'Removed directory: {dir_path}')

        except self.EXPECTED_DELETION_ERRORS as e:
            # Expected filesystem error - rollback already happened
            if logger:
                await logger.error(f'Deletion failed (rolled back): {e}')

            return DeleteResult(
                session_id=session_id,
                was_dry_run=False,
                success=False,
                error_message=f'Deletion failed (rolled back): {e}',
                backup_path=None,
                files_deleted=0,
                size_freed_bytes=0,
                deleted_files=[],
                session_files_deleted=0,
                plan_files_deleted=0,
                tool_results_deleted=0,
                todos_deleted=0,
                tasks_deleted=0,
                directories_removed=[],
                duration_ms=(datetime.now(UTC) - start_time).total_seconds() * 1000,
                deleted_at=datetime.now(UTC),
            )

        # Unexpected exceptions propagate after rollback (fail loudly)
        # If we reach here, deletion succeeded

        # Handle backup based on --no-backup flag
        final_backup_path: str | None = None
        if no_backup:
            # User doesn't want undo capability - remove backup
            backup_path.unlink(missing_ok=True)
            if logger:
                await logger.info('Backup removed (--no-backup specified)')
        else:
            # Keep backup for potential undo
            final_backup_path = str(backup_path)

        end_time = datetime.now(UTC)

        if logger:
            await logger.info(
                f'Deleted {len(deleted_files)} files, {len(directories_removed)} directories ({size_freed:,} bytes)'
            )

        # Compute per-artifact-type counts
        counts = self._compute_artifact_counts(manifest)

        return DeleteResult(
            session_id=session_id,
            was_dry_run=False,
            success=True,
            error_message=None,
            backup_path=final_backup_path,
            files_deleted=len(deleted_files),
            size_freed_bytes=size_freed,
            deleted_files=deleted_files,
            session_files_deleted=counts['session_files'],
            plan_files_deleted=counts['plan_files'],
            tool_results_deleted=counts['tool_results'],
            todos_deleted=counts['todos'],
            tasks_deleted=counts['tasks'],
            directories_removed=directories_removed,
            duration_ms=(end_time - start_time).total_seconds() * 1000,
            deleted_at=end_time,
        )

    async def _create_backup(
        self,
        session_id: str,
        logger: LoggerProtocol | None,
    ) -> str:
        """
        Create a backup archive for rollback capability.

        Uses SessionArchiveService to create a consistent archive format
        that can be restored with restore --in-place.
        """
        self.DELETED_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime('%Y%m%d-%H%M%S')
        backup_filename = f'{session_id}-{timestamp}.json'
        backup_path = self.DELETED_SESSIONS_DIR / backup_filename

        archive_service = SessionArchiveService(
            session_id=session_id,
            temp_dir=self.DELETED_SESSIONS_DIR,
            parser_service=self.parser_service,
            project_path=self.project_path,
        )

        storage = LocalFileSystemStorage(self.DELETED_SESSIONS_DIR)
        metadata = await archive_service.create_archive(
            storage=storage,
            output_path=str(backup_path),
            format_param='json',
            logger=logger,
        )

        return metadata.file_path

    @staticmethod
    def _parse_backup_data(data: dict[str, Any]) -> SessionArchiveV2:
        """
        Parse backup data with version detection.

        Mirrors restore service's approach for consistency.
        Backups created by this service are always v2, but this handles
        legacy backups for robustness.

        Args:
            data: Parsed JSON backup data

        Returns:
            SessionArchiveV2 (v1 is migrated in memory)
        """
        version = data.get('version', '1.0')

        if version.startswith('2.'):
            return SessionArchiveV2.model_validate(data)
        else:
            # V1 backup (legacy) - convert records then migrate
            for filename, records in data['files'].items():
                data['files'][filename] = [SessionRecordAdapter.validate_python(r) for r in records]
            v1 = SessionArchiveV1.model_validate(data)
            return migrate_v1_to_v2(v1)

    async def _rollback_from_backup(self, backup_path: Path) -> None:
        """
        Restore session from backup after failed deletion.

        Reads the backup archive and writes files back to their original
        locations (in-place restoration with original session ID).

        Uses SessionArchiveV2 structure (v1 backups are migrated).
        """
        logger.info(f'Rolling back from backup: {backup_path}')

        # Parse backup using version detection
        backup_data = json.loads(backup_path.read_text(encoding='utf-8'))
        archive = self._parse_backup_data(backup_data)

        # Get target directory from archive metadata
        encoded_path = encode_path(Path(archive.original_project_path))
        target_dir = self.claude_sessions_dir / encoded_path
        target_dir.mkdir(parents=True, exist_ok=True)

        # Restore main session file
        # write_jsonl uses exclude_unset=True for round-trip fidelity
        # Empty mappings ({}, {}) = identity (no slug/agent ID remapping for rollback)
        main_file_path = target_dir / f'{archive.session_id}.jsonl'
        write_jsonl(main_file_path, archive.main_session.records, {}, {})
        logger.info(f'Restored: {main_file_path}')

        # Restore agent files (with nested structure support)
        for agent in archive.agent_files:
            agent_filename = f'agent-{agent.agent_id}.jsonl'

            if agent.nested:
                agent_dir = target_dir / archive.session_id / 'subagents'
            else:
                agent_dir = target_dir

            agent_dir.mkdir(parents=True, exist_ok=True)
            agent_path = agent_dir / agent_filename
            write_jsonl(agent_path, agent.records, {}, {})
            logger.info(f'Restored: {agent_path}')

        # Restore tool results (exist_ok=True: files may survive partial deletion)
        if archive.tool_results:
            tool_results_mapping = {tr.tool_use_id: tr.content for tr in archive.tool_results}
            write_tool_results(tool_results_mapping, target_dir, archive.session_id, exist_ok=True)
            logger.info(f'Restored {len(archive.tool_results)} tool result files')

        # Restore todo files (exist_ok=True: files may survive partial deletion)
        if archive.todos:
            write_todos(
                ((f'{archive.session_id}-agent-{todo.agent_id}.json', todo.content) for todo in archive.todos),
                exist_ok=True,
            )
            logger.info(f'Restored {len(archive.todos)} todo files')

        # Restore task files (exist_ok=True: files may survive partial deletion)
        if archive.tasks:
            tasks_restored = write_tasks(archive.session_id, archive.tasks, exist_ok=True)
            logger.info(f'Restored {tasks_restored} task files')

        # Restore task metadata (exist_ok=True: files may survive partial deletion)
        if archive.task_metadata:
            metadata_restored = write_task_metadata(archive.session_id, archive.task_metadata, exist_ok=True)
            logger.info(f'Restored {metadata_restored} task metadata files')

        # Restore plan files (exist_ok=True: files may survive partial deletion)
        if archive.plan_files:
            write_plan_files(
                ((plan.slug, plan.content) for plan in archive.plan_files),
                exist_ok=True,
            )
            logger.info(f'Restored {len(archive.plan_files)} plan files')

        # Recreate session-env directory
        create_session_env_dir(archive.session_id)

        logger.info('Rollback completed successfully')

    @staticmethod
    def _compute_artifact_counts(manifest: DeleteManifest) -> dict[str, int]:
        """
        Compute per-artifact-type counts from manifest.

        Returns:
            Dict with keys: session_files, plan_files, tool_results, todos, tasks
        """
        return {
            'session_files': 1 + len(manifest.agent_files),  # main + agents
            'plan_files': len(manifest.plan_files),
            'tool_results': len(manifest.tool_result_files),
            'todos': len(manifest.todo_files),
            'tasks': len(manifest.task_files),
        }


class ArtifactCounts(TypedDict):
    """Per-artifact-type deletion counts."""

    session_files: int  # main + agents
    plan_files: int
    tool_results: int
    todos: int
    tasks: int


# ==============================================================================
# Utility Functions
# ==============================================================================


def is_native_session(session_id: str) -> bool:
    """
    Check if session is native (UUIDv4) vs cloned/restored (UUIDv7).

    Native Claude sessions use UUIDv4 (random).
    Cloned/restored sessions use UUIDv7 (time-ordered).
    """
    try:
        uid = uuid.UUID(session_id)
        return uid.version != 7
    except ValueError:
        # Invalid UUID format - treat as native (safer)
        return True


def get_restoration_timestamp(session_id: str) -> datetime | None:
    """
    Extract the restoration timestamp from a restored session ID.

    UUIDv7 embeds a Unix timestamp in the first 48 bits.
    """
    try:
        uid = uuid.UUID(session_id)
        if uid.version != 7:
            return None
        timestamp_ms = int.from_bytes(uid.bytes[:6], 'big')
        timestamp_s = timestamp_ms / 1000
        return datetime.fromtimestamp(timestamp_s, tz=UTC)
    except (ValueError, OSError):
        return None
