"""
Session move service - relocate sessions between projects.

Moves a session from one project to another, preserving the original session ID.
Path translation is applied to update cwd and tool path references.

Safety features:
- Native sessions (UUIDv4) require force flag
- Backup always created before source deletion (rollback capability)
- Two-phase approach: write to target first, then delete source
- Fail-fast pre-check prevents partial writes
- Running sessions require explicit terminate flag

What moves (project-specific artifacts):
- Main JSONL, agent JSONLs, tool results, session-memory

What stays (global, session-ID-keyed):
- Plans, todos, tasks, session-env, debug logs
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from src.exceptions import NativeSessionMoveError, SameProjectMoveError
from src.paths import encode_path
from src.protocols import LoggerProtocol
from src.schemas.operations.discovery import SessionInfo
from src.schemas.operations.move import MoveResult
from src.schemas.session import SessionRecord
from src.services.archive import SessionArchiveService
from src.services.artifacts import (
    AgentFileInfo,
    collect_agent_file_info,
    collect_session_memory,
    collect_tool_results,
    detect_agent_structure,
    extract_custom_title_from_records,
    extract_source_project_path,
    write_jsonl,
    write_session_memory,
    write_tool_results,
)
from src.services.delete import is_native_session
from src.services.discovery import SessionDiscoveryService
from src.services.parser import SessionParserService
from src.services.restore import PathTranslator

logger = logging.getLogger(__name__)

# Backup location (same as delete service)
DELETED_SESSIONS_DIR = Path.home() / '.claude-session-mcp' / 'deleted'


class SessionMoveService:
    """
    Service for relocating sessions between projects.

    Preserves the original session ID, agent IDs, and slugs.
    Only project-specific artifacts are relocated; global artifacts
    (plans, todos, tasks) stay in place since they're keyed by session ID.
    """

    def __init__(
        self,
        target_project_path: Path,
        discovery_service: SessionDiscoveryService | None = None,
        parser_service: SessionParserService | None = None,
    ) -> None:
        self.target_project_path = target_project_path.resolve()
        self.discovery_service = discovery_service or SessionDiscoveryService()
        self.parser_service = parser_service or SessionParserService()
        self.claude_sessions_dir = Path.home() / '.claude' / 'projects'

    async def move_session(
        self,
        session_id: str,
        force: bool = False,
        no_backup: bool = False,
        dry_run: bool = False,
        terminate_pid: int | None = None,
        log: LoggerProtocol | None = None,
    ) -> MoveResult:
        """
        Move a session from its current project to the target project.

        Args:
            session_id: Session ID to move (full UUID or prefix)
            force: Required to move native (UUIDv4) sessions
            no_backup: Don't keep backup after successful move
            dry_run: Preview what would happen without making changes
            terminate_pid: PID to SIGKILL before deleting source
            log: Optional logger

        Returns:
            MoveResult with details of the operation

        Raises:
            FileNotFoundError: If session not found
            SameProjectMoveError: If session is already in target project
            NativeSessionMoveError: If native session and force=False
            FileExistsError: If target files already exist
        """
        start_time = time.monotonic()
        warnings: list[str] = []

        # =====================================================================
        # Phase 1: Resolve & Collect
        # =====================================================================
        session_info = await self._resolve_session(session_id, log)
        source_session_dir = session_info.session_folder

        if log:
            await log.info(f'Moving session: {session_info.session_id}')

        # Discover session files (main + agents)
        session_files, agent_structure = await self._discover_session_files(session_info, log)

        # Load all records
        files_data = await self.parser_service.load_session_files(session_files, log)

        # Determine source project path from record cwd fields
        source_project_path = extract_source_project_path(files_data)

        # Validate: not already in target project
        if source_project_path.resolve() == self.target_project_path:
            raise SameProjectMoveError(session_info.session_id, str(self.target_project_path))

        # Native session check
        if is_native_session(session_info.session_id) and not force:
            raise NativeSessionMoveError(session_info.session_id)

        if log:
            await log.info(f'Source project: {source_project_path}')
            await log.info(f'Target project: {self.target_project_path}')

        # Collect agent file info
        agent_infos = collect_agent_file_info(files_data, agent_structure)

        # Collect tool results
        tool_results = collect_tool_results(source_session_dir, session_info.session_id)

        # Collect session memory
        session_memory = collect_session_memory(source_session_dir, session_info.session_id)

        # Extract custom title
        custom_title = extract_custom_title_from_records(files_data)

        if log:
            await log.info(f'Found {len(agent_infos)} agent files, {len(tool_results)} tool results')
            if session_memory:
                await log.info('Found session-memory/summary.md')

        # =====================================================================
        # Phase 2: Compute Target Paths & Pre-check
        # =====================================================================
        target_dir = self._get_target_directory()
        sid = session_info.session_id

        all_output_paths: list[Path] = []

        # Main session file
        all_output_paths.append(target_dir / f'{sid}.jsonl')

        # Agent files (preserving flat/nested structure)
        for info in agent_infos:
            filename = f'agent-{info.agent_id}.jsonl'
            if info.nested:
                all_output_paths.append(target_dir / sid / 'subagents' / filename)
            else:
                all_output_paths.append(target_dir / filename)

        # Tool results
        if tool_results:
            tool_results_dir = target_dir / sid / 'tool-results'
            all_output_paths.extend(tool_results_dir / tr.filename for tr in tool_results)

        # Session memory
        if session_memory:
            all_output_paths.append(target_dir / sid / 'session-memory' / 'summary.md')

        # Fail-fast: check no output path already exists
        existing_files = [p for p in all_output_paths if p.exists()]
        if existing_files:
            existing_list = '\n  '.join(str(p) for p in existing_files[:5])
            more = f'\n  ... and {len(existing_files) - 5} more' if len(existing_files) > 5 else ''
            raise FileExistsError(
                f'Cannot move: {len(existing_files)} file(s) already exist at target:\n  {existing_list}{more}'
            )

        if log:
            await log.info(f'Verified {len(all_output_paths)} target paths are available')

        # =====================================================================
        # Dry-run exit point
        # =====================================================================
        if dry_run:
            duration_ms = (time.monotonic() - start_time) * 1000
            return MoveResult(
                session_id=sid,
                source_project=str(source_project_path),
                target_project=str(self.target_project_path),
                files_moved=len(all_output_paths),
                files_deleted=0,
                paths_translated=source_project_path.resolve() != self.target_project_path,
                was_running=terminate_pid is not None,
                was_terminated=False,
                backup_path=None,
                custom_title=custom_title,
                resume_command=f'cd {self.target_project_path} && claude --resume {sid}',
                was_dry_run=True,
                duration_ms=duration_ms,
                moved_at=datetime.now(UTC),
                warnings=warnings,
            )

        # =====================================================================
        # Phase 3: Write to Target (source still intact = implicit backup)
        # =====================================================================
        translator = PathTranslator(str(source_project_path), str(self.target_project_path))

        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)

        # Write main session JSONL (path translation, no ID remapping)
        main_filename = f'{sid}.jsonl'
        main_records = files_data[main_filename]
        translated_main = self._translate_records(main_records, translator)
        write_jsonl(target_dir / main_filename, translated_main, {}, {})

        if log:
            await log.info(f'Wrote main session: {len(translated_main)} records')

        # Write agent files (preserving structure)
        for info in agent_infos:
            records = files_data[info.filename]
            filename = f'agent-{info.agent_id}.jsonl'

            if info.nested:
                output_path = target_dir / sid / 'subagents' / filename
            else:
                output_path = target_dir / filename

            output_path.parent.mkdir(parents=True, exist_ok=True)
            translated = self._translate_records(records, translator)
            write_jsonl(output_path, translated, {}, {})

            if log:
                await log.info(f'Wrote agent {filename}: {len(translated)} records')

        # Write tool results (raw copy, no translation)
        if tool_results:
            write_tool_results(tool_results, target_dir, sid)
            if log:
                await log.info(f'Wrote {len(tool_results)} tool result files')

        # Write session memory (raw copy)
        if session_memory:
            write_session_memory(target_dir, sid, session_memory)
            if log:
                await log.info('Wrote session-memory/summary.md')

        files_written = len(all_output_paths)

        if log:
            await log.info(f'Phase 3 complete: {files_written} files written to target')

        # =====================================================================
        # Phase 4: Terminate & Delete Source
        # =====================================================================

        # Create backup before deleting source
        backup_path = await self._create_backup(session_info, files_data, agent_infos, log)

        # Terminate running process if needed
        was_terminated = False
        if terminate_pid is not None:
            if log:
                await log.info(f'Terminating PID {terminate_pid}')
            os.kill(terminate_pid, signal.SIGKILL)
            time.sleep(0.2)
            was_terminated = True

        # Delete source files
        files_deleted = 0
        source_dirs_to_cleanup: list[Path] = []

        try:
            # Delete main session file
            main_source = source_session_dir / f'{sid}.jsonl'
            main_source.unlink()
            files_deleted += 1

            # Delete agent files
            for info in agent_infos:
                if info.nested:
                    agent_path = source_session_dir / sid / 'subagents' / f'agent-{info.agent_id}.jsonl'
                else:
                    agent_path = source_session_dir / f'agent-{info.agent_id}.jsonl'
                agent_path.unlink()
                files_deleted += 1

            # Delete tool results
            for tr in tool_results:
                tr_path = source_session_dir / sid / 'tool-results' / tr.filename
                tr_path.unlink()
                files_deleted += 1

            # Delete session memory
            if session_memory:
                sm_path = source_session_dir / sid / 'session-memory' / 'summary.md'
                sm_path.unlink()
                files_deleted += 1
                source_dirs_to_cleanup.append(source_session_dir / sid / 'session-memory')

            # Clean up empty directories (deepest first)
            if tool_results:
                source_dirs_to_cleanup.append(source_session_dir / sid / 'tool-results')

            # Check for nested agent subagents directory
            subagents_dir = source_session_dir / sid / 'subagents'
            if subagents_dir.exists() and not any(subagents_dir.iterdir()):
                source_dirs_to_cleanup.append(subagents_dir)

            # Session ID subdirectory itself
            source_dirs_to_cleanup.append(source_session_dir / sid)

            for dir_path in source_dirs_to_cleanup:
                if dir_path.exists() and not any(dir_path.iterdir()):
                    dir_path.rmdir()

        except Exception as e:
            msg = f'Source deletion partially failed: {e}. Session exists in both projects. Backup at: {backup_path}'
            warnings.append(msg)
            if log:
                await log.error(msg)

        if log:
            await log.info(f'Phase 4 complete: {files_deleted} source files deleted')

        # Clean up backup if requested
        final_backup_path: str | None = str(backup_path)
        if no_backup and not warnings:
            backup_path.unlink(missing_ok=True)
            final_backup_path = None
            if log:
                await log.info('Backup removed (--no-backup)')

        # =====================================================================
        # Phase 5: Result
        # =====================================================================
        duration_ms = (time.monotonic() - start_time) * 1000

        return MoveResult(
            session_id=sid,
            source_project=str(source_project_path),
            target_project=str(self.target_project_path),
            files_moved=files_written,
            files_deleted=files_deleted,
            paths_translated=translator.needs_translation,
            was_running=terminate_pid is not None,
            was_terminated=was_terminated,
            backup_path=final_backup_path,
            custom_title=custom_title,
            resume_command=f'cd {self.target_project_path} && claude --resume {sid}',
            was_dry_run=False,
            duration_ms=duration_ms,
            moved_at=datetime.now(UTC),
            warnings=list(warnings),
        )

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _get_target_directory(self) -> Path:
        """Compute the target project directory under ~/.claude/projects/."""
        encoded = encode_path(self.target_project_path)
        return self.claude_sessions_dir / encoded

    async def _resolve_session(
        self,
        session_id_or_prefix: str,
        log: LoggerProtocol | None,
    ) -> SessionInfo:
        """Resolve a session ID or prefix to a full session."""
        match = await self.discovery_service.find_session_by_id(session_id_or_prefix)
        if not match:
            raise FileNotFoundError(f'No session found matching: {session_id_or_prefix}')
        return match

    async def _discover_session_files(
        self,
        session_info: SessionInfo,
        log: LoggerProtocol | None,
    ) -> tuple[list[Path], dict[str, bool]]:
        """Discover all JSONL files for a session with structure detection.

        Returns:
            Tuple of (file_paths, agent_structure_map)
            agent_structure_map: filename -> is_nested
        """
        session_dir = session_info.session_folder

        if not session_dir.exists():
            raise FileNotFoundError(f'Session directory not found: {session_dir}')

        main_file = session_dir / f'{session_info.session_id}.jsonl'
        if not main_file.exists():
            raise FileNotFoundError(f'Main session file not found: {main_file}')

        session_files = [main_file]
        agent_structure: dict[str, bool] = {}

        # Find agent files belonging to this session (both flat and nested)
        result = subprocess.run(
            [
                'rg',
                '--files-with-matches',
                f'"sessionId":\\s*"{session_info.session_id}"',
                '--glob',
                '**/agent-*.jsonl',
                str(session_dir),
            ],
            capture_output=True,
            text=True,
        )

        if result.stdout.strip():
            agent_files = [Path(line) for line in result.stdout.strip().split('\n')]
            for agent_path in agent_files:
                is_nested = detect_agent_structure(agent_path, session_info.session_id, session_dir)
                agent_structure[agent_path.name] = is_nested
            session_files.extend(agent_files)

        if log:
            await log.info(f'Found {len(session_files)} session files ({len(agent_structure)} agents)')

        return session_files, agent_structure

    def _translate_records(
        self,
        records: Sequence[SessionRecord],
        translator: PathTranslator,
    ) -> list[SessionRecord]:
        """Translate paths in records without changing session ID or other identifiers."""
        return [translator.translate_record(r) for r in records]

    async def _create_backup(
        self,
        session_info: SessionInfo,
        files_data: Mapping[str, list[SessionRecord]],
        agent_infos: Sequence[AgentFileInfo],
        log: LoggerProtocol | None,
    ) -> Path:
        """Create a backup archive of the session before deleting source."""
        DELETED_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix='claude-session-move-') as temp_dir:
            archive_service = SessionArchiveService(
                session_id=session_info.session_id,
                temp_dir=Path(temp_dir),
                parser_service=self.parser_service,
                session_folder=session_info.session_folder,
            )

            from src.storage.local import LocalFileSystemStorage

            storage = LocalFileSystemStorage(DELETED_SESSIONS_DIR)
            metadata = await archive_service.create_archive(
                storage=storage,
                output_path=None,
                format_param='json',
                logger=log,
            )

            backup_path = Path(metadata.file_path)
            if log:
                await log.info(f'Backup created: {backup_path}')

            return backup_path
