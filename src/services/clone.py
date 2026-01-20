"""
Session clone service - direct session-to-session cloning.

Clones a session directly without creating an intermediate archive file.
Faster than archive+restore for local cloning operations.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import uuid6

from src.paths import encode_path
from src.protocols import LoggerProtocol
from src.schemas.operations.discovery import SessionInfo
from src.schemas.operations.restore import RestoreResult
from src.schemas.session import CustomTitleRecord, SessionRecord, SessionRecordAdapter
from src.services.artifacts import (
    TODOS_DIR,
    apply_agent_id_mapping,
    apply_slug_mapping,
    collect_plan_files,
    collect_todos,
    collect_tool_results,
    create_session_env_dir,
    extract_agent_ids_from_files,
    extract_custom_title_from_records,
    extract_slugs_from_records,
    extract_source_project_path,
    generate_agent_id_mapping,
    generate_clone_custom_title,
    generate_clone_slug,
    transform_agent_filename,
    transform_todo_filename,
    validate_session_env_empty,
    write_todos,
    write_tool_results,
)
from src.services.discovery import SessionDiscoveryService
from src.services.lineage import LineageService
from src.services.parser import SessionParserService
from src.services.restore import PathTranslator


class SessionCloneService:
    """
    Service for direct session-to-session cloning.

    Reads source session JSONL files directly, transforms them
    (new session ID, path translation), and writes to target location.
    No intermediate archive file is created.
    """

    def __init__(
        self,
        target_project_path: Path,
        discovery_service: SessionDiscoveryService | None = None,
        parser_service: SessionParserService | None = None,
    ) -> None:
        """
        Initialize clone service.

        Args:
            target_project_path: Target project directory for cloned session
            discovery_service: Optional discovery service (creates one if not provided)
            parser_service: Optional parser service (creates one if not provided)
        """
        self.target_project_path = target_project_path.resolve()
        self.discovery_service = discovery_service or SessionDiscoveryService()
        self.parser_service = parser_service or SessionParserService()
        self.claude_sessions_dir = Path.home() / '.claude' / 'projects'

    async def clone(
        self,
        source_session_id: str,
        translate_paths: bool = True,
        logger: LoggerProtocol | None = None,
    ) -> RestoreResult:
        """
        Clone a session directly without creating an archive file.

        Args:
            source_session_id: Session ID to clone (full UUID or prefix)
            translate_paths: Whether to translate paths to target project
            logger: Optional logger instance

        Returns:
            RestoreResult with new session ID and details

        Raises:
            FileNotFoundError: If source session not found
            FileExistsError: If any output file already exists (checked before writing)
            AmbiguousSessionError: If session ID prefix matches multiple sessions
        """
        # Resolve session ID (handle prefix matching)
        session_info = await self._resolve_session(source_session_id, logger)

        if logger:
            await logger.info(f'Cloning session: {session_info.session_id}')
            await logger.info(f'Source session folder: {session_info.session_folder}')

        # Source session directory (we have it directly from discovery, no encoding needed)
        source_session_dir = session_info.session_folder

        # Discover all session files (main + agents)
        session_files = await self._discover_session_files(session_info, logger)

        if logger:
            await logger.info(f'Found {len(session_files)} session files')

        # Load all records
        files_data = await self.parser_service.load_session_files(session_files, logger)

        # Collect tool results from source session
        tool_results = collect_tool_results(source_session_dir, session_info.session_id)
        if logger:
            await logger.info(f'Found {len(tool_results)} tool result files')

        # Collect todos from source session
        todos = collect_todos(session_info.session_id)
        if logger:
            await logger.info(f'Found {len(todos)} todo files')

        # Validate session-env is empty (fail fast if Claude starts using it)
        validate_session_env_empty(session_info.session_id)

        # Extract slugs and collect plan files from source session
        slugs = extract_slugs_from_records(files_data)
        plan_files = collect_plan_files(slugs)
        if logger:
            await logger.info(f'Found {len(slugs)} slugs, {len(plan_files)} plan files')

        # Extract custom title from source session
        source_custom_title = extract_custom_title_from_records(files_data)
        if logger and source_custom_title:
            await logger.info(f'Found custom title: {source_custom_title}')

        # Generate new session ID (UUIDv7 for identification)
        new_session_id = str(uuid6.uuid7())
        if logger:
            await logger.info(f'Generated new session ID (UUIDv7): {new_session_id}')
            await logger.info(f'Original session ID: {session_info.session_id}')

        # Pre-compute slug mapping (but don't write yet - fail-fast check first!)
        slug_mapping: Mapping[str, str] = {}
        if plan_files:
            slug_mapping = {old_slug: generate_clone_slug(old_slug, new_session_id) for old_slug in plan_files}

        # Generate agent ID mapping (CRITICAL for same-project forking)
        agent_ids = extract_agent_ids_from_files(files_data)
        agent_id_mapping = generate_agent_id_mapping(agent_ids, new_session_id)
        if logger and agent_id_mapping:
            await logger.info(f'Generated {len(agent_id_mapping)} agent ID mappings')
            for old_id, new_id in agent_id_mapping.items():
                await logger.info(f'  {old_id} -> {new_id}')

        # Create path translator if needed
        # Note: We use the actual cwd from records, not the decoded path from discovery,
        # because the path encoding is lossy (multiple chars → '-') and decoding is unreliable.
        source_path = extract_source_project_path(files_data)
        translator = None
        if translate_paths and source_path != self.target_project_path:
            translator = PathTranslator(str(source_path), str(self.target_project_path))
            if logger:
                await logger.info(f'Path translation: {source_path} -> {self.target_project_path}')

        # Get target directory and plans directory
        target_dir = self._get_session_directory()
        plans_dir = Path.home() / '.claude' / 'plans'

        # =========================================================================
        # FAIL-FAST: Pre-compute ALL output paths and check for existence
        # This must happen BEFORE writing anything to avoid partial clones
        # =========================================================================
        all_output_paths: list[Path] = []

        # 1. Session files (main + agents with transformed names)
        for filename in files_data:
            if filename == f'{session_info.session_id}.jsonl':
                all_output_paths.append(target_dir / f'{new_session_id}.jsonl')
            elif filename.startswith('agent-'):
                new_filename = transform_agent_filename(filename, agent_id_mapping)
                all_output_paths.append(target_dir / new_filename)
            else:
                all_output_paths.append(target_dir / filename)

        # 2. Tool results
        if tool_results:
            tool_results_dir = target_dir / new_session_id / 'tool-results'
            all_output_paths.extend(tool_results_dir / f'{tool_use_id}.txt' for tool_use_id in tool_results)

        # 3. Todos
        if todos:
            for old_filename in todos:
                new_filename = transform_todo_filename(old_filename, session_info.session_id, new_session_id)
                all_output_paths.append(TODOS_DIR / new_filename)

        # 4. Plan files
        if plan_files:
            all_output_paths.extend(plans_dir / f'{new_slug}.md' for new_slug in slug_mapping.values())

        # Check ALL paths before writing ANYTHING
        existing_files = [p for p in all_output_paths if p.exists()]
        if existing_files:
            existing_list = '\n  '.join(str(p) for p in existing_files[:5])
            more = f'\n  ... and {len(existing_files) - 5} more' if len(existing_files) > 5 else ''
            raise FileExistsError(
                f'Cannot clone: {len(existing_files)} file(s) already exist:\n  {existing_list}{more}\n'
                'This may indicate cloning into an existing session or a previous failed clone.'
            )

        if logger:
            await logger.info(f'Verified {len(all_output_paths)} output paths are available')

        # =========================================================================
        # Now safe to write - no files will be overwritten
        # =========================================================================

        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)
        if logger:
            await logger.info(f'Target directory: {target_dir}')

        # Write plan files
        if plan_files:
            plans_dir.mkdir(parents=True, exist_ok=True)
            for old_slug, content in plan_files.items():
                new_slug = slug_mapping[old_slug]
                plan_path = plans_dir / f'{new_slug}.md'
                plan_path.write_text(content, encoding='utf-8')
            if logger:
                await logger.info(f'Wrote {len(slug_mapping)} plan files with new slugs')
                for old_slug, new_slug in slug_mapping.items():
                    await logger.info(f'  {old_slug} -> {new_slug}')

        # Write tool results
        if tool_results:
            tool_results_count = write_tool_results(tool_results, target_dir, new_session_id)
            if logger:
                await logger.info(f'Wrote {tool_results_count} tool result files')

        # Write todos with updated filenames
        if todos:
            todos_mapping = write_todos(todos, session_info.session_id, new_session_id)
            if logger:
                await logger.info(f'Wrote {len(todos_mapping)} todo files')

        # Create session-env directory
        create_session_env_dir(new_session_id)

        # Write transformed session files
        main_file_path = None
        agent_files = []
        total_records = 0

        for filename, records in files_data.items():
            # Determine new filename
            if filename == f'{session_info.session_id}.jsonl':
                new_filename = f'{new_session_id}.jsonl'
                main_file_path = str(target_dir / new_filename)
            elif filename.startswith('agent-'):
                # Transform agent filename with new ID for fork safety
                new_filename = transform_agent_filename(filename, agent_id_mapping)
                agent_files.append(str(target_dir / new_filename))
            else:
                new_filename = filename

            # Transform records
            updated_records: list[SessionRecord] = []
            for record in records:
                # Transform CustomTitleRecord with clone title
                if isinstance(record, CustomTitleRecord):
                    new_title = generate_clone_custom_title(record.customTitle, new_session_id)
                    updated_record: SessionRecord = CustomTitleRecord(
                        type='custom-title',
                        customTitle=new_title,
                        sessionId=new_session_id,
                    )
                else:
                    record_dict = record.model_dump(exclude_unset=True, mode='json')
                    if 'sessionId' in record_dict:
                        record_dict['sessionId'] = new_session_id
                    updated_record = SessionRecordAdapter.validate_python(record_dict)

                # Translate paths if needed
                if translator:
                    updated_record = translator.translate_record(updated_record)

                updated_records.append(updated_record)

            # Write JSONL file (with slug and agent ID mappings applied)
            output_path = target_dir / new_filename
            await self._write_jsonl(output_path, updated_records, slug_mapping, agent_id_mapping, logger)
            total_records += len(updated_records)

            if logger:
                await logger.info(f'Cloned {new_filename}: {len(updated_records)} records')

        # Record lineage (source_path already extracted above)
        lineage_service = LineageService()
        lineage_service.record_clone(
            child_session_id=new_session_id,
            parent_session_id=session_info.session_id,
            cloned_at=datetime.now(UTC),
            parent_project_path=source_path,
            target_project_path=self.target_project_path,
            method='clone',
            parent_machine_id=None,  # Clone is always same machine
            paths_translated=translator is not None,
            archive_path=None,
        )
        if logger:
            await logger.info(f'Recorded lineage: {session_info.session_id} -> {new_session_id}')

        return RestoreResult(
            new_session_id=new_session_id,
            original_session_id=session_info.session_id,
            restored_at=datetime.now(UTC),
            project_path=str(self.target_project_path),
            files_restored=len(files_data),
            records_restored=total_records,
            paths_translated=translator is not None,
            main_session_file=main_file_path or '',
            agent_files=agent_files,
        )

    async def _resolve_session(
        self,
        session_id_or_prefix: str,
        logger: LoggerProtocol | None,
    ) -> SessionInfo:
        """
        Resolve a session ID or prefix to a full session.

        Delegates to SessionDiscoveryService which handles both exact matches
        and prefix matching.

        Args:
            session_id_or_prefix: Full session ID or prefix
            logger: Optional logger

        Returns:
            SessionInfo for the matched session

        Raises:
            FileNotFoundError: If no sessions match
            AmbiguousSessionError: If multiple sessions match (from discovery service)
        """
        match = await self.discovery_service.find_session_by_id(session_id_or_prefix)
        if not match:
            raise FileNotFoundError(f'No session found matching: {session_id_or_prefix}')
        return match

    async def _discover_session_files(
        self,
        session_info: SessionInfo,
        logger: LoggerProtocol | None,
    ) -> list[Path]:
        """Discover all JSONL files for a session (main + agents)."""
        # Use session folder directly from discovery (no encoding needed)
        session_dir = session_info.session_folder

        if not session_dir.exists():
            raise FileNotFoundError(f'Session directory not found: {session_dir}')

        # Find main session file
        main_file = session_dir / f'{session_info.session_id}.jsonl'
        if not main_file.exists():
            raise FileNotFoundError(f'Main session file not found: {main_file}')

        session_files = [main_file]

        # Find agent files belonging to this session
        # Note: JSON may have optional space after colon, so we use regex pattern
        result = subprocess.run(
            [
                'rg',
                '--files-with-matches',
                f'"sessionId":\\s*"{session_info.session_id}"',
                '--glob',
                'agent-*.jsonl',
                str(session_dir),
            ],
            capture_output=True,
            text=True,
        )

        if result.stdout.strip():
            agent_files = [Path(line) for line in result.stdout.strip().split('\n')]
            session_files.extend(agent_files)

        return session_files

    async def _write_jsonl(
        self,
        path: Path,
        records: Sequence[SessionRecord],
        slug_mapping: Mapping[str, str],
        agent_id_mapping: Mapping[str, str],
        logger: LoggerProtocol | None,
    ) -> None:
        """Write records to JSONL file, applying slug and agent ID mappings."""
        with open(path, 'w', encoding='utf-8') as f:
            for record in records:
                # Use exclude_unset for round-trip fidelity
                json_data = record.model_dump(exclude_unset=True, mode='json')
                # Use compact separators for consistent, smaller output
                json_str = json.dumps(json_data, separators=(',', ':'))

                # Apply slug mapping first (longer strings, less collision risk)
                if slug_mapping:
                    json_str = apply_slug_mapping(json_str, slug_mapping)

                # Apply agent ID mapping second
                if agent_id_mapping:
                    json_str = apply_agent_id_mapping(json_str, agent_id_mapping)

                f.write(json_str + '\n')

    def _get_session_directory(self) -> Path:
        """Get the session directory for the target project."""
        encoded = encode_path(self.target_project_path)
        return self.claude_sessions_dir / encoded
