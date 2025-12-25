"""
Session restore service - restores archived sessions with new IDs.

Similar to Claude's teleport feature, but creates new session IDs and handles
path translation for cross-machine restoration.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import uuid6
import zstandard

from src.base_model import StrictModel
from src.introspection import get_path_fields
from src.models import SessionRecord, SessionRecordAdapter
from src.services.archive import LoggerProtocol, SessionArchive
from src.services.artifacts import (
    TODOS_DIR,
    apply_agent_id_mapping,
    apply_slug_mapping,
    create_session_env_dir,
    extract_agent_ids_from_files,
    generate_agent_id_mapping,
    generate_clone_slug,
    transform_agent_filename,
    transform_todo_filename,
    write_todos,
    write_tool_results,
)

# ==============================================================================
# Path Translation Service
# ==============================================================================


class PathTranslator:
    """Handles path translation when restoring sessions across machines."""

    def __init__(self, from_path: str, to_path: str) -> None:
        """
        Initialize path translator.

        Args:
            from_path: Original project path from archive
            to_path: Current project path for restoration
        """
        self.from_path = Path(from_path).resolve()
        self.to_path = Path(to_path).resolve()
        self.needs_translation = self.from_path != self.to_path

    def translate_record(self, record: SessionRecord) -> SessionRecord:
        """
        Translate paths in a record if needed.

        Args:
            record: Session record to translate

        Returns:
            Record with translated paths
        """
        if not self.needs_translation:
            return record

        # Get model class and find path fields
        model_class = type(record)
        path_fields = get_path_fields(model_class)

        # Create dict for updating
        record_dict = record.model_dump(exclude_unset=True, mode='json')

        # Translate each path field
        for field_name in path_fields:
            if field_name in record_dict:
                value = record_dict[field_name]
                if isinstance(value, str):
                    record_dict[field_name] = self._translate_path(value)
                elif isinstance(value, list):
                    record_dict[field_name] = [self._translate_path(p) if isinstance(p, str) else p for p in value]

        # Reconstruct record with translated paths
        return model_class(**record_dict)

    def _translate_path(self, path: str) -> str:
        """Translate a single path string."""
        try:
            old_path = Path(path).resolve()
            if old_path.is_relative_to(self.from_path):
                relative = old_path.relative_to(self.from_path)
                return str(self.to_path / relative)
        except (ValueError, OSError):
            pass  # Path doesn't need translation
        return path


# ==============================================================================
# Restore Result
# ==============================================================================


class RestoreResult(StrictModel):
    """Result of a session restore operation."""

    new_session_id: str
    original_session_id: str
    restored_at: datetime
    project_path: str
    files_restored: int
    records_restored: int
    paths_translated: bool
    main_session_file: str
    agent_files: list[str]


# ==============================================================================
# Utility Functions
# ==============================================================================


def is_restored_session(session_id: str) -> bool:
    """
    Check if a session ID is from a restored session.

    Restored sessions use UUIDv7 (time-ordered) while Claude uses UUIDv4 (random).

    Args:
        session_id: Session ID to check

    Returns:
        True if this is a restored session (UUIDv7), False otherwise
    """
    uid = uuid.UUID(session_id)
    return uid.version == 7


def get_restoration_timestamp(session_id: str) -> datetime | None:
    """
    Extract the restoration timestamp from a restored session ID.

    UUIDv7 embeds a Unix timestamp in the first 48 bits.

    Args:
        session_id: Session ID (must be UUIDv7)

    Returns:
        Restoration datetime or None if not a UUIDv7
    """
    uid = uuid.UUID(session_id)

    if uid.version != 7:
        return None

    # Extract timestamp from first 48 bits (in milliseconds)
    timestamp_ms = int.from_bytes(uid.bytes[:6], 'big')
    timestamp_s = timestamp_ms / 1000
    return datetime.fromtimestamp(timestamp_s, tz=UTC)


# ==============================================================================
# Session Restore Service
# ==============================================================================


class SessionRestoreService:
    """
    Service for restoring archived sessions.

    Creates new session IDs (like Claude's teleport) and handles path translation.
    """

    def __init__(self, project_path: Path) -> None:
        """
        Initialize restore service.

        Args:
            project_path: Current project directory for restoration
        """
        self.project_path = project_path.resolve()
        self.claude_sessions_dir = Path.home() / '.claude' / 'projects'

    async def restore_archive(
        self,
        archive_path: str,
        translate_paths: bool = True,
        in_place: bool = False,
        logger: LoggerProtocol | None = None,
    ) -> RestoreResult:
        """
        Restore a session archive.

        Args:
            archive_path: Path to archive file (JSON or .zst)
            translate_paths: Whether to translate paths to current project
            in_place: If True, restore to original paths with original IDs (undo delete).
                      Fails if any artifact already exists.
            logger: Optional logger instance

        Returns:
            RestoreResult with details of restoration

        Raises:
            FileNotFoundError: If archive doesn't exist
            FileExistsError: If any output file already exists (checked before writing)
            ValueError: If archive format is invalid
        """
        # Load archive
        archive_file = Path(archive_path)
        if not archive_file.exists():
            raise FileNotFoundError(f'Archive not found: {archive_path}')

        if logger:
            await logger.info(f'Loading archive: {archive_path}')
            if in_place:
                await logger.info('In-place mode: restoring to original paths with original IDs')

        # Detect format and load
        if archive_path.endswith('.zst'):
            archive = await self._load_zst_archive(archive_file, logger)
        else:
            archive = await self._load_json_archive(archive_file, logger)

        # Path translation is independent of in_place mode - paths should be
        # translated if they differ, regardless of whether IDs are preserved
        translator = None
        if translate_paths and archive.original_project_path != str(self.project_path):
            translator = PathTranslator(archive.original_project_path, str(self.project_path))
            if logger:
                await logger.info(f'Path translation: {archive.original_project_path} -> {self.project_path}')

        # Determine session ID and mappings based on mode
        if in_place:
            # In-place mode: preserve original IDs (for undo delete)
            # Note: paths are still translated if they differ (handled above)
            new_session_id = archive.session_id
            slug_mapping: Mapping[str, str] = {}
            agent_id_mapping: Mapping[str, str] = {}
            if logger:
                await logger.info(f'Using original session ID: {new_session_id}')
        else:
            # Normal mode: generate new IDs for forking
            new_session_id = str(uuid6.uuid7())
            if logger:
                await logger.info(f'Generated new session ID (UUIDv7): {new_session_id}')
                await logger.info(f'Original session ID: {archive.session_id}')

            # Generate agent ID mapping for fork safety
            agent_ids = extract_agent_ids_from_files(archive.files)
            agent_id_mapping = generate_agent_id_mapping(agent_ids, new_session_id)
            if logger and agent_id_mapping:
                await logger.info(f'Generated {len(agent_id_mapping)} agent ID mappings')
                for old_id, new_id in agent_id_mapping.items():
                    await logger.info(f'  {old_id} -> {new_id}')

            # Pre-compute slug mapping (but don't write yet!)
            slug_mapping = {}
            if archive.plan_files:
                # Generate new slug names without writing
                slug_mapping = {
                    old_slug: generate_clone_slug(old_slug, new_session_id)
                    for old_slug in archive.plan_files
                }

        # Get target directory and plans directory
        target_dir = self._get_session_directory()
        plans_dir = Path.home() / '.claude' / 'plans'

        # =========================================================================
        # FAIL-FAST: Pre-compute ALL output paths and check for existence
        # This must happen BEFORE writing anything to avoid partial restores
        # =========================================================================
        all_output_paths: list[Path] = []

        # 1. Session files (main + agents)
        for filename in archive.files:
            if filename == f'{archive.session_id}.jsonl':
                all_output_paths.append(target_dir / f'{new_session_id}.jsonl')
            elif filename.startswith('agent-'):
                if in_place:
                    all_output_paths.append(target_dir / filename)
                else:
                    new_filename = transform_agent_filename(filename, agent_id_mapping)
                    all_output_paths.append(target_dir / new_filename)
            else:
                all_output_paths.append(target_dir / filename)

        # 2. Tool results
        if archive.tool_results:
            tool_results_dir = target_dir / new_session_id / 'tool-results'
            for tool_use_id in archive.tool_results:
                all_output_paths.append(tool_results_dir / f'{tool_use_id}.txt')

        # 3. Todos
        if archive.todos:
            for old_filename in archive.todos:
                if in_place:
                    all_output_paths.append(TODOS_DIR / old_filename)
                else:
                    new_filename = transform_todo_filename(old_filename, archive.session_id, new_session_id)
                    all_output_paths.append(TODOS_DIR / new_filename)

        # 4. Plan files (non-in-place mode only)
        if not in_place and archive.plan_files:
            for new_slug in slug_mapping.values():
                all_output_paths.append(plans_dir / f'{new_slug}.md')

        # Check ALL paths before writing ANYTHING
        existing_files = [p for p in all_output_paths if p.exists()]
        if existing_files:
            existing_list = '\n  '.join(str(p) for p in existing_files[:5])
            more = f'\n  ... and {len(existing_files) - 5} more' if len(existing_files) > 5 else ''
            raise FileExistsError(
                f'Cannot restore: {len(existing_files)} file(s) already exist:\n  {existing_list}{more}\n'
                'Use a different project or delete the existing session first.'
            )

        # =========================================================================
        # Now safe to write - no files will be overwritten
        # =========================================================================

        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)
        if logger:
            await logger.info(f'Target directory: {target_dir}')

        # Write plan files (non-in-place mode only)
        if not in_place and archive.plan_files:
            plans_dir.mkdir(parents=True, exist_ok=True)
            for old_slug, content in archive.plan_files.items():
                new_slug = slug_mapping[old_slug]
                plan_path = plans_dir / f'{new_slug}.md'
                plan_path.write_text(content, encoding='utf-8')
            if logger:
                await logger.info(f'Restored {len(slug_mapping)} plan files')
                for old_slug, new_slug in slug_mapping.items():
                    await logger.info(f'  {old_slug} -> {new_slug}')

        # Write tool results (v1.2+ archives)
        if archive.tool_results:
            tool_results_count = write_tool_results(archive.tool_results, target_dir, new_session_id)
            if logger:
                await logger.info(f'Restored {tool_results_count} tool result files')

        # Write todos (v1.2+ archives)
        if archive.todos:
            if in_place:
                TODOS_DIR.mkdir(parents=True, exist_ok=True)
                for filename, content in archive.todos.items():
                    (TODOS_DIR / filename).write_text(content, encoding='utf-8')
                if logger:
                    await logger.info(f'Restored {len(archive.todos)} todo files (in-place)')
            else:
                todos_mapping = write_todos(archive.todos, archive.session_id, new_session_id)
                if logger:
                    await logger.info(f'Restored {len(todos_mapping)} todo files')

        # Create session-env directory
        create_session_env_dir(new_session_id)

        # Restore session files
        main_file_path = None
        agent_files = []
        total_records = 0

        for filename, records in archive.files.items():
            # Determine new filename
            if filename == f'{archive.session_id}.jsonl':
                new_filename = f'{new_session_id}.jsonl'
                main_file_path = str(target_dir / new_filename)
            elif filename.startswith('agent-'):
                if in_place:
                    new_filename = filename
                else:
                    new_filename = transform_agent_filename(filename, agent_id_mapping)
                agent_files.append(str(target_dir / new_filename))
            else:
                new_filename = filename

            # Update session IDs and translate paths in records
            updated_records = []
            for record in records:
                record_dict = record.model_dump(exclude_unset=True, mode='json')

                # Update session ID if present (even in in_place mode, records need correct ID)
                if 'sessionId' in record_dict:
                    record_dict['sessionId'] = new_session_id

                # Validate and reconstruct record
                updated_record = SessionRecordAdapter.validate_python(record_dict)

                # Translate paths if needed
                if translator:
                    updated_record = translator.translate_record(updated_record)

                updated_records.append(updated_record)

            # Write JSONL file
            output_path = target_dir / new_filename
            await self._write_jsonl(output_path, updated_records, slug_mapping, agent_id_mapping, logger)
            total_records += len(updated_records)

            if logger:
                await logger.info(f'Restored {new_filename}: {len(updated_records)} records')

        return RestoreResult(
            new_session_id=new_session_id,
            original_session_id=archive.session_id,
            restored_at=datetime.now(UTC),
            project_path=str(self.project_path),
            files_restored=len(archive.files),
            records_restored=total_records,
            paths_translated=translator is not None,
            main_session_file=main_file_path or '',
            agent_files=agent_files,
        )

    async def _load_json_archive(self, archive_file: Path, logger: LoggerProtocol | None) -> SessionArchive:
        """Load archive from JSON file."""
        with open(archive_file, encoding='utf-8') as f:
            data = json.load(f)

        # Convert records from dicts to SessionRecord objects
        for filename, records in data['files'].items():
            data['files'][filename] = [SessionRecordAdapter.validate_python(r) for r in records]

        return SessionArchive(**data)

    async def _load_zst_archive(self, archive_file: Path, logger: LoggerProtocol | None) -> SessionArchive:
        """Load archive from Zstandard compressed file."""
        dctx = zstandard.ZstdDecompressor()
        with open(archive_file, 'rb') as f:
            decompressed = dctx.decompress(f.read())

        data = json.loads(decompressed)

        # Convert records from dicts to SessionRecord objects
        for filename, records in data['files'].items():
            data['files'][filename] = [SessionRecordAdapter.validate_python(r) for r in records]

        return SessionArchive(**data)

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
        """Get the session directory for the current project."""
        # Encode project path for Claude's directory structure
        encoded = self._encode_path(self.project_path)
        return self.claude_sessions_dir / encoded

    def _encode_path(self, path: Path) -> str:
        """
        Encode path for Claude's directory naming.

        Claude Code encodes paths by replacing special characters with hyphens:
        - Forward slashes (/) → hyphens (-)
        - Periods (.) → hyphens (-)
        - Spaces ( ) → hyphens (-)
        - Tildes (~) → hyphens (-)

        Example: /Users/user/Library/Mobile Documents/com~apple~CloudDocs/project
              → -Users-user-Library-Mobile-Documents-com-apple-CloudDocs-project
        """
        result = str(path)
        result = result.replace('/', '-')
        result = result.replace('.', '-')
        result = result.replace(' ', '-')
        result = result.replace('~', '-')
        return result
