"""
Session restore service - restores archived sessions with new IDs.

Similar to Claude's teleport feature, but creates new session IDs and handles
path translation for cross-machine restoration.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import uuid6
import zstandard

from src.base_model import StrictModel
from src.models import SessionRecord, SessionRecordAdapter
from src.services.archive import SessionArchive, LoggerProtocol
from src.introspection import get_path_fields
from src.types import JsonDatetime


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
    return datetime.fromtimestamp(timestamp_s, tz=timezone.utc)


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
        logger: LoggerProtocol | None = None,
    ) -> RestoreResult:
        """
        Restore a session archive with a new session ID.

        Args:
            archive_path: Path to archive file (JSON or .zst)
            translate_paths: Whether to translate paths to current project
            logger: Optional logger instance

        Returns:
            RestoreResult with details of restoration

        Raises:
            FileNotFoundError: If archive doesn't exist
            ValueError: If archive format is invalid
        """
        # Load archive
        archive_file = Path(archive_path)
        if not archive_file.exists():
            raise FileNotFoundError(f'Archive not found: {archive_path}')

        if logger:
            await logger.info(f'Loading archive: {archive_path}')

        # Detect format and load
        if archive_path.endswith('.zst'):
            archive = await self._load_zst_archive(archive_file, logger)
        else:
            archive = await self._load_json_archive(archive_file, logger)

        # Generate new session ID using UUIDv7
        # UUIDv7 makes restored sessions immediately identifiable:
        # - Claude uses UUIDv4 (random), we use UUIDv7 (time-ordered)
        # - Version nibble: 4 vs 7 makes them visually distinct
        # - Embeds restoration timestamp in the UUID itself
        new_session_id = str(uuid6.uuid7())
        if logger:
            await logger.info(f'Generated new session ID (UUIDv7): {new_session_id}')
            await logger.info(f'Original session ID: {archive.session_id}')

        # Create path translator if needed
        translator = None
        if translate_paths and archive.original_project_path != str(self.project_path):
            translator = PathTranslator(archive.original_project_path, str(self.project_path))
            if logger:
                await logger.info(f'Path translation: {archive.original_project_path} -> {self.project_path}')

        # Create target directory in ~/.claude/projects/
        target_dir = self._get_session_directory()
        target_dir.mkdir(parents=True, exist_ok=True)
        if logger:
            await logger.info(f'Target directory: {target_dir}')

        # Restore files with new session ID
        main_file_path = None
        agent_files = []
        total_records = 0

        for filename, records in archive.files.items():
            # Determine new filename
            if filename == f'{archive.session_id}.jsonl':
                # Main session file - use new session ID
                new_filename = f'{new_session_id}.jsonl'
                main_file_path = str(target_dir / new_filename)
            elif filename.startswith('agent-'):
                # Agent file - preserve agent ID
                new_filename = filename
                agent_files.append(str(target_dir / new_filename))
            else:
                # Other files - preserve name
                new_filename = filename

            # Update session IDs and translate paths in records
            updated_records = []
            for record in records:
                record_dict = record.model_dump(exclude_unset=True, mode='json')

                # Update session ID if present
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
            await self._write_jsonl(output_path, updated_records, logger)
            total_records += len(updated_records)

            if logger:
                await logger.info(f'Restored {new_filename}: {len(updated_records)} records')

        return RestoreResult(
            new_session_id=new_session_id,
            original_session_id=archive.session_id,
            restored_at=datetime.now(timezone.utc),
            project_path=str(self.project_path),
            files_restored=len(archive.files),
            records_restored=total_records,
            paths_translated=translator is not None,
            main_session_file=main_file_path or '',
            agent_files=agent_files,
        )

    async def _load_json_archive(self, archive_file: Path, logger: LoggerProtocol | None) -> SessionArchive:
        """Load archive from JSON file."""
        with open(archive_file, 'r', encoding='utf-8') as f:
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

    async def _write_jsonl(self, path: Path, records: list[SessionRecord], logger: LoggerProtocol | None) -> None:
        """Write records to JSONL file."""
        with open(path, 'w', encoding='utf-8') as f:
            for record in records:
                # Use exclude_unset for round-trip fidelity
                json_data = record.model_dump(exclude_unset=True, mode='json')
                f.write(json.dumps(json_data) + '\n')

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
