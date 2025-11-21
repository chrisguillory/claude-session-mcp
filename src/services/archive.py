"""
Session archive service - framework-agnostic domain logic.

Pure service layer with no MCP/FastAPI dependencies. Handles session discovery,
archive creation, compression, and format detection.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from src.base_model import StrictModel
from src.models import (
    SessionRecord,
    UserRecord,
    AssistantRecord,
    LocalCommandSystemRecord,
    CompactBoundarySystemRecord,
    ApiErrorSystemRecord,
    InformationalSystemRecord,
)
from src.storage.protocol import StorageBackend
from src.config.mcp import settings
from src.services.parser import SessionParserService
from src.types import JsonDatetime


# ==============================================================================
# Logger Protocol (allows service to be framework-agnostic)
# ==============================================================================


class LoggerProtocol(Protocol):
    """Protocol for logger - enables service to work with any logging implementation."""

    async def info(self, message: str) -> None: ...
    async def warning(self, message: str) -> None: ...
    async def error(self, message: str) -> None: ...


# ==============================================================================
# Output Models (MCP Tool Return Types)
# ==============================================================================


class FileMetadata(StrictModel):
    """Metadata about a single JSONL file in the archive."""

    filename: str
    record_count: int


class ArchiveMetadata(StrictModel):
    """
    Metadata about created archive.

    Returned by save_current_session MCP tool.
    """

    file_path: str
    session_id: str
    format: Literal['json', 'zst']
    size_mb: float  # Size in megabytes, rounded to 2 decimal places
    archived_at: datetime
    record_count: int  # Total records across all files
    file_count: int  # Number of JSONL files included (main + agents)
    files: list[FileMetadata]  # Per-file breakdown


# ==============================================================================
# Archive Structure (Serialized to JSON)
# ==============================================================================


class SessionArchive(StrictModel):
    """
    Complete session archive structure (written to JSON).

    This is the JSON file structure that gets saved to disk.
    """

    version: str = '1.0'  # Archive format version
    session_id: str
    archived_at: JsonDatetime
    original_project_path: str
    claude_code_version: str  # Claude Code version at archive time
    files: dict[str, list[SessionRecord]]  # filename -> records


# ==============================================================================
# Archive Format Detection
# ==============================================================================


class FormatDetector:
    """Detects and validates archive format from file extension and format parameter."""

    SUPPORTED_FORMATS = {'json', 'zst'}  # 'zst' = JSON with zstd compression

    # Extension to format mapping
    EXTENSION_MAP = {
        '.json': 'json',
        '.json.zst': 'zst',
        '.zst': 'zst',
    }

    @classmethod
    def detect_format(cls, output_path: Path, format_param: Literal['json', 'zst'] | None) -> Literal['json', 'zst']:
        """
        Detect archive format from path and validate against format parameter.

        Validates that extension and explicit format parameter don't conflict.
        Handles multi-part extensions like '.json.zst'.

        Args:
            output_path: Output file path
            format_param: Optional explicit format parameter

        Returns:
            Detected format string

        Raises:
            ValueError: If format is ambiguous or conflicts
        """
        # Try to detect from extension
        detected_format = cls._detect_from_extension(output_path)

        if detected_format:
            # Format detectable from extension
            if format_param is None:
                # Case 3: Use detected format
                return detected_format
            elif format_param == detected_format:
                # Case 1: Redundant but consistent
                return detected_format
            else:
                # Case 2: Mismatch
                raise ValueError(
                    f"Format mismatch: extension indicates '{detected_format}' but format parameter is '{format_param}'"
                )
        else:
            # Format NOT detectable from extension
            if format_param:
                # Case 4: Use specified format
                if format_param not in cls.SUPPORTED_FORMATS:
                    raise ValueError(
                        f"Unsupported format: '{format_param}'. Supported: {sorted(cls.SUPPORTED_FORMATS)}"
                    )
                return format_param
            else:
                # Case 5: Ambiguous
                raise ValueError(
                    f"Cannot detect format from extension '{output_path.suffix}'. "
                    f'Please specify format parameter. Supported: {sorted(cls.SUPPORTED_FORMATS)}'
                )

    @classmethod
    def _detect_from_extension(cls, output_path: Path) -> Literal['json', 'zst'] | None:
        """
        Detect format from file extension.

        Handles multi-part extensions like '.json.zst'.

        Returns:
            Format string if detectable, None otherwise
        """
        path_str = output_path.name.lower()

        # Try multi-part extensions first (longest match)
        for ext, fmt in sorted(cls.EXTENSION_MAP.items(), key=lambda x: -len(x[0])):
            if path_str.endswith(ext):
                return fmt

        return None


# ==============================================================================
# Session Archive Service
# ==============================================================================

# All record types that have a version field
VERSION_RECORD_TYPES = (
    UserRecord,
    AssistantRecord,
    LocalCommandSystemRecord,
    CompactBoundarySystemRecord,
    ApiErrorSystemRecord,
    InformationalSystemRecord,
)


class SessionArchiveService:
    """
    Service for creating session archives.

    Pure domain logic - no MCP/FastAPI dependencies. Takes session_id and
    discovers all related JSONL files from ~/.claude/projects/.
    """

    def __init__(
        self, session_id: str, project_path: Path, temp_dir: Path, parser_service: SessionParserService
    ) -> None:
        """
        Initialize archive service.

        Args:
            session_id: Current Claude Code session ID
            project_path: Current project directory
            temp_dir: Temporary directory for default output
            parser_service: Session parser service for loading JSONL files
        """
        self.session_id = session_id
        self.project_path = project_path
        self.temp_dir = temp_dir
        self.parser_service = parser_service

        # Claude stores sessions here
        self.claude_sessions_dir = Path.home() / '.claude' / 'projects'

    async def create_archive(
        self,
        storage: StorageBackend,
        output_path: str | None,
        format_param: Literal['json', 'zst'] | None,
        logger: LoggerProtocol,
    ) -> ArchiveMetadata:
        """
        Create archive of current session.

        Args:
            storage: Storage backend (call-time parameter!)
            output_path: Optional output path (None = temp file)
            format_param: Optional format override
            logger: Logger instance

        Returns:
            Archive metadata

        Raises:
            ValueError: If format detection fails or conflicts
            FileNotFoundError: If session files not found
        """
        await logger.info(f'Creating archive for session {self.session_id}')

        # Determine output path
        if output_path:
            output_file = Path(output_path)

            # Check if file already exists
            if output_file.exists():
                raise FileExistsError(
                    f'File already exists: {output_file}\nUse a different filename or delete the existing file first.'
                )

            # Fail fast: validate directory exists
            if not output_file.parent.exists():
                raise ValueError(f'Output directory does not exist: {output_file.parent}. Please create it first.')

            await logger.info(f'Output path: {output_file}')
        else:
            # Use temp directory (default)
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            filename = f'session-{self.session_id[:8]}-{timestamp}.json'
            output_file = self.temp_dir / filename
            await logger.info(f'Using temp file: {output_file}')

        # Detect format
        archive_format = FormatDetector.detect_format(output_file, format_param)
        await logger.info(f'Archive format: {archive_format}')

        # Discover session files
        session_files = await self._discover_session_files(logger)
        await logger.info(f'Found {len(session_files)} session files')

        # Load and parse all records
        files_data, total_records = await self._load_session_files(session_files, logger)
        await logger.info(f'Loaded {total_records} total records')

        # Extract Claude Code version from records
        claude_code_version = await self._extract_claude_code_version(files_data, logger)

        # Create archive structure
        archive = SessionArchive(
            session_id=self.session_id,
            archived_at=datetime.now(timezone.utc),
            original_project_path=str(self.project_path),
            claude_code_version=claude_code_version,
            files=files_data,
        )

        # Serialize and compress
        if archive_format == 'json':
            data = await self._serialize_json(archive, logger)
        elif archive_format == 'zst':
            data = await self._serialize_zst(archive, logger)
        else:
            raise ValueError(f'Unsupported format: {archive_format}')

        # Save via storage backend
        final_path = await storage.save(output_file.name, data)
        await logger.info(f'Archive saved: {len(data):,} bytes')

        # Calculate size in MB (rounded to 2 decimal places)
        size_mb = round(len(data) / (1024 * 1024), 2)

        # Build per-file metadata
        file_metadata = [
            FileMetadata(filename=filename, record_count=len(records)) for filename, records in files_data.items()
        ]

        return ArchiveMetadata(
            file_path=final_path,
            session_id=self.session_id,
            format=archive_format,
            size_mb=size_mb,
            archived_at=datetime.now(timezone.utc),
            record_count=total_records,
            file_count=len(session_files),
            files=file_metadata,
        )

    async def _discover_session_files(self, logger: LoggerProtocol) -> list[Path]:
        """
        Discover all JSONL files for current session.

        Finds:
        - Main session file: {session_id}.jsonl
        - Agent session files: agent-*.jsonl (if any)

        Returns:
            List of paths to JSONL files

        Raises:
            FileNotFoundError: If main session file not found
        """
        await logger.info('Discovering session files')

        # Find project directory in ~/.claude/projects/
        # Pattern: ~/.claude/projects/-path-to-project-{session_id}.jsonl
        project_folders = list(self.claude_sessions_dir.glob('*'))

        # Find folder that matches our project path
        project_folder = None
        encoded_project = self._encode_path(self.project_path)

        await logger.info(f'Looking for encoded project path: {encoded_project}')

        for folder in project_folders:
            if folder.name.startswith(encoded_project):
                project_folder = folder
                break

        if not project_folder:
            raise FileNotFoundError(
                f'Could not find session folder for project {self.project_path} in {self.claude_sessions_dir}'
            )

        await logger.info(f'Found project folder: {project_folder.name}')

        # Find main session file
        main_file = project_folder / f'{self.session_id}.jsonl'
        if not main_file.exists():
            raise FileNotFoundError(f'Main session file not found: {main_file}')

        session_files = [main_file]

        # Find agent files that belong to this session using rg
        result = subprocess.run(
            ['rg', '--files-with-matches', f'"sessionId":"{self.session_id}"', '--glob', 'agent-*.jsonl', str(project_folder)],
            capture_output=True,
            text=True,
            timeout=5,
        )

        agent_files = []
        if result.stdout.strip():
            agent_files = [Path(line) for line in result.stdout.strip().split('\n')]

        session_files.extend(agent_files)

        await logger.info(f'Found {len(session_files)} files: 1 main + {len(agent_files)} agents')

        return session_files

    async def _load_session_files(
        self, session_files: list[Path], logger: LoggerProtocol
    ) -> tuple[dict[str, list[SessionRecord]], int]:
        """
        Load and parse all session files using parser service.

        Args:
            session_files: List of JSONL file paths
            logger: Logger instance

        Returns:
            Tuple of (files_data dict, total_record_count)

        Raises:
            json.JSONDecodeError: If JSONL is invalid (fail fast)
            pydantic.ValidationError: If record validation fails (fail fast)
        """
        # Delegate parsing to parser service
        files_data = await self.parser_service.load_session_files(session_files, logger)

        # Calculate total records
        total_records = sum(len(records) for records in files_data.values())

        return files_data, total_records

    async def _extract_claude_code_version(
        self, files_data: dict[str, list[SessionRecord]], logger: LoggerProtocol
    ) -> str:
        """
        Extract Claude Code version from session records.

        Strategy:
        1. Find first record with a version field (check all types that have version)
        2. Fallback: subprocess call to `claude --version` (if data corrupted)

        Args:
            files_data: Parsed session records
            logger: Logger instance

        Returns:
            Claude Code version string (e.g., '2.0.37')
        """
        # Try to find first record with a version field
        for filename, records in files_data.items():
            for record in records:
                # Single isinstance check for all types with version
                if isinstance(record, VERSION_RECORD_TYPES) and record.version:
                    await logger.info(f'Extracted Claude version from {record.__class__.__name__}: {record.version}')
                    return record.version

        # Fallback: get from subprocess (data is corrupted if we reach here)
        await logger.warning('No records with version field found - using subprocess fallback')
        result = subprocess.run(['claude', '--version'], capture_output=True, text=True, timeout=5)
        version = result.stdout.strip()
        await logger.info(f'Extracted Claude version from subprocess: {version}')
        return version

    async def _serialize_json(self, archive: SessionArchive, logger: LoggerProtocol) -> bytes:
        """Serialize archive to uncompressed JSON."""
        await logger.info('Serializing to JSON')
        json_str = archive.model_dump_json(indent=2, exclude_unset=True)
        return json_str.encode('utf-8')

    async def _serialize_zst(self, archive: SessionArchive, logger: LoggerProtocol) -> bytes:
        """Serialize archive to zstd-compressed JSON."""
        await logger.info('Serializing to zstd-compressed JSON')

        try:
            import zstandard as zstd
        except ImportError:
            raise RuntimeError('zstandard library not available. Install with: uv add zstandard')

        # Serialize to JSON
        json_str = archive.model_dump_json(indent=2, exclude_unset=True)
        json_bytes = json_str.encode('utf-8')

        # Compress with zstd (level from settings)
        compressor = zstd.ZstdCompressor(level=settings.COMPRESSION_LEVEL)
        compressed = compressor.compress(json_bytes)

        compression_ratio = len(json_bytes) / len(compressed)
        await logger.info(f'Compressed {len(json_bytes):,} → {len(compressed):,} bytes ({compression_ratio:.1f}x)')

        return compressed

    def _encode_path(self, path: Path) -> str:
        """
        Encode filesystem path for folder name.

        Claude uses pattern: /Users/rushi.arumalla/project → -Users-rushi-arumalla-project
        """
        return str(path).replace('/', '-').replace('.', '-')
