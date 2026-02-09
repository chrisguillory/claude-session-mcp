"""
Session archive service - framework-agnostic domain logic.

Pure service layer with no MCP/FastAPI dependencies. Handles session discovery,
archive creation, compression, and format detection.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from src.config.mcp import settings
from src.paths import encode_path
from src.protocols import LoggerProtocol, NullLogger
from src.schemas.operations.archive import (
    ARCHIVE_FORMAT_VERSION,
    AgentFileEntry,
    ArchiveMetadata,
    FileMetadata,
    MainSessionFileEntry,
    PlanFileEntry,
    SessionArchiveV2,
    TodoFileEntry,
    ToolResultEntry,
    parse_agent_metadata,
)
from src.schemas.session import SessionRecord
from src.services.artifacts import (
    collect_plan_files,
    collect_task_metadata,
    collect_todos,
    collect_tool_results,
    detect_agent_structure,
    extract_custom_title_from_records,
    extract_slugs_from_records,
    extract_source_project_path,
    iter_tasks,
    validate_session_env_empty,
)
from src.services.lineage import get_machine_id
from src.services.parser import SessionParserService
from src.services.version import get_version
from src.storage.protocol import StorageBackend

# ==============================================================================
# Archive Format Detection
# ==============================================================================


class FormatDetector:
    """Detects and validates archive format from file extension and format parameter."""

    SUPPORTED_FORMATS = {'json', 'zst'}  # 'zst' = JSON with zstd compression

    # Extension to format mapping
    EXTENSION_MAP: dict[str, Literal['json', 'zst']] = {
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


class SessionArchiveService:
    """
    Service for creating session archives.

    Pure domain logic - no MCP/FastAPI dependencies. Takes session_id and
    discovers all related JSONL files from ~/.claude/projects/.
    """

    def __init__(
        self,
        session_id: str,
        temp_dir: Path,
        parser_service: SessionParserService,
        *,
        project_path: Path | None = None,
        session_folder: Path | None = None,
        claude_pid: int | None = None,
    ) -> None:
        """
        Initialize archive service.

        Args:
            session_id: Current Claude Code session ID
            temp_dir: Temporary directory for default output
            parser_service: Session parser service for loading JSONL files
            project_path: Current project directory (used to find session folder via encoding)
            session_folder: Session folder directly (bypasses encoding, use when discovered)
            claude_pid: Optional PID of running Claude process (for accurate version detection)

        Note: Provide either project_path OR session_folder.
        - MCP handlers have the real project_path from lsof - use that
        - CLI archive command has session_folder from discovery - use that
        """
        if not project_path and not session_folder:
            raise ValueError('Either project_path or session_folder must be provided')

        self.session_id = session_id
        self.project_path = project_path
        self.temp_dir = temp_dir
        self.parser_service = parser_service
        self.claude_pid = claude_pid

        # Claude stores sessions here
        self.claude_sessions_dir = Path.home() / '.claude' / 'projects'

        # If session_folder provided, use it directly (no encoding needed)
        # Otherwise, computed lazily from project_path
        self._project_folder: Path | None = session_folder

    def _get_project_folder(self) -> Path:
        """
        Get the project folder in ~/.claude/projects/.

        Computes and caches the encoded project path. This is the directory
        containing session JSONL files for this project.

        Returns:
            Path to project folder

        Raises:
            FileNotFoundError: If project folder not found
        """
        if self._project_folder is not None:
            return self._project_folder

        assert self.project_path is not None  # Ensured by __init__ validation
        encoded_project = encode_path(self.project_path)
        project_folders = list(self.claude_sessions_dir.glob('*'))

        for folder in project_folders:
            if folder.name.startswith(encoded_project):
                self._project_folder = folder
                return folder

        raise FileNotFoundError(
            f'Could not find session folder for project {self.project_path} in {self.claude_sessions_dir}'
        )

    async def create_archive(
        self,
        storage: StorageBackend,
        output_path: str | None,
        format_param: Literal['json', 'zst'] | None,
        logger: LoggerProtocol | None = None,
    ) -> ArchiveMetadata:
        """
        Create archive of current session.

        Args:
            storage: Storage backend (call-time parameter!)
            output_path: Optional output path (None = temp file)
            format_param: Optional format override
            logger: Optional logger instance (uses NullLogger if None)

        Returns:
            Archive metadata

        Raises:
            ValueError: If format detection fails or conflicts
            FileNotFoundError: If session files not found
        """
        log = logger or NullLogger()
        await log.info(f'Creating archive for session {self.session_id}')

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

            await log.info(f'Output path: {output_file}')
        else:
            # Use temp directory (default) with correct extension for format
            timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
            ext = '.json.zst' if format_param == 'zst' else '.json'
            filename = f'session-{self.session_id[:8]}-{timestamp}{ext}'
            output_file = self.temp_dir / filename
            await log.info(f'Using temp file: {output_file}')

        # Detect format
        archive_format = FormatDetector.detect_format(output_file, format_param)
        await log.info(f'Archive format: {archive_format}')

        # Get project folder (computed once, cached)
        project_folder = self._get_project_folder()
        await log.info(f'Project folder: {project_folder.name}')

        # Discover session files
        session_files = await self._discover_session_files(project_folder, log)
        await log.info(f'Found {len(session_files)} session files')

        # Load and parse all records
        files_data, total_records = await self._load_session_files(session_files, log)
        await log.info(f'Loaded {total_records} total records')

        # Extract Claude Code version from records
        claude_code_version = await self._extract_claude_code_version(files_data, log)

        # Extract slugs and collect plan files
        slugs = extract_slugs_from_records(files_data)
        await log.info(f'Found {len(slugs)} unique slugs in session')

        plan_files = collect_plan_files(slugs)
        await log.info(f'Collected {len(plan_files)} plan files (of {len(slugs)} slugs)')

        # Collect tool results (v1.2+)
        tool_results = collect_tool_results(project_folder, self.session_id)
        await log.info(f'Collected {len(tool_results)} tool result files')

        # Collect todos (v1.2+)
        todos = collect_todos(self.session_id)
        await log.info(f'Collected {len(todos)} todo files')

        # Validate session-env is empty (future-proofing)
        validate_session_env_empty(self.session_id)

        # Extract source project path from session records (source of truth)
        # We use cwd from records, not self.project_path, because:
        # 1. MCP handlers pass Claude's cwd which is correct
        # 2. CLI archive command may pass a lossy-decoded path from SessionDiscoveryService
        # The record cwd field is always the authoritative source
        source_project_path = extract_source_project_path(files_data)

        # Extract custom title (user-defined session name from /rename)
        custom_title = extract_custom_title_from_records(files_data)
        if custom_title:
            await log.info(f'Found custom title: {custom_title}')

        # Collect tasks
        tasks = list(iter_tasks(self.session_id))
        if tasks:
            await log.info(f'Collected {len(tasks)} tasks')

        # Collect task metadata (.highwatermark, etc.)
        task_metadata = collect_task_metadata(self.session_id)
        if task_metadata:
            await log.info(f'Collected {len(task_metadata)} task metadata files')

        # Build v2 explicit entry models
        main_filename = f'{self.session_id}.jsonl'
        main_records = files_data.get(main_filename, [])
        main_session = MainSessionFileEntry(
            record_count=len(main_records),
            records=main_records,
        )

        # Build agent entries with structure detection
        agent_entries: list[AgentFileEntry] = []
        for file_path in session_files:
            if not file_path.name.startswith('agent-'):
                continue

            agent_id, agent_type = parse_agent_metadata(file_path.name)
            nested = detect_agent_structure(file_path, self.session_id, project_folder)
            records = files_data.get(file_path.name, [])

            agent_entries.append(
                AgentFileEntry(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    nested=nested,
                    record_count=len(records),
                    records=records,
                )
            )

        # Convert artifacts to v2 entry models
        plan_entries = [PlanFileEntry(slug=slug, content=content) for slug, content in plan_files.items()]

        tool_result_entries = [
            ToolResultEntry(tool_use_id=tool_use_id, content=content) for tool_use_id, content in tool_results.items()
        ]

        # Extract agent_id from todo filename: {session_id}-agent-{agent_id}.json
        # collect_todos() already filters by session_id, so we know the prefix matches
        todo_prefix = f'{self.session_id}-agent-'
        todo_entries = [
            TodoFileEntry(
                agent_id=filename.removeprefix(todo_prefix).removesuffix('.json'),
                content=content,
            )
            for filename, content in todos.items()
        ]

        # Calculate statistics
        total_agent_records = sum(agent.record_count for agent in agent_entries)

        # Handle None version (v2 requires string)
        if claude_code_version is None:
            claude_code_version = 'unknown'
            await log.warning('Claude Code version not found, using "unknown"')

        # Create v2 archive structure
        archive = SessionArchiveV2(
            version=ARCHIVE_FORMAT_VERSION,
            session_id=self.session_id,
            archived_at=datetime.now(UTC),
            original_project_path=str(source_project_path),
            claude_code_version=claude_code_version,
            machine_id=get_machine_id(),
            custom_title=custom_title,
            main_session=main_session,
            agent_files=agent_entries,
            plan_files=plan_entries,
            tool_results=tool_result_entries,
            todos=todo_entries,
            tasks=tasks,
            task_metadata=task_metadata,
            total_session_records=len(main_records) + total_agent_records,
            total_agent_records=total_agent_records,
        )

        # Serialize and compress
        if archive_format == 'json':
            data = await self._serialize_json(archive, log)
        elif archive_format == 'zst':
            data = await self._serialize_zst(archive, log)
        else:
            raise ValueError(f'Unsupported format: {archive_format}')

        # Save via storage backend
        final_path = await storage.save(output_file.name, data)
        await log.info(f'Archive saved: {len(data):,} bytes')

        # Calculate size in MB (rounded to 2 decimal places)
        size_mb = round(len(data) / (1024 * 1024), 2)

        # Build per-file metadata and calculate record breakdown
        file_metadata = []
        session_records = 0
        agent_records = 0
        for filename, records in files_data.items():
            record_count = len(records)
            file_metadata.append(FileMetadata(filename=filename, record_count=record_count))
            if filename.startswith('agent-'):
                agent_records += record_count
            else:
                session_records += record_count

        return ArchiveMetadata(
            file_path=final_path,
            session_id=self.session_id,
            format=archive_format,
            size_mb=size_mb,
            archived_at=datetime.now(UTC),
            session_records=session_records,
            agent_records=agent_records,
            file_count=len(session_files),
            files=file_metadata,
            custom_title=custom_title,
        )

    async def _discover_session_files(self, project_folder: Path, logger: LoggerProtocol) -> Sequence[Path]:
        """
        Discover all JSONL files for current session.

        Finds:
        - Main session file: {session_id}.jsonl
        - Agent session files (flat, pre-2.1.2): agent-*.jsonl
        - Agent session files (nested, 2.1.2+): {session_id}/subagents/agent-*.jsonl

        Args:
            project_folder: Path to project folder in ~/.claude/projects/
            logger: Logger for progress messages

        Returns:
            Sequence of JSONL file paths

        Raises:
            FileNotFoundError: If main session file not found
        """
        await logger.info('Discovering session files')

        # Find main session file
        main_file = project_folder / f'{self.session_id}.jsonl'
        if not main_file.exists():
            raise FileNotFoundError(f'Main session file not found: {main_file}')

        session_files = [main_file]

        # Find agent files that belong to this session using rg
        # Search both flat (project_folder/) and nested (project_folder/<sid>/subagents/)
        # Note: JSON may have optional space after colon, so we use regex pattern
        pattern = f'"sessionId":\\s*"{self.session_id}"'
        result = subprocess.run(
            ['rg', '--files-with-matches', pattern, '--glob', '**/agent-*.jsonl', str(project_folder)],
            capture_output=True,
            text=True,
        )

        agent_files = []
        if result.stdout.strip():
            agent_files = [Path(line) for line in result.stdout.strip().split('\n')]

        session_files.extend(agent_files)

        await logger.info(f'Found {len(session_files)} files: 1 main + {len(agent_files)} agents')

        return session_files

    async def _load_session_files(
        self, session_files: Sequence[Path], logger: LoggerProtocol
    ) -> tuple[dict[str, list[SessionRecord]], int]:
        """
        Load and parse all session files using parser service.

        Args:
            session_files: Sequence of JSONL file paths
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
    ) -> str | None:
        """
        Extract Claude Code version using best available method.

        Fallback chain:
        1. Process-based detection (if claude_pid was provided and process exists)
        2. Session records (find first record with version field)
        3. None (if version cannot be determined)

        Args:
            files_data: Parsed session records
            logger: Logger instance

        Returns:
            Claude Code version string (e.g., '2.0.37'), or None if not found
        """
        # Flatten all records from all files
        all_records = [record for records in files_data.values() for record in records]

        version = get_version(claude_pid=self.claude_pid, records=all_records)

        if version:
            await logger.info(f'Extracted Claude version: {version}')
        else:
            await logger.warning('No version found')

        return version

    async def _serialize_json(self, archive: SessionArchiveV2, logger: LoggerProtocol) -> bytes:
        """Serialize archive to uncompressed JSON."""
        await logger.info('Serializing to JSON')
        json_str = archive.model_dump_json(indent=2, exclude_unset=True)
        return json_str.encode('utf-8')

    async def _serialize_zst(self, archive: SessionArchiveV2, logger: LoggerProtocol) -> bytes:
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
