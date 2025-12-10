"""
Session clone service - direct session-to-session cloning.

Clones a session directly without creating an intermediate archive file.
Faster than archive+restore for local cloning operations.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import uuid6

from src.models import SessionRecord, SessionRecordAdapter
from src.services.archive import LoggerProtocol
from src.services.discovery import SessionDiscoveryService, SessionInfo
from src.services.parser import SessionParserService
from src.services.restore import PathTranslator, RestoreResult


class AmbiguousSessionError(Exception):
    """Raised when a session ID prefix matches multiple sessions."""

    def __init__(self, prefix: str, matches: list[str]) -> None:
        self.prefix = prefix
        self.matches = matches
        matches_str = '\n  '.join(matches[:10])
        if len(matches) > 10:
            matches_str += f'\n  ... and {len(matches) - 10} more'
        super().__init__(
            f"Session ID prefix '{prefix}' is ambiguous. Matches {len(matches)} sessions:\n  {matches_str}"
        )


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
            AmbiguousSessionError: If session ID prefix matches multiple sessions
        """
        # Resolve session ID (handle prefix matching)
        session_info = await self._resolve_session(source_session_id, logger)

        if logger:
            await logger.info(f'Cloning session: {session_info.session_id}')
            await logger.info(f'Source project: {session_info.project_path}')

        # Discover all session files (main + agents)
        session_files = await self._discover_session_files(session_info, logger)

        if logger:
            await logger.info(f'Found {len(session_files)} session files')

        # Load all records
        files_data = await self.parser_service.load_session_files(session_files, logger)

        # Generate new session ID (UUIDv7 for identification)
        new_session_id = str(uuid6.uuid7())
        if logger:
            await logger.info(f'Generated new session ID (UUIDv7): {new_session_id}')

        # Create path translator if needed
        translator = None
        if translate_paths and session_info.project_path != self.target_project_path:
            translator = PathTranslator(str(session_info.project_path), str(self.target_project_path))
            if logger:
                await logger.info(f'Path translation: {session_info.project_path} -> {self.target_project_path}')

        # Create target directory
        target_dir = self._get_session_directory()
        target_dir.mkdir(parents=True, exist_ok=True)
        if logger:
            await logger.info(f'Target directory: {target_dir}')

        # Write transformed files
        main_file_path = None
        agent_files = []
        total_records = 0

        for filename, records in files_data.items():
            # Determine new filename
            if filename == f'{session_info.session_id}.jsonl':
                new_filename = f'{new_session_id}.jsonl'
                main_file_path = str(target_dir / new_filename)
            elif filename.startswith('agent-'):
                new_filename = filename
                agent_files.append(str(target_dir / new_filename))
            else:
                new_filename = filename

            # Transform records
            updated_records = []
            for record in records:
                record_dict = record.model_dump(exclude_unset=True, mode='json')

                # Update session ID if present
                if 'sessionId' in record_dict:
                    record_dict['sessionId'] = new_session_id

                # Validate and reconstruct
                updated_record = SessionRecordAdapter.validate_python(record_dict)

                # Translate paths if needed
                if translator:
                    updated_record = translator.translate_record(updated_record)

                updated_records.append(updated_record)

            # Write JSONL file
            output_path = target_dir / new_filename
            await self._write_jsonl(output_path, updated_records)
            total_records += len(updated_records)

            if logger:
                await logger.info(f'Cloned {new_filename}: {len(updated_records)} records')

        return RestoreResult(
            new_session_id=new_session_id,
            original_session_id=session_info.session_id,
            restored_at=datetime.now(timezone.utc),
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

        Args:
            session_id_or_prefix: Full session ID or prefix
            logger: Optional logger

        Returns:
            SessionInfo for the matched session

        Raises:
            FileNotFoundError: If no sessions match
            AmbiguousSessionError: If multiple sessions match
        """
        # First try exact match
        exact_match = await self.discovery_service.find_session_by_id(session_id_or_prefix)
        if exact_match:
            return exact_match

        # Try prefix match
        if logger:
            await logger.info(f'No exact match, trying prefix: {session_id_or_prefix}')

        matches = await self._find_sessions_by_prefix(session_id_or_prefix)

        if not matches:
            raise FileNotFoundError(f'No session found matching: {session_id_or_prefix}')

        if len(matches) > 1:
            raise AmbiguousSessionError(session_id_or_prefix, [m.session_id for m in matches])

        return matches[0]

    async def _find_sessions_by_prefix(self, prefix: str) -> list[SessionInfo]:
        """Find all sessions matching a prefix."""
        if not self.claude_sessions_dir.exists():
            return []

        # Use rg to find all matching session files
        result = subprocess.run(
            ['rg', '--files', '--glob', f'{prefix}*.jsonl', str(self.claude_sessions_dir)],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if not result.stdout.strip():
            return []

        matches = []
        for line in result.stdout.strip().split('\n'):
            session_file = Path(line)
            # Skip agent files
            if session_file.name.startswith('agent-'):
                continue

            session_id = session_file.stem
            project_dir = session_file.parent
            project_path = self.discovery_service._decode_path(project_dir.name)

            matches.append(SessionInfo(session_id=session_id, project_path=project_path))

        return matches

    async def _discover_session_files(
        self,
        session_info: SessionInfo,
        logger: LoggerProtocol | None,
    ) -> list[Path]:
        """Discover all JSONL files for a session (main + agents)."""
        # Get the session directory
        encoded_path = self._encode_path(session_info.project_path)
        session_dir = self.claude_sessions_dir / encoded_path

        if not session_dir.exists():
            raise FileNotFoundError(f'Session directory not found: {session_dir}')

        # Find main session file
        main_file = session_dir / f'{session_info.session_id}.jsonl'
        if not main_file.exists():
            raise FileNotFoundError(f'Main session file not found: {main_file}')

        session_files = [main_file]

        # Find agent files belonging to this session
        result = subprocess.run(
            [
                'rg',
                '--files-with-matches',
                f'"sessionId":"{session_info.session_id}"',
                '--glob',
                'agent-*.jsonl',
                str(session_dir),
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.stdout.strip():
            agent_files = [Path(line) for line in result.stdout.strip().split('\n')]
            session_files.extend(agent_files)

        return session_files

    async def _write_jsonl(self, path: Path, records: list[SessionRecord]) -> None:
        """Write records to JSONL file."""
        with open(path, 'w', encoding='utf-8') as f:
            for record in records:
                json_data = record.model_dump(exclude_unset=True, mode='json')
                f.write(json.dumps(json_data) + '\n')

    def _get_session_directory(self) -> Path:
        """Get the session directory for the target project."""
        encoded = self._encode_path(self.target_project_path)
        return self.claude_sessions_dir / encoded

    def _encode_path(self, path: Path) -> str:
        """Encode path for Claude's directory naming."""
        return str(path).replace('/', '-').replace('.', '-')