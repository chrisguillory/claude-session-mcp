"""
Session parser service - JSONL file parsing and validation.

Framework-agnostic service for loading and validating Claude Code session files.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from src.protocols import LoggerProtocol, NullLogger
from src.schemas.session import SessionRecord, SessionRecordAdapter

# ==============================================================================
# Session Parser Service
# ==============================================================================


class SessionParserService:
    """
    Service for parsing Claude Code session JSONL files.

    Pure domain logic - loads and validates JSONL files into typed SessionRecord objects.
    """

    async def load_session_files(
        self, session_files: Sequence[Path], logger: LoggerProtocol | None = None
    ) -> dict[str, list[SessionRecord]]:
        """
        Load and parse session JSONL files.

        Args:
            session_files: Sequence of JSONL file paths
            logger: Logger instance (optional, uses NullLogger if None)

        Returns:
            Dict mapping filename to list of validated records
        """
        log = logger or NullLogger()
        files_data: dict[str, list[SessionRecord]] = {}

        for file_path in session_files:
            await log.info(f'Loading {file_path.name}')

            records = await self._parse_jsonl_file(file_path, log)
            files_data[file_path.name] = records

            await log.info(f'Loaded {len(records)} records from {file_path.name}')

        return files_data

    async def _parse_jsonl_file(self, file_path: Path, logger: LoggerProtocol) -> list[SessionRecord]:
        """
        Parse a single JSONL file into validated records.

        Args:
            file_path: Path to JSONL file
            logger: Logger instance

        Returns:
            List of validated SessionRecord objects
        """
        records = []

        with open(file_path, encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # Parse JSON (fail fast on error)
                raw_data = json.loads(line)

                # Validate with Pydantic (fail fast on error)
                record = SessionRecordAdapter.validate_python(raw_data)
                records.append(record)

        return records
