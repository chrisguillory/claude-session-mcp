"""
Operation schemas for service results.

This package contains Pydantic models for operation results returned by services.
These are extracted from service files to enable reuse and cleaner separation.
"""

from __future__ import annotations

from src.schemas.operations.archive import (
    ARCHIVE_FORMAT_VERSION,
    ArchiveMetadata,
    FileMetadata,
    SessionArchive,
)
from src.schemas.operations.context import SessionContext, SessionSource, SessionState
from src.schemas.operations.delete import ArtifactFile, DeleteManifest, DeleteResult
from src.schemas.operations.discovery import SessionInfo
from src.schemas.operations.gist import GistArchiveResult
from src.schemas.operations.lineage import LineageEntry, LineageFile, LineageResult
from src.schemas.operations.restore import RestoreResult

__all__ = [
    # Archive
    'ARCHIVE_FORMAT_VERSION',
    'FileMetadata',
    'ArchiveMetadata',
    'SessionArchive',
    # Context
    'SessionContext',
    'SessionSource',
    'SessionState',
    # Delete
    'ArtifactFile',
    'DeleteManifest',
    'DeleteResult',
    # Discovery
    'SessionInfo',
    # Gist
    'GistArchiveResult',
    # Lineage
    'LineageEntry',
    'LineageFile',
    'LineageResult',
    # Restore
    'RestoreResult',
]
