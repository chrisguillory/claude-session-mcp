"""
Local filesystem storage backend.

Implements StorageBackend protocol for local filesystem storage.
"""

from __future__ import annotations

import pathlib


class LocalFileSystemStorage:
    """Local filesystem storage backend."""

    def __init__(self, base_path: pathlib.Path) -> None:
        """
        Initialize local filesystem storage.

        Args:
            base_path: Base directory for storing archives

        Raises:
            ValueError: If base_path doesn't exist (fail-fast)
        """
        if not base_path.exists():
            raise ValueError(f'Storage path does not exist: {base_path}. Please create it first.')

        if not base_path.is_dir():
            raise ValueError(f'Storage path is not a directory: {base_path}')

        self.base_path = base_path

    async def save(self, filename: str, data: bytes) -> str:
        """
        Save archive to local filesystem.

        Args:
            filename: Name of archive file
            data: Archive data (JSON or compressed JSON)

        Returns:
            Absolute path to saved file
        """
        file_path = self.base_path / filename
        file_path.write_bytes(data)
        return str(file_path.absolute())

    async def exists(self, filename: str) -> bool:
        """
        Check if archive exists in local filesystem.

        Args:
            filename: Name of archive file

        Returns:
            True if file exists
        """
        file_path = self.base_path / filename
        return file_path.exists()
