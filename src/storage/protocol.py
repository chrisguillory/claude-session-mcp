"""
Storage backend protocol for session archives.

Defines interface for different storage backends (local, S3, Gist).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for session archive storage backends."""

    async def save(self, filename: str, data: bytes) -> str:
        """
        Save archive data.

        Args:
            filename: Name of archive file
            data: Archive data (JSON or compressed JSON)

        Returns:
            Final URI/path where archive was saved

        Raises:
            ValueError: If save location is invalid
            RuntimeError: If save operation fails
        """
        ...

    async def exists(self, filename: str) -> bool:
        """
        Check if archive exists.

        Args:
            filename: Name of archive file

        Returns:
            True if archive exists
        """
        ...

    async def load(self, filename: str) -> bytes:
        """
        Load archive data (optional - not all backends support loading).

        Args:
            filename: Name of archive file

        Returns:
            Archive data as bytes

        Raises:
            NotImplementedError: If backend doesn't support loading
            ValueError: If file not found
            RuntimeError: If load operation fails
        """
        raise NotImplementedError(f'{type(self).__name__} does not support loading')
