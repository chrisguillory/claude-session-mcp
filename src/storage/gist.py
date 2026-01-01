"""
GitHub Gist storage backend for session archives.

Provides async storage backend that saves/loads archives to/from GitHub Gists.
"""

from __future__ import annotations

from typing import Literal

import httpx


class GistStorage:
    """
    GitHub Gist storage backend.

    Stores session archives as GitHub Gists. Supports creating new gists
    or updating existing ones.
    """

    # GitHub API limits: 100MB hard limit, 50MB warning threshold
    MAX_FILE_SIZE_MB = 100
    WARNING_FILE_SIZE_MB = 50

    def __init__(
        self,
        token: str,
        gist_id: str | None = None,
        visibility: Literal['public', 'secret'] = 'secret',
        description: str = 'Claude Code Session Archive',
    ) -> None:
        """
        Initialize Gist storage backend.

        Args:
            token: GitHub Personal Access Token with 'gist' scope (empty string allowed for read-only)
            gist_id: Optional existing gist ID (if None, creates new gist on save)
            visibility: 'public' or 'secret' (default: secret)
            description: Gist description

        Raises:
            ValueError: If token is empty and trying to save (checked at save time)
        """
        self.token = token
        self.gist_id = gist_id
        self.visibility = visibility
        self.description = description
        self.base_url = 'https://api.github.com'

    async def save(self, filename: str, data: bytes) -> str:
        """
        Save archive to GitHub Gist.

        Args:
            filename: Archive filename
            data: Archive data (JSON or compressed)

        Returns:
            Gist URL (e.g., https://gist.github.com/{user}/{gist_id})

        Raises:
            ValueError: If file too large (>100MB) or token missing
            httpx.HTTPStatusError: If GitHub API call fails
        """
        # Check token
        if not self.token:
            raise ValueError('GitHub token is required to save to Gist')

        # Check file size (100MB hard limit)
        size_mb = len(data) / (1024 * 1024)
        if size_mb > self.MAX_FILE_SIZE_MB:
            raise ValueError(
                f'File too large for Gist: {size_mb:.2f}MB. '
                f'GitHub Gist files are limited to {self.MAX_FILE_SIZE_MB}MB. '
                f'Consider splitting the archive or using local storage.'
            )

        # Decode to text (Gists are text-based)
        try:
            content = data.decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError(
                'Binary files not supported by GitHub Gists. '
                'Gists only support UTF-8 text files. '
                'Use JSON format instead of compressed (.zst).'
            )

        # Create or update gist
        if self.gist_id:
            return await self._update_gist(filename, content)
        else:
            return await self._create_gist(filename, content)

    async def _create_gist(self, filename: str, content: str) -> str:
        """Create new gist."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f'{self.base_url}/gists',
                headers={
                    'Authorization': f'Bearer {self.token}',
                    'Accept': 'application/vnd.github.v3+json',
                    'X-GitHub-Api-Version': '2022-11-28',
                },
                json={
                    'description': self.description,
                    'public': self.visibility == 'public',
                    'files': {filename: {'content': content}},
                },
            )
            response.raise_for_status()

            gist_data = response.json()
            self.gist_id = gist_data['id']  # Store for future updates
            return gist_data['html_url']  # Return web URL

    async def _update_gist(self, filename: str, content: str) -> str:
        """Update existing gist."""
        async with httpx.AsyncClient() as client:
            response = await client.patch(
                f'{self.base_url}/gists/{self.gist_id}',
                headers={
                    'Authorization': f'Bearer {self.token}',
                    'Accept': 'application/vnd.github.v3+json',
                    'X-GitHub-Api-Version': '2022-11-28',
                },
                json={'files': {filename: {'content': content}}},
            )
            response.raise_for_status()

            gist_data = response.json()
            return gist_data['html_url']

    async def exists(self, filename: str) -> bool:
        """
        Check if gist exists and contains the specified file.

        Args:
            filename: Archive filename to check

        Returns:
            True if gist exists and contains the file
        """
        if not self.gist_id:
            return False

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self.base_url}/gists/{self.gist_id}',
                    headers={
                        'Accept': 'application/vnd.github.v3+json',
                        'X-GitHub-Api-Version': '2022-11-28',
                    },
                )

                if response.status_code == 404:
                    return False

                response.raise_for_status()
                gist_data = response.json()
                return filename in gist_data.get('files', {})
        except httpx.HTTPError:
            return False

    async def load(self, filename: str) -> bytes:
        """
        Load archive from GitHub Gist.

        Args:
            filename: Archive filename to load

        Returns:
            Archive data as bytes

        Raises:
            ValueError: If gist_id not set or file not found
            httpx.HTTPStatusError: If GitHub API call fails
        """
        if not self.gist_id:
            raise ValueError('Cannot load from gist: no gist_id provided')

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f'{self.base_url}/gists/{self.gist_id}',
                headers={
                    'Accept': 'application/vnd.github.v3+json',
                    'X-GitHub-Api-Version': '2022-11-28',
                },
            )
            response.raise_for_status()

            gist_data = response.json()
            files = gist_data.get('files', {})

            if filename not in files:
                raise ValueError(f"File '{filename}' not found in gist {self.gist_id}")

            # Get file content (must fetch from raw_url for truncated files)
            file_data = files[filename]
            if file_data.get('truncated', False) or 'content' not in file_data:
                # Truncated or missing content - fetch full content from raw_url
                raw_url = file_data.get('raw_url')
                if not raw_url:
                    raise ValueError(
                        f"File '{filename}' is truncated but no raw_url available. "
                        f'File size: {file_data.get("size", "unknown")} bytes'
                    )
                raw_response = await client.get(raw_url)
                raw_response.raise_for_status()
                content = raw_response.text
            else:
                content = file_data['content']

            return content.encode('utf-8')
