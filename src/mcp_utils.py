"""Shared utilities for MCP servers."""

from __future__ import annotations

# Standard Library
from datetime import UTC, datetime
from typing import Any

# Third-Party Libraries
from mcp.server.fastmcp import Context


class DualLogger:
    """Logs messages to both stdout and MCP client context."""

    def __init__(self, ctx: Context[Any, Any, Any]) -> None:
        self.ctx = ctx

    def _timestamp(self) -> str:
        return datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')

    async def info(self, message: str) -> None:
        print(f'[{self._timestamp()}] [INFO] {message}')
        await self.ctx.info(message)

    async def debug(self, message: str) -> None:
        print(f'[{self._timestamp()}] [DEBUG] {message}')
        await self.ctx.debug(message)

    async def warning(self, message: str) -> None:
        print(f'[{self._timestamp()}] [WARNING] {message}')
        await self.ctx.warning(message)

    async def error(self, message: str) -> None:
        print(f'[{self._timestamp()}] [ERROR] {message}')
        await self.ctx.error(message)
