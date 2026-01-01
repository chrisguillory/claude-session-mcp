"""
Shared protocols for Claude Session services.

This module contains Protocol definitions used across multiple services.
Having a single source of truth for protocols prevents type incompatibility
issues when the same protocol is defined in multiple modules.
"""

from __future__ import annotations

from typing import Protocol


class LoggerProtocol(Protocol):
    """
    Protocol for async logger - enables services to work with any logging implementation.

    Implementations:
    - DualLogger (mcp_utils.py): Logs to both stdout and MCP client
    - CLILogger (cli_logger.py): Logs to stdout with optional verbose mode
    - NullLogger (below): No-op implementation for when logging is optional
    """

    async def info(self, message: str) -> None: ...
    async def warning(self, message: str) -> None: ...
    async def error(self, message: str) -> None: ...


class NullLogger:
    """
    No-op logger implementation for when logging is optional.

    Use this when a function requires a LoggerProtocol but the caller
    doesn't need logging output.
    """

    async def info(self, message: str) -> None:
        pass

    async def warning(self, message: str) -> None:
        pass

    async def error(self, message: str) -> None:
        pass
