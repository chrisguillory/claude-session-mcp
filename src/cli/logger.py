"""
CLI logger adapter - implements LoggerProtocol for command-line usage.

Provides a simple logger that outputs to stdout/stderr for CLI commands.
"""

from __future__ import annotations


class CLILogger:
    """
    Logger implementation for CLI (implements LoggerProtocol from services).

    Outputs messages to stdout/stderr with optional verbose mode.
    """

    def __init__(self, verbose: bool = False) -> None:
        """
        Initialize CLI logger.

        Args:
            verbose: If True, show info messages. If False, only warnings/errors.
        """
        self.verbose = verbose

    async def info(self, message: str) -> None:
        """Log info message (only if verbose)."""
        if self.verbose:
            print(f'[INFO] {message}')

    async def warning(self, message: str) -> None:
        """Log warning message."""
        print(f'[WARNING] {message}')

    async def error(self, message: str) -> None:
        """Log error message."""
        print(f'[ERROR] {message}')
