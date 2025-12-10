"""
Claude Code launcher utility.

Provides functionality to launch Claude Code with a specific session.
"""

from __future__ import annotations

import os
import shutil


def launch_claude_with_session(session_id: str) -> None:
    """
    Launch Claude Code with --resume, replacing current process.

    Uses os.execvp() for clean process handoff - the current process
    is replaced by Claude Code, so this function never returns.

    Args:
        session_id: Session ID to resume

    Raises:
        RuntimeError: If Claude Code CLI is not found in PATH
    """
    claude_path = shutil.which('claude')
    if not claude_path:
        raise RuntimeError(
            'Claude Code CLI not found in PATH.\n'
            'Install from: https://claude.ai/code'
        )

    # Replace current process with Claude
    os.execvp('claude', ['claude', '--resume', session_id])