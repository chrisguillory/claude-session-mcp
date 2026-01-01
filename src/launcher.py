"""
Claude Code launcher utility.

Provides functionality to launch Claude Code with a specific session.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence


def launch_claude_with_session(session_id: str, extra_args: Sequence[str]) -> None:
    """
    Launch Claude Code with --resume, replacing current process.

    Uses os.execvp() for clean process handoff - the current process
    is replaced by Claude Code, so this function never returns.

    Args:
        session_id: Session ID to resume
        extra_args: Additional arguments to pass to claude CLI (e.g., --chrome)

    Raises:
        RuntimeError: If Claude Code CLI is not found in PATH
    """
    claude_path = shutil.which('claude')
    if not claude_path:
        raise RuntimeError('Claude Code CLI not found in PATH.\nInstall from: https://claude.ai/code')

    # Build command with optional extra args
    cmd = ['claude', '--resume', session_id]
    if extra_args:
        cmd.extend(extra_args)

    # Replace current process with Claude
    os.execvp('claude', cmd)
