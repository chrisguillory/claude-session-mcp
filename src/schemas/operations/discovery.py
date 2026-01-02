"""
Discovery operation schemas.

Models for session discovery results.
Extracted from services/discovery.py for reuse.
"""

from __future__ import annotations

from pathlib import Path

from src.schemas.base import StrictModel


class SessionInfo(StrictModel):
    """
    Information about a discovered Claude Code session.

    Note: The project path cannot be reliably determined from the folder name
    due to lossy encoding (/, ., ' ', ~ all become -). Use extract_source_project_path()
    on loaded session records to get the actual project path.
    """

    session_id: str
    session_folder: Path  # The ~/.claude/projects/{encoded}/ folder, NOT decoded path
