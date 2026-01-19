"""
Custom title handling for clone/restore operations.

Handles:
- Custom title extraction from session records
- Base title extraction (stripping clone suffix)
- Clone title generation with provenance

Custom title naming follows a similar pattern to slugs:
- Native: "My Session Name"
- Cloned: "My Session Name (clone-XXXXXXXX)" (8-char session prefix)

When cloning a clone, we extract the base title first to avoid accumulation:
- "Auth Feature (clone-019b5227)" → "Auth Feature (clone-019b53d9)"
  (NOT "Auth Feature (clone-019b5227) (clone-019b53d9)")
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.schemas.session import CustomTitleRecord, SessionRecord

# Regex to match clone suffix: " (clone-XXXXXXXX)" where X is hex char
CLONE_SUFFIX_PATTERN = re.compile(r'\s*\(clone-[0-9a-fA-F]{8}\)$')


def extract_custom_title_from_records(
    files_data: Mapping[str, Sequence[SessionRecord]],
) -> str | None:
    """
    Extract the custom title from session records.

    Session files are append-only, so if the user renamed multiple times,
    there will be multiple CustomTitleRecord entries. We return the last
    one (most recent).

    Note: Assumes CustomTitleRecord only appears in the main session file
    (user renames are not agent operations). If multiple files contain
    CustomTitleRecords, the 'last' is determined by dict iteration order
    which may not be chronological across files.

    Args:
        files_data: Mapping of filename -> sequence of SessionRecord

    Returns:
        The custom title string, or None if no CustomTitleRecord found
    """
    custom_title: str | None = None
    for records in files_data.values():
        for record in records:
            if isinstance(record, CustomTitleRecord):
                custom_title = record.customTitle
    return custom_title


def extract_base_custom_title(title: str) -> str:
    """
    Extract the base title, removing any (clone-XXXXXXXX) suffix.

    For native titles, returns the title unchanged.
    For cloned titles, returns the portion before the clone suffix.

    Examples:
        'Auth Feature' -> 'Auth Feature'
        'Auth Feature (clone-019b5227)' -> 'Auth Feature'

    This enables flat cloning: cloning a clone produces the same
    format as cloning a native session, just with a different suffix.

    Args:
        title: Custom title string (native or cloned format)

    Returns:
        Base title without any clone suffix
    """
    return CLONE_SUFFIX_PATTERN.sub('', title)


def generate_clone_custom_title(original_title: str, new_session_id: str) -> str:
    """
    Generate new custom title with provenance.

    Format: {base_title} (clone-{session_prefix})
    Example: "Auth Feature (clone-019b51bd)"

    When cloning a clone, we extract the base title first to keep
    the naming flat rather than accumulating suffixes.

    Args:
        original_title: Original custom title string (may already be cloned)
        new_session_id: New session ID for the clone

    Returns:
        New custom title string showing provenance
    """
    base_title = extract_base_custom_title(original_title)
    prefix = new_session_id[:8]
    return f'{base_title} (clone-{prefix})'


def extract_custom_title_from_file(session_file: Path) -> str | None:
    """
    Extract the custom title from a session file efficiently.

    Uses grep to find custom-title records without parsing the entire file,
    then parses only the matching lines. Returns the last custom title found.

    Args:
        session_file: Path to the session JSONL file

    Returns:
        The custom title string, or None if no CustomTitleRecord found
    """
    if not session_file.exists():
        return None

    # Use rg to find lines containing custom-title records
    result = subprocess.run(
        ['rg', '--no-filename', '"type":"custom-title"', str(session_file)],
        capture_output=True,
        text=True,
    )

    if not result.stdout.strip():
        return None

    # Parse each matching line and return the last custom title
    custom_title: str | None = None
    for line in result.stdout.strip().split('\n'):
        record = json.loads(line)
        if record.get('type') == 'custom-title':
            custom_title = record.get('customTitle')

    return custom_title
