"""
JSONL file writing for session clone/restore/delete operations.

Extracted from clone.py and restore.py where it was duplicated identically.
Handles serialization with slug and agent ID mapping application.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.schemas.session import SessionRecord

from .agent_ids import apply_agent_id_mapping
from .plan_files import apply_slug_mapping


def write_jsonl(
    path: Path,
    records: Sequence[SessionRecord],
    slug_mapping: Mapping[str, str],
    agent_id_mapping: Mapping[str, str],
) -> None:
    """Write session records to JSONL file, applying slug and agent ID mappings.

    Serialization uses exclude_unset=True for round-trip fidelity (preserves
    fields explicitly set to None, unlike exclude_none which drops them).

    Args:
        path: Output file path
        records: Session records to write
        slug_mapping: Old slug -> new slug mapping (empty for identity/rollback)
        agent_id_mapping: Old agent ID -> new agent ID mapping (empty for identity/rollback)
    """
    with open(path, 'w', encoding='utf-8') as f:
        for record in records:
            # Use exclude_unset for round-trip fidelity
            json_data = record.model_dump(exclude_unset=True, mode='json')
            # Use compact separators for consistent, smaller output
            json_str = json.dumps(json_data, separators=(',', ':'))

            # Apply slug mapping first (longer strings, less collision risk)
            if slug_mapping:
                json_str = apply_slug_mapping(json_str, slug_mapping)

            # Apply agent ID mapping second
            if agent_id_mapping:
                json_str = apply_agent_id_mapping(json_str, agent_id_mapping)

            f.write(json_str + '\n')
