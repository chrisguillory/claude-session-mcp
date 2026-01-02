"""
Agent ID handling for clone/restore operations.

Handles:
- Agent ID extraction from session files
- Clone agent ID generation with provenance
- Agent ID mapping for session transformations

Agent file patterns:
- Native: agent-<hex-id>.jsonl (e.g., "agent-5271c147.jsonl")
- Cloned: agent-<hex-id>-clone-<prefix>.jsonl (e.g., "agent-5271c147-clone-019b51bd.jsonl")

Cloning always produces a flat structure: {base-hex}-clone-{new-prefix}
When cloning a clone, we extract the base hex and apply the new prefix,
NOT append another -clone- segment. This keeps filenames simple and consistent.

Key insight: New agent IDs are REQUIRED for same-project forking.
Without them, cloning a session within the same project would overwrite
the parent session's agent files.

Note: Agent session IDs (in isSidechain records) are full UUIDs
that reference separate agent-{agentId}.jsonl files. The agentId
is the hex portion extracted from these filenames.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence, Set

from src.schemas.session import SessionRecord

# Pattern for agent filenames - matches both native and cloned formats:
# - agent-5271c147.jsonl (native)
# - agent-5271c147-clone-019b51bd.jsonl (cloned)
AGENT_FILENAME_PATTERN = re.compile(r'^agent-([a-f0-9]+(?:-clone-[a-f0-9]+)?)\.jsonl$')


def extract_base_agent_id(agent_id: str) -> str:
    """
    Extract the base hex ID from an agent ID.

    For native IDs, returns the ID unchanged.
    For cloned IDs, returns the hex portion before '-clone-'.

    Examples:
        '5271c147' -> '5271c147'
        '5271c147-clone-019b51bd' -> '5271c147'

    This enables flat cloning: cloning a clone produces the same
    format as cloning a native session, just with a different suffix.

    Args:
        agent_id: Agent ID (native or cloned format)

    Returns:
        Base hex ID without any clone suffix
    """
    if '-clone-' in agent_id:
        return agent_id.split('-clone-')[0]
    return agent_id


def extract_agent_ids_from_files(
    files_data: Mapping[str, Sequence[SessionRecord]],
) -> Set[str]:
    """
    Extract agent IDs from loaded session files.

    Agent IDs are extracted from filenames, not record content.
    Handles both native and cloned filename patterns.

    Ignores the main session file and any non-agent files.

    Args:
        files_data: Mapping of filename -> sequence of SessionRecord

    Returns:
        Set of agent ID strings. May include:
        - Native IDs: {"5271c147", "5848e60e"}
        - Cloned IDs: {"5271c147-clone-019b51bd"}
    """
    agent_ids: set[str] = set()

    for filename in files_data:
        match = AGENT_FILENAME_PATTERN.match(filename)
        if match:
            agent_ids.add(match.group(1))

    return agent_ids


def generate_clone_agent_id(old_agent_id: str, new_session_id: str) -> str:
    """
    Generate new agent ID with provenance.

    Format: {base_hex}-clone-{session_prefix}

    Examples:
        ('5271c147', '019b51bd-...') -> '5271c147-clone-019b51bd'
        ('5271c147-clone-019b51bd', '019c1234-...') -> '5271c147-clone-019c1234'

    Note: When cloning a clone, we extract the base hex ID first.
    This produces a flat structure rather than accumulating -clone- segments.
    Each cloned session gets a unique suffix from its session ID prefix.

    Args:
        old_agent_id: Original agent ID (native hex or cloned format)
        new_session_id: New session ID for provenance

    Returns:
        New agent ID: {base_hex}-clone-{8-char-prefix}
    """
    base_id = extract_base_agent_id(old_agent_id)
    prefix = new_session_id[:8]
    return f'{base_id}-clone-{prefix}'


def generate_agent_id_mapping(
    agent_ids: Set[str],
    new_session_id: str,
) -> Mapping[str, str]:
    """
    Generate mapping of old agent IDs to new cloned agent IDs.

    Applies generate_clone_agent_id to each agent ID.

    Args:
        agent_ids: Set of original agent IDs
        new_session_id: New session ID for provenance

    Returns:
        Mapping of old_agent_id -> new_agent_id
    """
    return {old_id: generate_clone_agent_id(old_id, new_session_id) for old_id in agent_ids}


def transform_agent_filename(
    old_filename: str,
    agent_id_mapping: Mapping[str, str],
) -> str:
    """
    Transform an agent filename using the agent ID mapping.

    Pattern: agent-{old_id}.jsonl -> agent-{new_id}.jsonl

    Extracts the agent ID from the filename, looks it up in the mapping,
    and constructs the new filename.

    Args:
        old_filename: Original filename (e.g., "agent-5271c147.jsonl")
        agent_id_mapping: Mapping of old_agent_id -> new_agent_id

    Returns:
        Transformed filename (e.g., "agent-5271c147-clone-019b51bd.jsonl")

    Raises:
        ValueError: If filename doesn't match agent-*.jsonl pattern
        KeyError: If agent ID not found in mapping
    """
    match = AGENT_FILENAME_PATTERN.match(old_filename)
    if not match:
        raise ValueError(f"Filename doesn't match agent-*.jsonl pattern: {old_filename}")

    old_agent_id = match.group(1)

    if old_agent_id not in agent_id_mapping:
        raise KeyError(f'Agent ID not found in mapping: {old_agent_id}')

    new_agent_id = agent_id_mapping[old_agent_id]
    return f'agent-{new_agent_id}.jsonl'


def apply_agent_id_mapping(json_str: str, agent_id_mapping: Mapping[str, str]) -> str:
    """
    Replace all old agent IDs with new agent IDs in serialized JSON.

    This catches:
    - References in isSidechain records pointing to agent files
    - Agent file path references in tool results
    - Any other string occurrences of agent IDs

    The short hex format (7-8 chars) makes false positives possible,
    so this should be applied AFTER slug mapping (which uses longer,
    more distinctive strings).

    Replacement is done in a single pass for efficiency.

    Args:
        json_str: Serialized JSON string
        agent_id_mapping: Mapping of old_agent_id -> new_agent_id

    Returns:
        JSON string with all agent IDs replaced
    """
    result = json_str
    for old_id, new_id in agent_id_mapping.items():
        result = result.replace(old_id, new_id)
    return result
