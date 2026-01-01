"""
Plan file handling for clone/restore operations.

Handles:
- Slug extraction from session records
- Plan file collection
- Clone slug generation (flat, not accumulating)
- Slug mapping application

Clone slug naming follows the same pattern as agent IDs:
- Native: adjective-verbing-noun (e.g., linked-twirling-tower)
- Cloned: adjective-verbing-noun-clone-XXXXXXXX (8-char session prefix)

When cloning a clone, we extract the base slug first to avoid accumulation:
- linked-twirling-tower-clone-019b5227 → linked-twirling-tower-clone-019b53d9
  (NOT linked-twirling-tower-clone-019b5227-clone-019b53d9)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from pathlib import Path

from src.models import (
    ApiErrorSystemRecord,
    AssistantRecord,
    CompactBoundarySystemRecord,
    LocalCommandSystemRecord,
    SessionRecord,
    UserRecord,
)

# Record types that have the slug field
SLUG_RECORD_TYPES = (
    UserRecord,
    AssistantRecord,
    LocalCommandSystemRecord,
    CompactBoundarySystemRecord,
    ApiErrorSystemRecord,
)


def extract_base_slug(slug: str) -> str:
    """
    Extract the base slug, removing any -clone-XXXXXXXX suffix.

    For native slugs, returns the slug unchanged.
    For cloned slugs, returns the portion before '-clone-'.

    Examples:
        'linked-twirling-tower' -> 'linked-twirling-tower'
        'linked-twirling-tower-clone-019b5227' -> 'linked-twirling-tower'

    This enables flat cloning: cloning a clone produces the same
    format as cloning a native session, just with a different suffix.

    Args:
        slug: Slug string (native or cloned format)

    Returns:
        Base slug without any clone suffix
    """
    if '-clone-' in slug:
        return slug.split('-clone-')[0]
    return slug


def extract_slugs_from_records(
    files_data: Mapping[str, Sequence[SessionRecord]],
) -> Set[str]:
    """
    Extract all unique slugs from session records.

    Uses structured extraction via Pydantic models, not regex.

    Args:
        files_data: Mapping of filename -> sequence of SessionRecord

    Returns:
        Set of unique slug strings found in records
    """
    slugs: set[str] = set()
    for records in files_data.values():
        for record in records:
            if isinstance(record, SLUG_RECORD_TYPES) and record.slug:
                slugs.add(record.slug)
    return slugs


def collect_plan_files(slugs: Set[str]) -> Mapping[str, str]:
    """
    Load plan file contents for slugs that have corresponding files.

    Slugs without plan files are normal (session never used plan mode,
    or file was auto-deleted after 30-day retention). These are silently
    skipped - NOT an error condition.

    Args:
        slugs: Set of slug strings extracted from records

    Returns:
        Mapping of slug -> file content (only for existing files)
    """
    plans_dir = Path.home() / '.claude' / 'plans'
    plan_files: dict[str, str] = {}
    for slug in slugs:
        plan_path = plans_dir / f'{slug}.md'
        if plan_path.exists():
            plan_files[slug] = plan_path.read_text(encoding='utf-8')
    return plan_files


def generate_clone_slug(old_slug: str, new_session_id: str) -> str:
    """
    Generate new slug with provenance.

    Format: {base_slug}-clone-{session_prefix}
    Example: curried-bubbling-wombat-clone-019b51bd

    When cloning a clone, we extract the base slug first to keep
    the naming flat rather than accumulating -clone- segments.

    Args:
        old_slug: Original slug string (may already be cloned)
        new_session_id: New session ID for the clone

    Returns:
        New slug string showing provenance
    """
    base_slug = extract_base_slug(old_slug)
    prefix = new_session_id[:8]
    return f'{base_slug}-clone-{prefix}'


def write_plan_files(
    plan_files: Mapping[str, str],
    new_session_id: str,
) -> Mapping[str, str]:
    """
    Write plan files to new locations with clone slugs.

    Args:
        plan_files: Mapping of old_slug -> content from archive/source
        new_session_id: New session ID for provenance

    Returns:
        Mapping of old_slug -> new_slug

    Raises:
        FileExistsError: If new plan file path already exists (slug collision)
    """
    plans_dir = Path.home() / '.claude' / 'plans'
    plans_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}

    for old_slug, content in plan_files.items():
        new_slug = generate_clone_slug(old_slug, new_session_id)
        new_path = plans_dir / f'{new_slug}.md'

        if new_path.exists():
            raise FileExistsError(
                f'Plan file already exists: {new_path}\nThis indicates a slug collision. Please investigate.'
            )

        new_path.write_text(content, encoding='utf-8')
        mapping[old_slug] = new_slug

    return mapping


def apply_slug_mapping(json_str: str, slug_mapping: Mapping[str, str]) -> str:
    """
    Replace all old slugs with new slugs in serialized JSON.

    This catches both:
    - "slug":"old-slug" field values
    - "plans/old-slug.md" path references
    - Any other string occurrences (enhancement queue refs, etc.)

    The slug format (adjective-verbing-noun) is distinctive enough
    that false positives are negligible.

    Args:
        json_str: Serialized JSON string
        slug_mapping: Mapping of old_slug -> new_slug

    Returns:
        JSON string with all slugs replaced
    """
    result = json_str
    for old_slug, new_slug in slug_mapping.items():
        result = result.replace(old_slug, new_slug)
    return result
