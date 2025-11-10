"""
Schema migration tools for handling Claude Code version changes.

These tools enable graceful handling of session files when Claude Code's
schema evolves between versions.
"""

from typing import Any, Callable
import json
import sys
from pathlib import Path


# ==============================================================================
# Migration Registry
# ==============================================================================

# Migrations are functions that transform record data from one version to another
Migration = Callable[[dict[str, Any]], dict[str, Any]]

# Registry mapping source_version -> target_version -> migration_function
MIGRATIONS: dict[tuple[str, str], Migration] = {}


def register_migration(from_version: str, to_version: str):
    """
    Decorator to register a migration function.

    Args:
        from_version: Source Claude Code version (e.g., '2.0.35')
        to_version: Target Claude Code version (e.g., '2.0.36')

    Example:
        @register_migration('2.0.35', '2.0.36')
        def migrate_2_0_35_to_2_0_36(record: dict) -> dict:
            # Add new field with default
            if 'newField' not in record:
                record['newField'] = None
            return record
    """

    def decorator(func: Migration) -> Migration:
        MIGRATIONS[(from_version, to_version)] = func
        return func

    return decorator


# ==============================================================================
# Version Detection
# ==============================================================================


def detect_record_version(record_data: dict[str, Any]) -> str | None:
    """
    Detect which Claude Code version created this record.

    Looks for version indicators in the record structure:
    - Presence/absence of specific fields
    - Field value patterns
    - Record structure

    Args:
        record_data: Raw record dictionary from JSONL

    Returns:
        Claude Code version string, or None if cannot determine

    Example:
        >>> record = {'type': 'user', 'version': '2.0.36', ...}
        >>> detect_record_version(record)
        '2.0.36'
    """
    # Check for explicit version field
    if 'version' in record_data:
        return record_data['version']

    # Check for version-specific field patterns
    # (Add heuristics as we learn about version differences)

    # Check for thinkingMetadata (added in 2.0.35)
    if 'thinkingMetadata' in record_data:
        return '2.0.35'  # Or later

    # Default: assume latest supported version
    return None


def needs_migration(record_data: dict[str, Any], target_version: str) -> bool:
    """
    Check if a record needs migration to target version.

    Args:
        record_data: Raw record dictionary
        target_version: Target Claude Code version

    Returns:
        True if migration needed, False otherwise
    """
    current_version = detect_record_version(record_data)

    if current_version is None:
        # Cannot determine version, assume it's current
        return False

    # Check if we have a migration path
    return (current_version, target_version) in MIGRATIONS


# ==============================================================================
# Migration Application
# ==============================================================================


def apply_migration(
    record_data: dict[str, Any], from_version: str, to_version: str
) -> dict[str, Any]:
    """
    Apply a specific migration to a record.

    Args:
        record_data: Raw record dictionary
        from_version: Source version
        to_version: Target version

    Returns:
        Migrated record data

    Raises:
        KeyError: If no migration exists for this version pair
    """
    migration_key = (from_version, to_version)

    if migration_key not in MIGRATIONS:
        raise KeyError(
            f'No migration found from {from_version} to {to_version}. '
            f'Available migrations: {list(MIGRATIONS.keys())}'
        )

    migration_func = MIGRATIONS[migration_key]
    return migration_func(record_data)


def auto_migrate(record_data: dict[str, Any], target_version: str) -> dict[str, Any]:
    """
    Automatically migrate a record to target version.

    Detects the current version and applies necessary migrations.

    Args:
        record_data: Raw record dictionary
        target_version: Target Claude Code version

    Returns:
        Migrated record data

    Example:
        >>> record = {'type': 'user', 'version': '2.0.35', ...}
        >>> migrated = auto_migrate(record, '2.0.36')
    """
    current_version = detect_record_version(record_data)

    if current_version is None:
        # Cannot determine version, return as-is
        return record_data

    if current_version == target_version:
        # Already at target version
        return record_data

    # For now, only support direct migrations
    # In the future, could support multi-step migration chains
    return apply_migration(record_data, current_version, target_version)


# ==============================================================================
# Batch Migration
# ==============================================================================


def migrate_session_file(
    input_path: Path, output_path: Path, target_version: str
) -> dict[str, int]:
    """
    Migrate an entire session JSONL file.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        target_version: Target Claude Code version

    Returns:
        Statistics dict with migration counts

    Example:
        >>> stats = migrate_session_file(
        ...     Path('old-session.jsonl'),
        ...     Path('new-session.jsonl'),
        ...     '2.0.36'
        ... )
        >>> print(f"Migrated {stats['migrated']} records")
    """
    stats = {
        'total': 0,
        'migrated': 0,
        'unchanged': 0,
        'errors': 0,
    }

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            if not line.strip():
                continue

            stats['total'] += 1

            try:
                record_data = json.loads(line)

                # Try to migrate
                if needs_migration(record_data, target_version):
                    record_data = auto_migrate(record_data, target_version)
                    stats['migrated'] += 1
                else:
                    stats['unchanged'] += 1

                # Write migrated record
                outfile.write(json.dumps(record_data) + '\n')

            except Exception as e:
                stats['errors'] += 1
                print(
                    f'Error migrating record at line {line_num}: {e}',
                    file=sys.stderr,
                )
                # Write original record on error
                outfile.write(line)

    return stats


# ==============================================================================
# Example Migrations
# ==============================================================================

# Example: When Claude Code 2.1 adds a new field
# @register_migration('2.0.36', '2.1.0')
# def migrate_2_0_36_to_2_1_0(record: dict[str, Any]) -> dict[str, Any]:
#     """Add contextWindow field with default value."""
#     if record.get('type') == 'user' and 'contextWindow' not in record:
#         record['contextWindow'] = None
#     return record


if __name__ == '__main__':
    # Demo
    print('=' * 80)
    print('Schema Migration Tools')
    print('=' * 80)
    print()

    print(f'Registered migrations: {len(MIGRATIONS)}')
    if MIGRATIONS:
        print('Available migration paths:')
        for (from_v, to_v), func in MIGRATIONS.items():
            print(f'  {from_v} → {to_v}: {func.__name__}')
    else:
        print('  (No migrations registered yet)')
    print()
    print('To add migrations, use the @register_migration decorator')
    print('Example:')
    print()
    print("  @register_migration('2.0.36', '2.1.0')")
    print('  def migrate_to_2_1(record: dict) -> dict:')
    print("      if 'newField' not in record:")
    print("          record['newField'] = None")
    print('      return record')