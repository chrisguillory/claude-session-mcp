#!/usr/bin/env -S uv run
"""Session file migration system.

Manages schema migrations for Claude Code session JSONL files.
Tracks applied migrations and enables incremental updates.

State and backups are stored in ~/.claude-session-mcp/ (not in ~/.claude/).
Backups are created by default before modifying any file.

Usage:
    uv run scripts/migrate_session_files.py status           # Show migration status
    uv run scripts/migrate_session_files.py list             # List available migrations
    uv run scripts/migrate_session_files.py run --dry-run    # Preview changes
    uv run scripts/migrate_session_files.py run              # Apply pending migrations
    uv run scripts/migrate_session_files.py run --migration ID  # Run specific migration
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

# =============================================================================
# Configuration
# =============================================================================

# State directory (outside ~/.claude/ to avoid conflicts)
BASE_DIR = Path.home() / '.claude-session-mcp'
STATE_DIR = BASE_DIR / 'migrations'
STATE_FILE = STATE_DIR / 'state.json'
LOG_FILE = STATE_DIR / 'log.jsonl'
BACKUP_DIR = BASE_DIR / 'backups'

# Claude Code session directory
CLAUDE_PROJECTS_DIR = Path.home() / '.claude' / 'projects'


# =============================================================================
# Migration Definitions
# =============================================================================


@dataclass
class Migration:
    """A schema migration definition."""

    id: str
    version: str
    description: str
    record_types: list[str]
    fields: list[str]
    operation: Literal['remove_null', 'rename', 'transform']
    # Optional transform function for complex migrations
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None


def remove_null_fields(record: dict[str, Any], fields: list[str]) -> tuple[dict[str, Any], list[str]]:
    """Remove fields that are explicitly null.

    Supports dot-notation for nested fields (e.g., 'message.container').
    """
    removed: list[str] = []
    for f in fields:
        if '.' in f:
            # Handle nested field paths
            parts = f.split('.')
            obj: dict[str, Any] | None = record
            for part in parts[:-1]:
                if obj is None or not isinstance(obj, dict) or part not in obj:
                    obj = None
                    break
                obj = obj[part]
            if obj is not None and isinstance(obj, dict):
                final_key = parts[-1]
                if final_key in obj and obj[final_key] is None:
                    del obj[final_key]
                    removed.append(f)
        else:
            # Simple top-level field
            if f in record and record[f] is None:
                del record[f]
                removed.append(f)
    return record, removed


# Migration registry - add new migrations here
MIGRATIONS: list[Migration] = [
    Migration(
        id='v0.2.8_remove_null_user_reserved',
        version='0.2.8',
        description='Remove explicit nulls for unused UserRecord fields (budgetTokens, mcp, projectPaths, skills)',
        record_types=['user'],
        fields=['budgetTokens', 'mcp', 'projectPaths', 'skills'],
        operation='remove_null',
    ),
    Migration(
        id='v0.2.8_remove_null_assistant_record_level',
        version='0.2.8',
        description='Remove explicit nulls for AssistantRecord fields now in message (usage, model, stopReason, requestDuration)',
        record_types=['assistant'],
        fields=['usage', 'model', 'stopReason', 'requestDuration'],
        operation='remove_null',
    ),
    Migration(
        id='v0.2.8_remove_null_assistant_bifurcated',
        version='0.2.8',
        description='Remove explicit nulls for bifurcated AssistantRecord fields (agentId, isApiErrorMessage, error, apiError)',
        record_types=['assistant'],
        fields=['agentId', 'isApiErrorMessage', 'error', 'apiError'],
        operation='remove_null',
    ),
    Migration(
        id='v0.2.8_remove_null_message_container',
        version='0.2.8',
        description='Remove explicit null container field from assistant messages (always null, wastes JSON bytes)',
        record_types=['assistant'],
        fields=['message.container'],
        operation='remove_null',
    ),
]


# =============================================================================
# State Management
# =============================================================================


@dataclass
class FileState:
    """Migration state for a single file."""

    path: str
    last_migrated: str
    migrations: list[str] = field(default_factory=list)


@dataclass
class MigrationState:
    """Overall migration state."""

    schema_version: str = '0.2.8'
    files: dict[str, FileState] = field(default_factory=dict)

    @classmethod
    def load(cls) -> MigrationState:
        """Load state from disk."""
        if not STATE_FILE.exists():
            return cls()

        data = json.loads(STATE_FILE.read_text())
        state = cls(schema_version=data.get('schema_version', '0.2.8'))

        for path, fdata in data.get('files', {}).items():
            state.files[path] = FileState(
                path=path,
                last_migrated=fdata.get('last_migrated', ''),
                migrations=fdata.get('migrations', []),
            )

        return state

    def save(self) -> None:
        """Save state to disk."""
        STATE_DIR.mkdir(parents=True, exist_ok=True)

        data = {
            'schema_version': self.schema_version,
            'files': {
                path: {
                    'last_migrated': fs.last_migrated,
                    'migrations': fs.migrations,
                }
                for path, fs in self.files.items()
            },
        }

        STATE_FILE.write_text(json.dumps(data, indent=2))

    def get_pending_migrations(self, file_path: str) -> list[Migration]:
        """Get migrations not yet applied to a file."""
        applied = set()
        if file_path in self.files:
            applied = set(self.files[file_path].migrations)

        return [m for m in MIGRATIONS if m.id not in applied]

    def mark_applied(self, file_path: str, migration_id: str) -> None:
        """Mark a migration as applied to a file."""
        now = datetime.now(UTC).isoformat()

        if file_path not in self.files:
            self.files[file_path] = FileState(path=file_path, last_migrated=now)

        fs = self.files[file_path]
        if migration_id not in fs.migrations:
            fs.migrations.append(migration_id)
        fs.last_migrated = now


def log_migration(file_path: str, migration_id: str, records_fixed: int, fields_removed: dict[str, int]) -> None:
    """Append to the migration log."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    entry = {
        'timestamp': datetime.now(UTC).isoformat(),
        'file': file_path,
        'migration_id': migration_id,
        'records_fixed': records_fixed,
        'fields_removed': fields_removed,
    }

    with LOG_FILE.open('a') as f:
        f.write(json.dumps(entry) + '\n')


# =============================================================================
# Backup Management
# =============================================================================


def create_backup(session_path: Path, run_timestamp: str, force: bool = False) -> Path:
    """Create a backup of a session file before migration.

    Backups are stored in a timestamped directory to group files from the same run.
    Structure: ~/.claude-session-mcp/backups/{run_timestamp}/{project_dir}/{filename}

    Args:
        session_path: Path to the session file to backup.
        run_timestamp: Timestamp for this migration run (groups related backups).
        force: If True, overwrite existing backups. If False, raise error.

    Returns:
        Path to the backup file.

    Raises:
        FileExistsError: If the backup file already exists and force=False.
    """
    # Preserve project directory structure in backup
    project_dir = session_path.parent.name
    backup_run_dir = BACKUP_DIR / run_timestamp / project_dir
    backup_run_dir.mkdir(parents=True, exist_ok=True)

    backup_path = backup_run_dir / session_path.name

    # Hard fail if backup already exists (unless force is set)
    if backup_path.exists() and not force:
        raise FileExistsError(
            f'Backup already exists at {backup_path}. '
            'This should not happen - refusing to overwrite. '
            'Use --force-backup to overwrite existing backups.'
        )

    shutil.copy2(session_path, backup_path)

    return backup_path


# =============================================================================
# Migration Execution
# =============================================================================


def apply_migration(
    session_path: Path,
    migration: Migration,
    dry_run: bool = True,
    verbose: bool = False,
) -> tuple[int, dict[str, int]]:
    """Apply a single migration to a session file.

    Returns:
        Tuple of (records_fixed, fields_removed_counts)
    """
    lines = session_path.read_text().splitlines()
    new_lines = []
    records_fixed = 0
    fields_removed: dict[str, int] = {}

    for i, line in enumerate(lines):
        if not line.strip():
            new_lines.append(line)
            continue

        try:
            record = json.loads(line)
            record_type = record.get('type', '')

            # Skip if not a target record type
            if record_type not in migration.record_types:
                new_lines.append(line)
                continue

            # Apply the operation
            if migration.operation == 'remove_null':
                fixed_record, removed = remove_null_fields(record, migration.fields)
            elif migration.transform:
                fixed_record = migration.transform(record)
                removed = []  # Custom transform handles its own tracking
            else:
                new_lines.append(line)
                continue

            if removed:
                records_fixed += 1
                for f in removed:
                    fields_removed[f] = fields_removed.get(f, 0) + 1

                if verbose:
                    print(f'    Line {i + 1}: removed {removed}')

                new_lines.append(json.dumps(fixed_record, separators=(',', ':')))
            else:
                new_lines.append(line)

        except json.JSONDecodeError:
            new_lines.append(line)

    if records_fixed > 0 and not dry_run:
        session_path.write_text('\n'.join(new_lines) + '\n')

    return records_fixed, fields_removed


def discover_sessions(project_filter: str | None = None) -> list[Path]:
    """Discover all session files to process."""
    if not CLAUDE_PROJECTS_DIR.exists():
        return []

    sessions = []
    for project_dir in CLAUDE_PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue

        if project_filter and project_filter not in project_dir.name:
            continue

        for session_file in project_dir.glob('*.jsonl'):
            if session_file.name.startswith('agent-'):
                continue
            sessions.append(session_file)

    return sorted(sessions)


# =============================================================================
# Commands
# =============================================================================


def cmd_status(args: argparse.Namespace) -> None:
    """Show migration status."""
    state = MigrationState.load()

    print('Migration Status')
    print('=' * 60)
    print(f'Schema version: {state.schema_version}')
    print(f'State directory: {STATE_DIR}')
    print()

    sessions = discover_sessions()
    print(f'Session files found: {len(sessions)}')

    # Count pending migrations
    pending_counts: dict[str, int] = {}
    for session in sessions:
        for migration in state.get_pending_migrations(str(session)):
            pending_counts[migration.id] = pending_counts.get(migration.id, 0) + 1

    if pending_counts:
        print()
        print('Pending migrations:')
        for mid, count in sorted(pending_counts.items()):
            print(f'  {mid}: {count} files')
    else:
        print()
        print('All files are up to date!')


def cmd_list(args: argparse.Namespace) -> None:
    """List available migrations."""
    print('Available Migrations')
    print('=' * 60)

    for m in MIGRATIONS:
        print(f'\n{m.id}')
        print(f'  Version: {m.version}')
        print(f'  Description: {m.description}')
        print(f'  Record types: {m.record_types}')
        print(f'  Fields: {m.fields}')
        print(f'  Operation: {m.operation}')


def cmd_run(args: argparse.Namespace) -> None:
    """Run migrations."""
    state = MigrationState.load()
    sessions = discover_sessions(args.project)

    # Create a timestamp for this run (used for backup directory)
    run_timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')

    print('Migration Run')
    print('=' * 60)
    print(f'Mode: {"DRY RUN" if args.dry_run else "LIVE"}')
    print(f'Backups: {"disabled" if args.no_backup else "enabled"}')
    if args.migration:
        print(f'Migration filter: {args.migration}')
    if args.project:
        print(f'Project filter: {args.project}')
    print(f'Sessions to process: {len(sessions)}')
    print()

    total_files_modified = 0
    total_records_fixed = 0
    total_fields_removed: dict[str, int] = {}
    backed_up_files: set[str] = set()  # Track files we've already backed up this run

    for session in sessions:
        session_str = str(session)
        pending = state.get_pending_migrations(session_str)

        if args.migration:
            pending = [m for m in pending if m.id == args.migration]

        if not pending:
            continue

        for migration in pending:
            records_fixed, fields_removed = apply_migration(
                session,
                migration,
                dry_run=args.dry_run,
                verbose=args.verbose,
            )

            if records_fixed > 0:
                # Create backup before first modification (unless disabled or dry run)
                if not args.dry_run and not args.no_backup and session_str not in backed_up_files:
                    backup_path = create_backup(session, run_timestamp, force=args.force_backup)
                    backed_up_files.add(session_str)
                    if args.verbose:
                        print(f'  Backed up to {backup_path}')

                total_files_modified += 1
                total_records_fixed += records_fixed

                for f, count in fields_removed.items():
                    total_fields_removed[f] = total_fields_removed.get(f, 0) + count

                action = 'Would apply' if args.dry_run else 'Applied'
                print(f'{action} {migration.id} to {session.name}: {records_fixed} records')

                if not args.dry_run:
                    state.mark_applied(session_str, migration.id)
                    log_migration(session_str, migration.id, records_fixed, fields_removed)

    if not args.dry_run:
        state.save()

    print()
    print('=' * 60)
    print('Summary')
    print('=' * 60)
    print(f'Files {"would be " if args.dry_run else ""}modified: {total_files_modified}')
    print(f'Records {"would be " if args.dry_run else ""}fixed: {total_records_fixed}')

    if backed_up_files:
        print(f'Files backed up: {len(backed_up_files)}')
        print(f'Backup location: {BACKUP_DIR / run_timestamp}')

    if total_fields_removed:
        print()
        print('Fields removed:')
        for f, count in sorted(total_fields_removed.items(), key=lambda x: -x[1]):
            print(f'  {f}: {count}')


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Session file migration system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # status command
    subparsers.add_parser('status', help='Show migration status')

    # list command
    subparsers.add_parser('list', help='List available migrations')

    # run command
    run_parser = subparsers.add_parser('run', help='Run migrations')
    run_parser.add_argument('--dry-run', action='store_true', help='Preview changes without modifying files')
    run_parser.add_argument('--no-backup', action='store_true', help='Skip creating backups (not recommended)')
    run_parser.add_argument('--force-backup', action='store_true', help='Overwrite existing backup files')
    run_parser.add_argument('--migration', type=str, help='Run only this specific migration')
    run_parser.add_argument('--project', type=str, help='Filter to specific project path')
    run_parser.add_argument('--verbose', action='store_true', help='Show details for each record')

    args = parser.parse_args()

    if args.command == 'status':
        cmd_status(args)
    elif args.command == 'list':
        cmd_list(args)
    elif args.command == 'run':
        cmd_run(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
