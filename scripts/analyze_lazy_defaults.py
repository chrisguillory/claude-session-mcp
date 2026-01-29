#!/usr/bin/env -S uv run
"""Comprehensive lazy default analysis for session schema.

Analyzes all optional fields across all record types to find:
1. How often each field is PRESENT vs ABSENT in JSON
2. When PRESENT, how often it's null vs non-null
3. Fields that are always null when present are candidates for removal

This helps identify:
- Fields that can be removed from schema entirely (always null)
- Fields that should be required (always present and non-null)
- Fields that need bifurcation (always present in some contexts)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any, TypedDict


# TypedDict for analysis results
class CandidateEntry(TypedDict):
    type: str
    field: str
    total_records: int
    present_count: int
    pct_present: float


class LowPresenceEntry(TypedDict):
    type: str
    field: str
    total_records: int
    present_count: int
    pct_present: float
    non_null_count: int
    pct_non_null: float
    samples: list[Any]


# ==============================================================================
# Configuration
# ==============================================================================

CLAUDE_PROJECTS_DIR = Path.home() / '.claude' / 'projects'


@dataclass
class FieldStats:
    """Statistics for a single field."""

    total_records: int = 0  # Records of this type seen
    present_count: int = 0  # Field present in JSON
    absent_count: int = 0  # Field absent from JSON (Pydantic fills None)
    null_when_present: int = 0  # Present with explicit null value
    non_null_when_present: int = 0  # Present with non-null value
    sample_values: list[Any] = field(default_factory=list)  # First few non-null values


@dataclass
class RecordTypeStats:
    """Statistics for all fields in a record type."""

    record_type: str
    total_records: int = 0
    fields: dict[str, FieldStats] = field(default_factory=dict)


def analyze_record(record: dict[str, Any], stats: dict[str, RecordTypeStats]) -> None:
    """Analyze a single record for lazy defaults."""
    record_type = record.get('type', '')

    # Handle nested message types specially
    if record_type in ('user', 'assistant') and 'message' in record:
        msg = record.get('message')
        if isinstance(msg, dict):
            role = msg.get('role', '')
            if role:
                analyze_message(msg, role, stats)

    # Get or create stats for this record type
    if record_type not in stats:
        stats[record_type] = RecordTypeStats(record_type=record_type)

    type_stats = stats[record_type]
    type_stats.total_records += 1

    # Analyze all fields in this record
    for field_name, value in record.items():
        if field_name not in type_stats.fields:
            type_stats.fields[field_name] = FieldStats()

        fs = type_stats.fields[field_name]
        fs.total_records += 1
        fs.present_count += 1

        if value is None:
            fs.null_when_present += 1
        else:
            fs.non_null_when_present += 1
            if len(fs.sample_values) < 3:
                # Store short repr of sample values
                sample = repr(value)[:80] if not isinstance(value, (dict, list)) else f'<{type(value).__name__}>'
                if sample not in fs.sample_values:
                    fs.sample_values.append(sample)


def analyze_message(msg: dict[str, Any], role: str, stats: dict[str, RecordTypeStats]) -> None:
    """Analyze a message object for lazy defaults."""
    msg_type = f'message:{role}'

    if msg_type not in stats:
        stats[msg_type] = RecordTypeStats(record_type=msg_type)

    type_stats = stats[msg_type]
    type_stats.total_records += 1

    # Analyze all fields in the message
    for field_name, value in msg.items():
        if field_name not in type_stats.fields:
            type_stats.fields[field_name] = FieldStats()

        fs = type_stats.fields[field_name]
        fs.total_records += 1
        fs.present_count += 1

        if value is None:
            fs.null_when_present += 1
        else:
            fs.non_null_when_present += 1
            if len(fs.sample_values) < 3:
                sample = repr(value)[:80] if not isinstance(value, (dict, list)) else f'<{type(value).__name__}>'
                if sample not in fs.sample_values:
                    fs.sample_values.append(sample)

    # Also analyze usage nested object if present
    if 'usage' in msg and isinstance(msg['usage'], dict):
        analyze_usage(msg['usage'], stats)


def analyze_usage(usage: dict[str, Any], stats: dict[str, RecordTypeStats]) -> None:
    """Analyze token usage object for lazy defaults."""
    usage_type = 'TokenUsage'

    if usage_type not in stats:
        stats[usage_type] = RecordTypeStats(record_type=usage_type)

    type_stats = stats[usage_type]
    type_stats.total_records += 1

    for field_name, value in usage.items():
        if field_name not in type_stats.fields:
            type_stats.fields[field_name] = FieldStats()

        fs = type_stats.fields[field_name]
        fs.total_records += 1
        fs.present_count += 1

        if value is None:
            fs.null_when_present += 1
        else:
            fs.non_null_when_present += 1
            if len(fs.sample_values) < 3:
                sample = repr(value)[:80] if not isinstance(value, (dict, list)) else f'<{type(value).__name__}>'
                if sample not in fs.sample_values:
                    fs.sample_values.append(sample)


def discover_sessions() -> list[Path]:
    """Discover all session files."""
    if not CLAUDE_PROJECTS_DIR.exists():
        return []

    sessions = [
        session_file
        for project_dir in CLAUDE_PROJECTS_DIR.iterdir()
        if project_dir.is_dir()
        for session_file in project_dir.glob('*.jsonl')
    ]

    return sorted(sessions)


def generate_models_docstring(
    stats: dict[str, RecordTypeStats],
    total_records: int,
    num_sessions: int,
) -> str:
    """Generate the LOW-INCIDENCE FIELDS TRACKING docstring section for models.py."""
    from datetime import datetime

    lines = [
        'LOW-INCIDENCE FIELDS TRACKING (run scripts/analyze_lazy_defaults.py to update):',
        f'Last analyzed: {datetime.now(UTC).strftime("%Y-%m-%d")} ({total_records:,} records across {num_sessions:,} sessions)',
        '',
    ]

    # Always-null fields
    always_null = []
    for type_name, type_stats in stats.items():
        for field_name, fs in type_stats.fields.items():
            if fs.present_count > 0 and fs.null_when_present == fs.present_count:
                pct_present = (fs.present_count / type_stats.total_records) * 100
                always_null.append((type_name, field_name, pct_present))

    if always_null:
        lines.append('Always-null fields (migration candidates):')
        for type_name, field_name, pct in sorted(always_null, key=lambda x: -x[2]):
            lines.append(f'- {type_name}.{field_name}: {pct:.1f}% present, ALWAYS NULL')
        lines.append('')

    # Group low-incidence fields by record type
    user_fields = []
    assistant_fields = []
    token_usage_fields = []

    for type_name, type_stats in stats.items():
        for field_name, fs in type_stats.fields.items():
            pct_present = (fs.present_count / type_stats.total_records) * 100 if type_stats.total_records > 0 else 0
            if 0 < pct_present < 5 and fs.non_null_when_present > 0:
                entry = (field_name, pct_present)
                if type_name == 'user':
                    user_fields.append(entry)
                elif type_name == 'assistant':
                    assistant_fields.append(entry)
                elif type_name == 'TokenUsage':
                    token_usage_fields.append(entry)

    if user_fields:
        lines.append('UserRecord low-incidence fields (<5% presence, have actual values):')
        for field_name, pct in sorted(user_fields, key=lambda x: x[1]):
            lines.append(f'- {field_name}: {pct:.2f}%')
        lines.append('')

    if assistant_fields:
        lines.append('AssistantRecord low-incidence fields (<5% presence, have actual values):')
        for field_name, pct in sorted(assistant_fields, key=lambda x: x[1]):
            lines.append(f'- {field_name}: {pct:.2f}%')
        lines.append('')

    if token_usage_fields:
        lines.append('TokenUsage low-incidence fields (<5% presence, have actual values):')
        for field_name, pct in sorted(token_usage_fields, key=lambda x: x[1]):
            lines.append(f'- {field_name}: {pct:.2f}%')
        lines.append('')

    return '\n'.join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description='Analyze lazy defaults in session schemas')
    parser.add_argument(
        '--output-docstring',
        action='store_true',
        help='Output models.py docstring section instead of full report',
    )
    args = parser.parse_args()

    sessions = discover_sessions()

    if not args.output_docstring:
        print('Lazy Default Analysis')
        print('=' * 80)
        print(f'Session files found: {len(sessions)}')
        print()

    stats: dict[str, RecordTypeStats] = {}
    total_records = 0

    for session in sessions:
        try:
            for line in session.read_text().splitlines():
                if not line.strip():
                    continue

                record = json.loads(line)
                analyze_record(record, stats)
                total_records += 1

        except Exception as e:
            print(f'Error processing {session.name}: {e}', file=sys.stderr)

    if args.output_docstring:
        print(generate_models_docstring(stats, total_records, len(sessions)))
        return

    print(f'Total records analyzed: {total_records:,}')
    print()

    # Report on lazy defaults
    print('=' * 80)
    print('LAZY DEFAULTS CANDIDATES (fields always null when present)')
    print('=' * 80)
    print()

    candidates: list[CandidateEntry] = []

    for type_name, type_stats in sorted(stats.items()):
        for field_name, fs in sorted(type_stats.fields.items()):
            # Skip fields that are always present and non-null (not optional)
            if fs.absent_count == 0 and fs.null_when_present == 0:
                continue

            # Find fields that are ALWAYS null when present
            if fs.present_count > 0 and fs.null_when_present == fs.present_count:
                pct_present = (fs.present_count / type_stats.total_records) * 100
                candidates.append(
                    CandidateEntry(
                        type=type_name,
                        field=field_name,
                        total_records=type_stats.total_records,
                        present_count=fs.present_count,
                        pct_present=pct_present,
                    )
                )

    # Sort by presence percentage (highest first) to prioritize
    candidates.sort(key=lambda x: -x['pct_present'])

    for c in candidates:
        print(f'{c["type"]}.{c["field"]}')
        print(f'  Records: {c["total_records"]:,}')
        print(f'  Present: {c["present_count"]:,} ({c["pct_present"]:.1f}%)')
        print('  Status: ALWAYS NULL when present - can be removed')
        print()

    print('=' * 80)
    print('LOW PRESENCE FIELDS (potential schema bloat)')
    print('=' * 80)
    print()

    low_presence: list[LowPresenceEntry] = []

    for type_name, type_stats in sorted(stats.items()):
        for field_name, fs in sorted(type_stats.fields.items()):
            # Find fields that are rarely present (< 5%) but sometimes non-null
            pct_present = (fs.present_count / type_stats.total_records) * 100 if type_stats.total_records > 0 else 0
            pct_non_null = (fs.non_null_when_present / fs.present_count) * 100 if fs.present_count > 0 else 0

            if 0 < pct_present < 5 and fs.non_null_when_present > 0:
                low_presence.append(
                    LowPresenceEntry(
                        type=type_name,
                        field=field_name,
                        total_records=type_stats.total_records,
                        present_count=fs.present_count,
                        pct_present=pct_present,
                        non_null_count=fs.non_null_when_present,
                        pct_non_null=pct_non_null,
                        samples=fs.sample_values,
                    )
                )

    low_presence.sort(key=lambda x: x['pct_present'])

    for lp in low_presence:
        print(f'{lp["type"]}.{lp["field"]}')
        print(f'  Records: {lp["total_records"]:,}')
        print(f'  Present: {lp["present_count"]:,} ({lp["pct_present"]:.2f}%)')
        print(f'  Non-null: {lp["non_null_count"]:,} ({lp["pct_non_null"]:.1f}% when present)')
        if lp['samples']:
            print(f'  Samples: {lp["samples"]}')
        print()

    print('=' * 80)
    print('POTENTIAL REQUIRED FIELDS (always present, always non-null)')
    print('=' * 80)
    print()

    # Find fields marked optional that might be required
    for type_name, type_stats in sorted(stats.items()):
        for field_name, fs in sorted(type_stats.fields.items()):
            # If present count equals total records and all non-null
            if (
                fs.present_count == type_stats.total_records
                and fs.non_null_when_present == fs.present_count
                and type_stats.total_records >= 100
            ):
                # Skip known required fields
                if field_name in ('type', 'uuid', 'timestamp', 'sessionId', 'role', 'content'):
                    continue
                print(f'{type_name}.{field_name}: Always present and non-null ({type_stats.total_records:,} records)')

    print()
    print('=' * 80)
    print('FULL FIELD BREAKDOWN BY TYPE')
    print('=' * 80)

    # Print detailed breakdown for key types
    key_types = ['assistant', 'user', 'message:assistant', 'message:user', 'TokenUsage', 'summary']

    for type_name in key_types:
        if type_name not in stats:
            continue

        type_stats = stats[type_name]
        print(f'\n{type_name} ({type_stats.total_records:,} records)')
        print('-' * 60)

        # Sort fields: always present first, then by presence %
        sorted_fields = sorted(
            type_stats.fields.items(),
            key=lambda x: (-x[1].present_count, x[0]),
        )

        for field_name, fs in sorted_fields:
            pct_present = (fs.present_count / type_stats.total_records) * 100 if type_stats.total_records > 0 else 0
            pct_null = (fs.null_when_present / fs.present_count) * 100 if fs.present_count > 0 else 0
            pct_non_null = 100 - pct_null

            status = ''
            if fs.present_count == type_stats.total_records:
                if fs.null_when_present == 0:
                    status = 'REQUIRED'
                elif fs.null_when_present == fs.present_count:
                    status = 'ALWAYS NULL'
                else:
                    status = f'{pct_non_null:.0f}% non-null'
            else:
                if fs.null_when_present == fs.present_count:
                    status = 'ALWAYS NULL when present'
                else:
                    status = f'{pct_non_null:.0f}% non-null when present'

            print(f'  {field_name}: {pct_present:.1f}% present, {status}')


if __name__ == '__main__':
    main()
