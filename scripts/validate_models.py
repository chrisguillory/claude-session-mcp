#!/usr/bin/env -S uv run --no-project
# /// script
# dependencies = ["pydantic>=2.0.0", "lazy-object-proxy>=1.10.0"]
# ///

"""
Validate Pydantic models against all Claude Code session files.

This script finds all .jsonl session files in ~/.claude/projects/*/ and validates
them against the Pydantic models to ensure complete schema coverage.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TypedDict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pydantic
from pydantic import ValidationError

from src.schemas.session import SessionRecordAdapter
from src.schemas.types import PermissiveModel


class FallbackUsage(TypedDict):
    """Information about a PermissiveModel fallback being used."""

    file: str  # Session filename
    line: int  # Line number in JSONL
    path: str  # Dot-separated path to the fallback
    fallback_type: str  # Class name (e.g., UnknownToolInput, UnknownToolResult)
    tool_name: str | None  # MCP tool name if available
    extra_fields: dict[str, str]  # Field name -> type name


class FileValidationResult(TypedDict):
    """Result of validating a single session file."""

    file: str
    total_records: int
    valid_records: int
    invalid_records: int
    record_types: Counter[str]
    errors: list[str]
    unknown_types: set[str]
    unknown_content_types: set[str]
    missing_fields: defaultdict[str, list[str]]
    fallbacks: list[FallbackUsage]


class TotalStats(TypedDict):
    """Aggregated statistics across all session files."""

    files: int
    total_records: int
    valid_records: int
    invalid_records: int
    record_types: Counter[str]
    unknown_types: set[str]
    unknown_content_types: set[str]
    fallbacks: list[FallbackUsage]


def find_all_session_files() -> list[Path]:
    """Find all .jsonl session files in ~/.claude/projects/*/"""
    claude_dir = Path.home() / '.claude' / 'projects'
    if not claude_dir.exists():
        return []

    session_files: list[Path] = []
    for project_dir in claude_dir.iterdir():
        if project_dir.is_dir():
            session_files.extend(project_dir.glob('*.jsonl'))

    return sorted(session_files)


# ==============================================================================
# Fallback Detection
# ==============================================================================


def find_fallbacks(
    obj: Any,
    path: str = '',
    tool_name: str | None = None,
) -> list[tuple[str, str, str | None, dict[str, str]]]:
    """
    Recursively find all PermissiveModel instances in a validated record.

    This detects where typed unions fell back to permissive fallback types
    (UnknownToolInput, UnknownToolResult).

    Args:
        obj: The object to search (typically a validated session record)
        path: Current dot-separated path for reporting
        tool_name: Tool name context (for tracking which MCP tool)

    Returns:
        List of tuples: (path, fallback_type, tool_name, extra_fields)
    """
    fallbacks: list[tuple[str, str, str | None, dict[str, str]]] = []

    if isinstance(obj, PermissiveModel):
        # Found a fallback! Record it with its extra fields
        extra = obj.get_extra_fields()
        fallbacks.append(
            (
                path or '(root)',
                type(obj).__name__,
                tool_name,
                {k: type(v).__name__ for k, v in extra.items()},
            )
        )

    if isinstance(obj, pydantic.BaseModel):
        # Recurse into Pydantic model fields
        for field_name in type(obj).model_fields:
            value = getattr(obj, field_name, None)
            if value is not None:
                child_path = f'{path}.{field_name}' if path else field_name
                # Track tool name for ToolUseContent
                current_tool_name = tool_name
                if field_name == 'input' and hasattr(obj, 'name'):
                    current_tool_name = getattr(obj, 'name', None)
                fallbacks.extend(find_fallbacks(value, child_path, current_tool_name))

    elif isinstance(obj, dict):
        # Recurse into dict values
        for key, value in obj.items():
            child_path = f'{path}.{key}' if path else str(key)
            fallbacks.extend(find_fallbacks(value, child_path, tool_name))

    elif isinstance(obj, (list, tuple)):
        # Recurse into sequence elements
        for i, item in enumerate(obj):
            child_path = f'{path}[{i}]' if path else f'[{i}]'
            fallbacks.extend(find_fallbacks(item, child_path, tool_name))

    return fallbacks


def validate_session_file(session_file: Path) -> FileValidationResult:
    """Validate a single session file and return statistics."""
    results: FileValidationResult = {
        'file': str(session_file),
        'total_records': 0,
        'valid_records': 0,
        'invalid_records': 0,
        'record_types': Counter(),
        'errors': [],
        'unknown_types': set(),
        'unknown_content_types': set(),
        'missing_fields': defaultdict(list),
        'fallbacks': [],
    }

    session_filename = Path(session_file).name

    with open(session_file) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            results['total_records'] += 1
            record_type = 'UNKNOWN'

            try:
                record_data = json.loads(line)
                record_type = record_data.get('type', 'UNKNOWN')
                results['record_types'][record_type] += 1

                # Try to parse with Pydantic
                record = SessionRecordAdapter.validate_python(record_data)
                results['valid_records'] += 1

                # Check for PermissiveModel fallbacks in the validated record
                for path, fb_type, tool_name, extra_fields in find_fallbacks(record):
                    results['fallbacks'].append(
                        {
                            'file': session_filename,
                            'line': line_num,
                            'path': path,
                            'fallback_type': fb_type,
                            'tool_name': tool_name,
                            'extra_fields': extra_fields,
                        }
                    )

            except ValidationError as e:
                # Check if this is a validator error about unmodeled tools
                is_validator_error = False
                for error in e.errors():
                    error_msg_str = str(error.get('ctx', {}).get('error', ''))
                    if 'Unmodeled Claude Code' in error_msg_str:
                        is_validator_error = True
                        # Surface this error clearly!
                        results['invalid_records'] += 1
                        error_msg = f'Line {line_num} ({record_type}): ⚠️  VALIDATOR ERROR: {error_msg_str}'
                        results['errors'].append(error_msg)
                        break

                if not is_validator_error:
                    # Regular validation error
                    results['invalid_records'] += 1
                    error_msg = f'Line {line_num} ({record_type}): {str(e)[:500]}'
                    results['errors'].append(error_msg)

            except Exception as e:
                results['invalid_records'] += 1
                error_msg = f'Line {line_num} ({record_type}): {str(e)[:500]}'
                results['errors'].append(error_msg)

                # Track unknown types and fields
                try:
                    if 'type' in record_data:
                        known_types = [
                            'user',
                            'assistant',
                            'summary',
                            'system',
                            'file-history-snapshot',
                            'queue-operation',
                        ]
                        if record_data['type'] not in known_types:
                            results['unknown_types'].add(record_data['type'])

                    # Track unknown content types
                    if record_type in ['user', 'assistant']:
                        msg = record_data.get('message', {})
                        content = msg.get('content', [])
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and 'type' in item:
                                    item_type = item['type']
                                    known_content_types = [
                                        'thinking',
                                        'text',
                                        'tool_use',
                                        'tool_result',
                                        'image',
                                    ]
                                    if item_type not in known_content_types:
                                        results['unknown_content_types'].add(item_type)

                    # Track missing fields
                    if 'Validation error' in str(e) or 'Field required' in str(e):
                        results['missing_fields'][record_type].append(str(e)[:150])
                except Exception:
                    pass  # Ignore errors in error handling

    return results


def main() -> None:
    print('=' * 80)
    print('Claude Code Session Model Validation')
    print('=' * 80)
    print()

    session_files = find_all_session_files()

    if not session_files:
        print('No session files found in ~/.claude/projects/*/')
        return

    print(f'Found {len(session_files)} session files')
    print()

    all_results: list[FileValidationResult] = []
    total_stats: TotalStats = {
        'files': 0,
        'total_records': 0,
        'valid_records': 0,
        'invalid_records': 0,
        'record_types': Counter(),
        'unknown_types': set(),
        'unknown_content_types': set(),
        'fallbacks': [],
    }

    for session_file in session_files:
        results = validate_session_file(session_file)
        all_results.append(results)

        total_stats['files'] += 1
        total_stats['total_records'] += results['total_records']
        total_stats['valid_records'] += results['valid_records']
        total_stats['invalid_records'] += results['invalid_records']
        total_stats['record_types'].update(results['record_types'])
        total_stats['unknown_types'].update(results['unknown_types'])
        total_stats['unknown_content_types'].update(results['unknown_content_types'])
        total_stats['fallbacks'].extend(results['fallbacks'])

    # Print summary
    print('SUMMARY')
    print('-' * 80)
    print(f'Total files processed: {total_stats["files"]}')
    print(f'Total records: {total_stats["total_records"]}')
    print(
        f'Valid records: {total_stats["valid_records"]} ({total_stats["valid_records"] / total_stats["total_records"] * 100:.1f}%)'
    )
    print(
        f'Invalid records: {total_stats["invalid_records"]} ({total_stats["invalid_records"] / total_stats["total_records"] * 100:.1f}%)'
    )
    print()

    print('Record types found:')
    for record_type, count in total_stats['record_types'].most_common():
        print(f'  {record_type}: {count}')
    print()

    if total_stats['unknown_types']:
        print('Unknown record types:')
        for unknown_type in sorted(total_stats['unknown_types']):
            print(f'  - {unknown_type}')
        print()

    if total_stats['unknown_content_types']:
        print('Unknown content types:')
        for unknown_type in sorted(total_stats['unknown_content_types']):
            print(f'  - {unknown_type}')
        print()

    # Print details for files with errors
    files_with_errors = [r for r in all_results if r['invalid_records'] > 0]
    if files_with_errors:
        print()
        print('FILES WITH VALIDATION ERRORS')
        print('-' * 80)
        for result in files_with_errors[:10]:  # Show first 10 files with errors
            print(f'\nFile: {Path(result["file"]).name}')
            print(
                f'  Total: {result["total_records"]}, Valid: {result["valid_records"]}, Invalid: {result["invalid_records"]}'
            )
            print(f'  Record types: {dict(result["record_types"])}')

            if result['errors']:
                print('  First 3 errors:')
                for error in result['errors'][:3]:
                    print(f'    - {error}')

    # Print PermissiveModel fallback usage (MCP tools)
    if total_stats['fallbacks']:
        print()
        print('PERMISSIVE MODEL FALLBACKS')
        print('-' * 80)
        print(f'⚠ {len(total_stats["fallbacks"])} records used fallback typing:')

        # Group fallbacks by type
        fallback_by_type: dict[str, list[FallbackUsage]] = defaultdict(list)
        for fb in total_stats['fallbacks']:
            fallback_by_type[fb['fallback_type']].append(fb)

        for fb_type, usages in sorted(fallback_by_type.items(), key=lambda x: -len(x[1])):
            print(f'\n  {fb_type}: {len(usages)} instance(s)')

            # Count tool name patterns (for MCP tools)
            tool_patterns: Counter[str] = Counter()
            for usage in usages:
                if usage['tool_name']:
                    tool_patterns[usage['tool_name']] += 1

            if tool_patterns:
                print('    Tool patterns:')
                for tool_name, count in tool_patterns.most_common(10):
                    print(f'      {tool_name} ({count}x)')
                if len(tool_patterns) > 10:
                    print(f'      ... and {len(tool_patterns) - 10} more tools')

    # Exit code based on validation success
    if total_stats['invalid_records'] == 0:
        print()
        print('✓ All records validated successfully!')
        sys.exit(0)
    else:
        print()
        print(f'✗ Validation failed for {total_stats["invalid_records"]} records')
        sys.exit(1)


if __name__ == '__main__':
    main()
