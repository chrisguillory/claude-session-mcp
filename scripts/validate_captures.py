#!/usr/bin/env -S uv run
"""
Validate capture schemas against all HTTP traffic capture files.

This script finds all .json capture files in the captures/ directory and validates
them against the Pydantic models to ensure complete schema coverage.

Usage:
    ./scripts/validate_captures.py [captures_dir]

If captures_dir is not specified, defaults to ./captures/
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import TypedDict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import ValidationError

from src.schemas.captures import (
    UnknownRequestCapture,
    UnknownResponseCapture,
    load_capture,
)


class FieldError(TypedDict):
    """Structured error info for a specific field."""

    loc: str  # Dot-separated path like "body.statsigMetadata.fallbackUrl"
    msg: str  # Error message
    type: str  # Error type like "extra_forbidden"


class CaptureValidationResult(TypedDict):
    """Result of validating captures in a session directory."""

    session_id: str
    total_captures: int
    typed_captures: int
    unknown_captures: int
    error_captures: int
    capture_types: Counter[str]
    errors: list[str]
    field_errors: list[FieldError]  # Structured error details


class TotalStats(TypedDict):
    """Aggregated statistics across all capture sessions."""

    sessions: int
    total_captures: int
    typed_captures: int
    unknown_captures: int
    error_captures: int
    capture_types: Counter[str]


def find_capture_sessions(captures_dir: Path) -> list[Path]:
    """Find all session directories in captures/"""
    if not captures_dir.exists():
        return []

    return sorted(item for item in captures_dir.iterdir() if item.is_dir() and not item.name.startswith('.'))


def validate_session_captures(session_dir: Path) -> CaptureValidationResult:
    """Validate all captures in a session directory."""
    results: CaptureValidationResult = {
        'session_id': session_dir.name,
        'total_captures': 0,
        'typed_captures': 0,
        'unknown_captures': 0,
        'error_captures': 0,
        'capture_types': Counter(),
        'errors': [],
        'field_errors': [],
    }

    for capture_file in sorted(session_dir.glob('*.json')):
        # Skip manifest files
        if 'manifest' in capture_file.name:
            continue

        results['total_captures'] += 1

        try:
            capture = load_capture(capture_file)
            capture_type = type(capture).__name__

            if isinstance(capture, (UnknownRequestCapture, UnknownResponseCapture)):
                results['unknown_captures'] += 1
                # Track which endpoints are unknown
                host = getattr(capture, 'host', 'unknown')
                path = getattr(capture, 'path', 'unknown')
                direction = getattr(capture, 'direction', 'unknown')
                results['capture_types'][f'unknown:{direction}:{host}{path[:50]}'] += 1
            else:
                results['typed_captures'] += 1
                results['capture_types'][capture_type] += 1

        except ValidationError as e:
            results['error_captures'] += 1
            error_msg = f'{capture_file.name}: {e.error_count()} errors'
            results['errors'].append(error_msg)

            # Extract structured field errors
            for err in e.errors():
                loc = '.'.join(str(x) for x in err['loc'])
                results['field_errors'].append(
                    {
                        'loc': loc,
                        'msg': err['msg'],
                        'type': err['type'],
                    }
                )

        except Exception as e:
            results['error_captures'] += 1
            error_msg = f'{capture_file.name}: {type(e).__name__}: {str(e)[:150]}'
            results['errors'].append(error_msg)

    return results


def main() -> None:
    print('=' * 80)
    print('Claude Code Capture Schema Validation')
    print('=' * 80)
    print()

    # Get captures directory from args or default
    if len(sys.argv) > 1:
        captures_dir = Path(sys.argv[1])
    else:
        captures_dir = Path(__file__).parent.parent / 'captures'

    session_dirs = find_capture_sessions(captures_dir)

    if not session_dirs:
        print(f'No capture sessions found in {captures_dir}')
        return

    print(f'Found {len(session_dirs)} capture sessions in {captures_dir}')
    print()

    all_results: list[CaptureValidationResult] = []
    total_stats: TotalStats = {
        'sessions': 0,
        'total_captures': 0,
        'typed_captures': 0,
        'unknown_captures': 0,
        'error_captures': 0,
        'capture_types': Counter(),
    }

    for session_dir in session_dirs:
        results = validate_session_captures(session_dir)
        all_results.append(results)

        total_stats['sessions'] += 1
        total_stats['total_captures'] += results['total_captures']
        total_stats['typed_captures'] += results['typed_captures']
        total_stats['unknown_captures'] += results['unknown_captures']
        total_stats['error_captures'] += results['error_captures']
        total_stats['capture_types'].update(results['capture_types'])

    # Print summary
    print('SUMMARY')
    print('-' * 80)
    print(f'Total sessions: {total_stats["sessions"]}')
    print(f'Total captures: {total_stats["total_captures"]}')

    if total_stats['total_captures'] > 0:
        typed_pct = total_stats['typed_captures'] / total_stats['total_captures'] * 100
        unknown_pct = total_stats['unknown_captures'] / total_stats['total_captures'] * 100
        error_pct = total_stats['error_captures'] / total_stats['total_captures'] * 100

        print(f'Typed captures: {total_stats["typed_captures"]} ({typed_pct:.1f}%)')
        print(f'Unknown captures: {total_stats["unknown_captures"]} ({unknown_pct:.1f}%)')
        print(f'Error captures: {total_stats["error_captures"]} ({error_pct:.1f}%)')
    print()

    # Print capture type breakdown
    typed_types = {k: v for k, v in total_stats['capture_types'].items() if not k.startswith('unknown:')}
    unknown_types = {k: v for k, v in total_stats['capture_types'].items() if k.startswith('unknown:')}

    if typed_types:
        print('Typed capture types:')
        for capture_type, count in sorted(typed_types.items(), key=lambda x: -x[1]):
            print(f'  {capture_type}: {count}')
        print()

    if unknown_types:
        print('Unknown endpoints:')
        for capture_type, count in sorted(unknown_types.items(), key=lambda x: -x[1]):
            # Parse unknown:direction:host/path
            parts = capture_type.split(':', 2)
            if len(parts) >= 3:
                direction, endpoint = parts[1], parts[2]
                print(f'  {direction:8} {endpoint}: {count}')
            else:
                print(f'  {capture_type}: {count}')
        print()

    # Print details for sessions with errors
    sessions_with_errors = [r for r in all_results if r['error_captures'] > 0]
    if sessions_with_errors:
        print('SESSIONS WITH VALIDATION ERRORS')
        print('-' * 80)
        for result in sessions_with_errors[:10]:
            print(f'\nSession: {result["session_id"]}')
            print(
                f'  Total: {result["total_captures"]}, '
                f'Typed: {result["typed_captures"]}, '
                f'Unknown: {result["unknown_captures"]}, '
                f'Errors: {result["error_captures"]}'
            )
            if result['errors']:
                print('  First 3 errors:')
                for error in result['errors'][:3]:
                    print(f'    - {error}')

    # Exit code based on validation success
    if total_stats['error_captures'] == 0 and total_stats['unknown_captures'] == 0:
        print()
        print('✓ All captures validated with full type coverage!')
        sys.exit(0)
    elif total_stats['error_captures'] == 0:
        print()
        print(f'⚠ All captures valid, but {total_stats["unknown_captures"]} use fallback types')
        sys.exit(0)
    else:
        print()
        print(f'✗ Validation failed for {total_stats["error_captures"]} captures')
        sys.exit(1)


if __name__ == '__main__':
    main()
