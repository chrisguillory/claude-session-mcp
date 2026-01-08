#!/usr/bin/env -S uv run
"""
Validate capture schemas against all HTTP traffic capture files.

This script finds all .json capture files in the captures/ directory and validates
them against the Pydantic models to ensure complete schema coverage.

Usage:
    ./scripts/validate_captures.py [OPTIONS] [captures_dir]

Options:
    --errors, -e     Show detailed error information for debugging
    --full, -f       Show complete values without truncation (implies --errors)
    --help, -h       Show this help message

Examples:
    ./scripts/validate_captures.py                    # Summary mode (default)
    ./scripts/validate_captures.py --errors           # Debugging mode with grouped errors
    ./scripts/validate_captures.py --errors --full    # Full values for AI/detailed analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TypedDict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pydantic

from src.schemas.captures import (
    UnknownRequestCapture,
    UnknownResponseCapture,
    load_capture,
)
from src.schemas.types import PermissiveModel

# ==============================================================================
# Terminal Color Support
# ==============================================================================


class Colors:
    """ANSI color codes for terminal output."""

    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

    @classmethod
    def disable(cls) -> None:
        """Disable all colors (for non-TTY output)."""
        cls.RED = ''
        cls.YELLOW = ''
        cls.BLUE = ''
        cls.CYAN = ''
        cls.GREEN = ''
        cls.BOLD = ''
        cls.DIM = ''
        cls.RESET = ''


# Disable colors if not a TTY
if not sys.stdout.isatty():
    Colors.disable()


# ==============================================================================
# Type Definitions
# ==============================================================================


class EnrichedFieldError(TypedDict):
    """Detailed error info including the actual value at the error path."""

    file: str  # Filename (not full path)
    file_path: Path  # Full path for reference
    loc: tuple[str | int, ...]  # Original location tuple
    loc_str: str  # Dot-separated path
    normalized_loc: str  # Path with Union[X,Y] patterns stripped
    generalized_loc: str  # Path with numeric indices replaced by *
    msg: str  # Error message
    error_type: str  # Error type like "extra_forbidden"
    value: Any  # The actual value at that path
    value_keys: list[str] | None  # Keys if value is a dict
    value_type: str  # Type name of the value


class ErrorGroup(TypedDict):
    """A group of related errors with the same pattern."""

    error_type: str
    generalized_path: str
    value_shape: str  # "object with keys [a, b, c]" or "string" etc.
    count: int
    errors: list[EnrichedFieldError]


class FallbackUsage(TypedDict):
    """Information about a PermissiveModel fallback being used."""

    file: str  # Filename
    path: str  # Dot-separated path to the fallback
    fallback_type: str  # Class name (e.g., UnknownConfigValue)
    extra_fields: dict[str, str]  # Field name -> type name


class CaptureValidationResult(TypedDict):
    """Result of validating captures in a session directory."""

    session_id: str
    total_captures: int
    typed_captures: int
    unknown_captures: int
    error_captures: int
    capture_types: Counter[str]
    errors: list[str]  # Simple error messages for summary mode
    enriched_errors: list[EnrichedFieldError]  # Detailed errors for --errors mode
    fallbacks: list[FallbackUsage]  # PermissiveModel instances found in captures


class TotalStats(TypedDict):
    """Aggregated statistics across all capture sessions."""

    sessions: int
    total_captures: int
    typed_captures: int
    unknown_captures: int
    error_captures: int
    capture_types: Counter[str]
    fallbacks: list[FallbackUsage]  # All PermissiveModel fallbacks found


# ==============================================================================
# Path Manipulation Utilities
# ==============================================================================


def normalize_path(loc: tuple[str | int, ...]) -> str:
    """
    Normalize a Pydantic error location path.

    Strips out Union[X, Y] and model name annotations that Pydantic adds
    for discriminated unions, producing a cleaner path for display.

    Example:
        ('body', 'data', 'Union[Cat, Dog]', 'Cat', 'age')
        -> 'body.data.age'
    """
    parts = []
    skip_next = False

    for part in loc:
        part_str = str(part)

        # Skip Union[...] annotations
        if part_str.startswith('Union['):
            skip_next = True  # Skip the following model name too
            continue

        # Skip model names that follow Union[...]
        if skip_next:
            skip_next = False
            continue

        # Skip function/validation wrapper names
        if part_str.startswith('function-'):
            continue

        parts.append(part_str)

    return '.'.join(parts)


def generalize_path(loc: tuple[str | int, ...]) -> str:
    """
    Generalize a path by replacing numeric indices with *.

    Used for grouping errors that occur at different array positions.

    Example:
        ('tools', 14, 'input_schema', 'properties', 'answers')
        -> 'tools.*.input_schema.properties.*'
    """
    normalized = normalize_path(loc)
    # Replace numeric path segments with *
    parts = normalized.split('.')
    generalized_parts = ['*' if part.isdigit() else part for part in parts]
    return '.'.join(generalized_parts)


# ==============================================================================
# Value Extraction and Formatting
# ==============================================================================


def extract_value_at_path(data: dict[str, Any], loc: tuple[str | int, ...]) -> Any:
    """
    Extract the value at a given path in a nested structure.

    Handles Pydantic's path annotations gracefully:
    - Skips Union[...] annotations
    - Skips type/model names that don't exist as keys (e.g., 'messages_request')
    - Skips function- prefixed validation wrappers

    Also handles the body wrapper transformation:
    - Raw JSON has: body.type, body.size, body.data
    - Pydantic sees: body (unwrapped to body.data)
    - When path says 'body', we look in body.data if body has type/data structure

    Returns the value at the path, or a dict with __missing__ info if not found.
    """
    current = data

    for i, part in enumerate(loc):
        part_str = str(part)

        # Skip Union[...] annotations
        if part_str.startswith('Union['):
            continue

        # Skip function/validation wrapper names
        if part_str.startswith('function-'):
            continue

        try:
            if isinstance(current, dict):
                if part_str in current:
                    current = current[part_str]

                    # Handle body wrapper: if we just entered 'body' and it has
                    # the wrapper structure {type, data, ...}, unwrap to data
                    if part_str == 'body' and isinstance(current, dict):
                        if 'type' in current and 'data' in current:
                            current = current.get('data', {})

                elif isinstance(part, int):
                    # Numeric key in a dict - unusual but handle it
                    if str(part) in current:
                        current = current[str(part)]
                    else:
                        return {'__missing__': part, '__parent_keys__': list(current.keys())}
                else:
                    # Key doesn't exist - might be a Pydantic type annotation
                    # Check if it looks like a model/type name (CamelCase or ends with known suffixes)
                    if (
                        part_str
                        and part_str[0].isupper()
                        or part_str.endswith('_request')
                        or part_str.endswith('_response')
                        or part_str.endswith('Schema')
                        or 'Schema' in part_str
                    ):
                        # Skip this - it's likely a type annotation
                        continue
                    # Path truly doesn't exist - return parent info
                    return {'__missing__': part_str, '__parent_keys__': list(current.keys())}

            elif isinstance(current, list):
                if isinstance(part, int) and 0 <= part < len(current):
                    current = current[part]
                else:
                    return {'__missing__': part, '__parent_length__': len(current)}
            else:
                # Can't traverse further
                return current

        except (KeyError, IndexError, TypeError):
            return None

    return current


def format_value_shape(value: Any) -> str:
    """
    Describe the shape of a value for grouping and display.

    Returns strings like:
        "object with keys [a, b, c]"
        "array with 5 elements"
        "string"
        "null"
    """
    if value is None:
        return 'null'
    elif isinstance(value, dict):
        if '__missing__' in value:
            return f'missing (parent has keys {sorted(value.get("__parent_keys__", []))})'
        keys = sorted(value.keys())
        if len(keys) <= 7:
            return f'object with keys {keys}'
        else:
            return f'object with {len(keys)} keys [{", ".join(keys[:5])}, ...]'
    elif isinstance(value, list):
        return f'array with {len(value)} elements'
    elif isinstance(value, bool):
        return f'boolean ({value})'
    elif isinstance(value, int):
        return 'integer'
    elif isinstance(value, float):
        return 'float'
    elif isinstance(value, str):
        if len(value) > 50:
            return f'string ({len(value)} chars)'
        return 'string'
    else:
        return type(value).__name__


def truncate_value(value: Any, full: bool = False) -> str:
    """
    Format a value for display, with optional truncation.

    Args:
        value: The value to format
        full: If True, don't truncate

    Truncation strategy:
        - Objects: Show up to 5 keys with brief values
        - Arrays: Show up to 3 elements
        - Strings: Show up to 200 chars
    """
    if full:
        return json.dumps(value, indent=2, ensure_ascii=False)

    if value is None:
        return 'null'
    elif isinstance(value, dict):
        if '__missing__' in value:
            parent_keys = value.get('__parent_keys__', [])
            return f'<missing field - parent has keys: {parent_keys}>'

        keys = list(value.keys())
        if len(keys) <= 5:
            # Show all keys with truncated values
            items = []
            for k in keys:
                v = value[k]
                v_str = json.dumps(v, ensure_ascii=False)
                if len(v_str) > 50:
                    v_str = v_str[:47] + '...'
                items.append(f'"{k}": {v_str}')
            return '{' + ', '.join(items) + '}'
        else:
            # Show first 5 keys
            items = []
            for k in keys[:5]:
                v = value[k]
                v_str = json.dumps(v, ensure_ascii=False)
                if len(v_str) > 30:
                    v_str = v_str[:27] + '...'
                items.append(f'"{k}": {v_str}')
            return '{' + ', '.join(items) + f', ... (+{len(keys) - 5} more keys)' + '}'

    elif isinstance(value, list):
        if len(value) <= 3:
            items = [json.dumps(v, ensure_ascii=False)[:50] for v in value]
            return '[' + ', '.join(items) + ']'
        else:
            items = [json.dumps(v, ensure_ascii=False)[:30] for v in value[:3]]
            return '[' + ', '.join(items) + f', ... (+{len(value) - 3} more)]'

    elif isinstance(value, str):
        if len(value) <= 200:
            return json.dumps(value, ensure_ascii=False)
        return json.dumps(value[:197] + '...', ensure_ascii=False)

    else:
        result = json.dumps(value, ensure_ascii=False)
        if len(result) > 200:
            return result[:197] + '...'
        return result


# ==============================================================================
# Error Grouping
# ==============================================================================


def compute_grouping_key(error: EnrichedFieldError) -> tuple[str, str, str]:
    """
    Compute the grouping key for an error.

    Groups by: (error_type, generalized_path, value_shape)
    """
    return (
        error['error_type'],
        error['generalized_loc'],
        format_value_shape(error['value']),
    )


def group_errors(errors: list[EnrichedFieldError]) -> list[ErrorGroup]:
    """
    Group errors by their pattern.

    Returns groups sorted by count (most frequent first).
    """
    groups: dict[tuple[str, str, str], list[EnrichedFieldError]] = defaultdict(list)

    for error in errors:
        key = compute_grouping_key(error)
        groups[key].append(error)

    result: list[ErrorGroup] = []
    for (error_type, gen_path, value_shape), group_errors_list in groups.items():
        result.append(
            {
                'error_type': error_type,
                'generalized_path': gen_path,
                'value_shape': value_shape,
                'count': len(group_errors_list),
                'errors': group_errors_list,
            }
        )

    # Sort by count descending
    result.sort(key=lambda g: -g['count'])
    return result


# ==============================================================================
# Fallback Detection
# ==============================================================================


def find_fallbacks(obj: Any, path: str = '') -> list[FallbackUsage]:
    """
    Recursively find all PermissiveModel instances in a validated capture.

    This detects where typed unions fell back to permissive fallback types
    (e.g., UnknownConfigValue, UnknownSegmentTraits).

    Args:
        obj: The object to search (typically a validated capture)
        path: Current dot-separated path for reporting

    Returns:
        List of FallbackUsage dicts describing each fallback found
    """
    fallbacks: list[FallbackUsage] = []

    if isinstance(obj, PermissiveModel):
        # Found a fallback! Record it with its extra fields
        extra = obj.get_extra_fields()
        fallbacks.append(
            {
                'file': '',  # Will be filled in by caller
                'path': path or '(root)',
                'fallback_type': type(obj).__name__,
                'extra_fields': {k: type(v).__name__ for k, v in extra.items()},
            }
        )

    if isinstance(obj, pydantic.BaseModel):
        # Recurse into Pydantic model fields
        for field_name in type(obj).model_fields:
            value = getattr(obj, field_name, None)
            if value is not None:
                child_path = f'{path}.{field_name}' if path else field_name
                fallbacks.extend(find_fallbacks(value, child_path))

    elif isinstance(obj, dict):
        # Recurse into dict values
        for key, value in obj.items():
            child_path = f'{path}.{key}' if path else str(key)
            fallbacks.extend(find_fallbacks(value, child_path))

    elif isinstance(obj, (list, tuple)):
        # Recurse into sequence elements
        for i, item in enumerate(obj):
            child_path = f'{path}[{i}]' if path else f'[{i}]'
            fallbacks.extend(find_fallbacks(item, child_path))

    return fallbacks


# ==============================================================================
# Validation with Value Extraction
# ==============================================================================


def validate_capture_with_values(
    capture_file: Path,
) -> tuple[str, Any, list[EnrichedFieldError]]:
    """
    Validate a capture file and extract values for any errors.

    Returns:
        ('success', capture_object, []) on success
        ('unknown', capture_object, []) for unknown types
        ('error', None, [enriched_errors]) on validation failure
        ('exception', error_msg, []) on other exceptions
    """
    # Load raw JSON for value extraction
    with open(capture_file) as f:
        raw_data = json.load(f)

    try:
        capture = load_capture(capture_file)

        if isinstance(capture, (UnknownRequestCapture, UnknownResponseCapture)):
            return ('unknown', capture, [])
        return ('success', capture, [])

    except pydantic.ValidationError as e:
        enriched_errors: list[EnrichedFieldError] = []

        for err in e.errors():
            loc = err['loc']
            value = extract_value_at_path(raw_data, loc)

            enriched_errors.append(
                {
                    'file': capture_file.name,
                    'file_path': capture_file,
                    'loc': loc,
                    'loc_str': '.'.join(str(x) for x in loc),
                    'normalized_loc': normalize_path(loc),
                    'generalized_loc': generalize_path(loc),
                    'msg': err['msg'],
                    'error_type': err['type'],
                    'value': value,
                    'value_keys': list(value.keys())
                    if isinstance(value, dict) and '__missing__' not in value
                    else None,
                    'value_type': type(value).__name__ if value is not None else 'null',
                }
            )

        return ('error', None, enriched_errors)

    except Exception as e:
        return ('exception', f'{type(e).__name__}: {str(e)[:150]}', [])


# ==============================================================================
# Session Validation
# ==============================================================================


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
        'enriched_errors': [],
        'fallbacks': [],
    }

    for capture_file in sorted(session_dir.glob('*.json')):
        # Skip manifest files
        if 'manifest' in capture_file.name:
            continue

        results['total_captures'] += 1

        status, result, enriched = validate_capture_with_values(capture_file)

        if status == 'success':
            results['typed_captures'] += 1
            results['capture_types'][type(result).__name__] += 1

            # Check for PermissiveModel fallbacks in the validated capture
            fallbacks = find_fallbacks(result)
            for fb in fallbacks:
                fb['file'] = capture_file.name
            results['fallbacks'].extend(fallbacks)

        elif status == 'unknown':
            results['unknown_captures'] += 1
            host = getattr(result, 'host', 'unknown')
            path = getattr(result, 'path', 'unknown')
            direction = getattr(result, 'direction', 'unknown')
            results['capture_types'][f'unknown:{direction}:{host}{path[:50]}'] += 1

        elif status == 'error':
            results['error_captures'] += 1
            results['errors'].append(f'{capture_file.name}: {len(enriched)} errors')
            results['enriched_errors'].extend(enriched)

        elif status == 'exception':
            results['error_captures'] += 1
            results['errors'].append(f'{capture_file.name}: {result}')

    return results


# ==============================================================================
# Output Formatting
# ==============================================================================


def format_error_group(group: ErrorGroup, full: bool = False) -> str:
    """Format an error group for display."""
    lines = []

    # Header with count, type, and path
    error_type = group['error_type']
    color = Colors.RED if 'forbidden' in error_type else Colors.YELLOW if 'missing' in error_type else Colors.BLUE

    lines.append(
        f'{Colors.BOLD}{group["count"]} error(s):{Colors.RESET} '
        f'{color}{error_type}{Colors.RESET} at '
        f'{Colors.CYAN}{group["generalized_path"]}{Colors.RESET}'
    )

    # Value shape
    lines.append(f'  {Colors.DIM}Shape:{Colors.RESET} {group["value_shape"]}')

    # Example
    example = group['errors'][0]
    lines.append(f'  {Colors.DIM}Example file:{Colors.RESET} {example["file"]}')
    lines.append(f'  {Colors.DIM}Example path:{Colors.RESET} {example["normalized_loc"]}')
    lines.append(f'  {Colors.DIM}Example value:{Colors.RESET}')

    # Format value with indentation
    value_str = truncate_value(example['value'], full=full)
    lines.extend(f'    {line}' for line in value_str.split('\n'))

    # Affected files
    all_files = sorted({e['file'] for e in group['errors']})
    if len(all_files) == 1:
        pass  # Already shown as example
    elif len(all_files) <= 10 or full:
        lines.append(f'  {Colors.DIM}Affected files ({len(all_files)}):{Colors.RESET}')
        lines.extend(f'    - {f}' for f in all_files)
    else:
        lines.append(f'  {Colors.DIM}Affected files ({len(all_files)}):{Colors.RESET}')
        lines.extend(f'    - {f}' for f in all_files[:10])
        lines.append(f'    ... and {len(all_files) - 10} more')

    return '\n'.join(lines)


def print_errors_mode(all_results: list[CaptureValidationResult], full: bool = False) -> None:
    """Print output in --errors mode: grouped errors first, summary at end."""

    # Collect all enriched errors
    all_errors: list[EnrichedFieldError] = []
    for result in all_results:
        all_errors.extend(result['enriched_errors'])

    if not all_errors:
        print(f'{Colors.GREEN}✓ No validation errors found!{Colors.RESET}')
        return

    # Group errors
    groups = group_errors(all_errors)

    # Print each group
    print(f'{Colors.BOLD}VALIDATION ERRORS{Colors.RESET}')
    print('=' * 80)
    print()

    for i, group in enumerate(groups):
        if i > 0:
            print()
            print('-' * 40)
            print()
        print(format_error_group(group, full=full))

    # Summary at end
    print()
    print('=' * 80)
    print(f'{Colors.BOLD}SUMMARY{Colors.RESET}')
    print('-' * 80)

    total_errors = sum(g['count'] for g in groups)
    affected_files = len({e['file'] for e in all_errors})

    print(f'Total errors: {Colors.RED}{total_errors}{Colors.RESET}')
    print(f'Unique patterns: {Colors.YELLOW}{len(groups)}{Colors.RESET}')
    print(f'Affected files: {Colors.CYAN}{affected_files}{Colors.RESET}')

    # Action summary
    if len(groups) == 1:
        print(f'\n{Colors.DIM}→ Fix 1 schema issue to resolve all {total_errors} errors{Colors.RESET}')
    else:
        print(f'\n{Colors.DIM}→ Fix {len(groups)} distinct schema issues{Colors.RESET}')


def print_summary_mode(
    all_results: list[CaptureValidationResult],
    total_stats: TotalStats,
) -> None:
    """Print output in summary mode (default): stats first, brief error list."""

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
            parts = capture_type.split(':', 2)
            if len(parts) >= 3:
                direction, endpoint = parts[1], parts[2]
                print(f'  {direction:8} {endpoint}: {count}')
            else:
                print(f'  {capture_type}: {count}')
        print()

    # Print brief error list
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

        print()
        print(f'{Colors.DIM}Tip: Use --errors for detailed error information{Colors.RESET}')

    # Print PermissiveModel fallback usage
    if total_stats['fallbacks']:
        print()
        print('PERMISSIVE MODEL FALLBACKS')
        print('-' * 80)
        print(f'{Colors.YELLOW}⚠ {len(total_stats["fallbacks"])} captures used fallback typing:{Colors.RESET}')

        # Group fallbacks by type
        fallback_by_type: dict[str, list[FallbackUsage]] = defaultdict(list)
        for fb in total_stats['fallbacks']:
            fallback_by_type[fb['fallback_type']].append(fb)

        for fb_type, usages in sorted(fallback_by_type.items(), key=lambda x: -len(x[1])):
            print(f'\n  {Colors.CYAN}{fb_type}{Colors.RESET}: {len(usages)} instance(s)')

            # Show unique field patterns
            field_patterns: Counter[str] = Counter()
            for usage in usages:
                if usage['extra_fields']:
                    pattern = ', '.join(f'{k}: {v}' for k, v in sorted(usage['extra_fields'].items()))
                    field_patterns[pattern] += 1

            if field_patterns:
                print(f'    {Colors.DIM}Field patterns:{Colors.RESET}')
                for pattern, count in field_patterns.most_common(5):
                    print(f'      {{{pattern}}} ({count}x)')
                if len(field_patterns) > 5:
                    print(f'      ... and {len(field_patterns) - 5} more patterns')
            else:
                print(f'    {Colors.DIM}(empty fallbacks - no extra fields){Colors.RESET}')


# ==============================================================================
# Main
# ==============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Validate capture schemas against HTTP traffic capture files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Summary mode (default)
  %(prog)s --errors             Debugging mode with grouped errors
  %(prog)s --errors --full      Full values for AI/detailed analysis
  %(prog)s -ef                  Short form of --errors --full
        """,
    )

    parser.add_argument(
        'captures_dir',
        nargs='?',
        type=Path,
        default=Path(__file__).parent.parent / 'captures',
        help='Directory containing capture sessions (default: ./captures/)',
    )

    parser.add_argument(
        '-e',
        '--errors',
        action='store_true',
        help='Show detailed error information with grouping and actual values',
    )

    parser.add_argument(
        '-f',
        '--full',
        action='store_true',
        help='Show complete values without truncation (implies --errors)',
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --full implies --errors
    if args.full:
        args.errors = True

    print('=' * 80)
    print('Claude Code Capture Schema Validation')
    print('=' * 80)
    print()

    session_dirs = find_capture_sessions(args.captures_dir)

    if not session_dirs:
        print(f'No capture sessions found in {args.captures_dir}')
        return

    print(f'Found {len(session_dirs)} capture sessions in {args.captures_dir}')
    print()

    # Validate all sessions
    all_results: list[CaptureValidationResult] = []
    total_stats: TotalStats = {
        'sessions': 0,
        'total_captures': 0,
        'typed_captures': 0,
        'unknown_captures': 0,
        'error_captures': 0,
        'capture_types': Counter(),
        'fallbacks': [],
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
        total_stats['fallbacks'].extend(results['fallbacks'])

    # Output based on mode
    if args.errors:
        print_errors_mode(all_results, full=args.full)
    else:
        print_summary_mode(all_results, total_stats)

    # Exit code
    if total_stats['error_captures'] == 0 and total_stats['unknown_captures'] == 0:
        print()
        print(f'{Colors.GREEN}✓ All captures validated with full type coverage!{Colors.RESET}')
        sys.exit(0)
    elif total_stats['error_captures'] == 0:
        print()
        print(
            f'{Colors.YELLOW}⚠ All captures valid, but {total_stats["unknown_captures"]} use fallback types{Colors.RESET}'
        )
        sys.exit(0)
    else:
        print()
        print(f'{Colors.RED}✗ Validation failed for {total_stats["error_captures"]} captures{Colors.RESET}')
        sys.exit(1)


if __name__ == '__main__':
    main()
