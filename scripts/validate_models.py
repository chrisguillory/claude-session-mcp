#!/usr/bin/env -S uv run --no-project
# /// script
# dependencies = ["pydantic>=2.0.0", "lazy-object-proxy>=1.10.0", "orjson>=3.10.0"]
# ///

"""
Validate Pydantic models against all Claude Code session files.

This script finds all .jsonl session files in ~/.claude/projects/*/ and validates
them against the Pydantic models to ensure complete schema coverage.

Usage:
    ./scripts/validate_models.py [OPTIONS] [file]

Options:
    --errors, -e     Show detailed error information with grouping and actual values
    --full, -f       Show complete values without truncation (implies --errors)
    --fast           Fast mode: skip fallback detection and error enrichment
    -j N, --workers N  Number of parallel workers (default: min(cpu_count, 8))
    --help, -h       Show this help message

Examples:
    ./scripts/validate_models.py                    # Summary mode (default)
    ./scripts/validate_models.py --errors           # Debugging mode with grouped errors
    ./scripts/validate_models.py --errors --full    # Full values for AI/detailed analysis
    ./scripts/validate_models.py --fast             # Quick pass/fail validation
    ./scripts/validate_models.py --fast -j 4        # Quick validation, 4 workers
    ./scripts/validate_models.py -j 1              # Sequential (no parallelism)
    ./scripts/validate_models.py -e path/to/file.jsonl  # Investigate specific file
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypedDict

import orjson

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pydantic
from pydantic import ValidationError

from src.schemas.session.models import (
    AssistantRecord,
    ToolResultContent,
    ToolUseContent,
    UserRecord,
    validate_session_record,
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
    line: int  # Line number in JSONL
    record_type: str  # Record type (user, assistant, etc.)
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

    file: str  # Session filename
    line: int  # Line number in JSONL
    path: str  # Dot-separated path to the fallback
    fallback_type: str  # Class name (e.g., MCPToolInput, MCPToolResult)
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
    enriched_errors: list[EnrichedFieldError]
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


# ==============================================================================
# Path Manipulation Utilities
# ==============================================================================


def normalize_path(loc: tuple[str | int, ...]) -> str:
    """
    Normalize a Pydantic error location path.

    Strips out Union[X, Y] and model name annotations that Pydantic adds
    for discriminated unions, producing a cleaner path for display.

    Example:
        ('message', 'content', 'Union[ThinkingContent, TextContent]', 'ToolUseContent', 'input')
        -> 'message.content.input'
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
        ('message', 'content', 2, 'input', 'pattern')
        -> 'message.content.*.input.pattern'
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
    - Skips type/model names that don't exist as keys
    - Skips function- prefixed validation wrappers

    Returns the value at the path, or a dict with __missing__ info if not found.
    """
    current = data

    for part in loc:
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
                elif isinstance(part, int):
                    # Numeric key in a dict - unusual but handle it
                    if str(part) in current:
                        current = current[str(part)]
                    else:
                        return {'__missing__': part, '__parent_keys__': list(current.keys())}
                else:
                    # Key doesn't exist - might be a Pydantic type annotation
                    if part_str and (
                        part_str[0].isupper()
                        or part_str.endswith('Record')
                        or part_str.endswith('Content')
                        or 'Tool' in part_str
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


def _may_contain_fallbacks(record: Any) -> bool:
    """Quick pre-check: can this record possibly contain PermissiveModel instances?

    Only user records with tool_result content and assistant records with
    tool_use content can contain PermissiveModel fallbacks. All other record
    types (system, summary, queue-operation, progress, etc.) cannot.

    This avoids the expensive recursive walk of find_fallbacks() for ~67% of records.
    """
    if isinstance(record, AssistantRecord):
        if record.message and isinstance(record.message.content, list):
            return any(isinstance(block, ToolUseContent) for block in record.message.content)
        return False

    if isinstance(record, UserRecord):
        if record.mcpMeta is not None:
            return True
        if record.message and isinstance(record.message.content, list):
            if any(isinstance(block, ToolResultContent) for block in record.message.content):
                return True
        return record.toolUseResult is not None

    return False


def find_fallbacks(
    obj: Any,
    path: str = '',
    tool_name: str | None = None,
) -> list[tuple[str, str, str | None, dict[str, str]]]:
    """
    Recursively find all PermissiveModel instances in a validated record.

    This detects where typed unions fell back to permissive fallback types
    (MCPToolInput, MCPToolResult).

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


def resolve_tool_name_for_result(
    record: UserRecord,
    tool_use_map: Mapping[str, str],
) -> str | None:
    """
    Look up tool name for a UserRecord's tool result.

    Uses sourceToolUseID or scans ToolResultContent blocks to find the
    corresponding tool_use_id, then looks it up in the map.

    Args:
        record: The UserRecord containing toolUseResult
        tool_use_map: Mapping of tool_use_id -> tool_name from AssistantRecords

    Returns:
        Tool name if found, None otherwise
    """
    # Try direct field first (most reliable)
    if record.sourceToolUseID and record.sourceToolUseID in tool_use_map:
        return tool_use_map[record.sourceToolUseID]

    # Try scanning ToolResultContent blocks in message
    if record.message and isinstance(record.message.content, list):
        for block in record.message.content:
            if not isinstance(block, ToolResultContent):
                continue
            if block.tool_use_id in tool_use_map:
                return tool_use_map[block.tool_use_id]

    return None


# ==============================================================================
# File Validation
# ==============================================================================


def find_all_session_files() -> list[Path]:
    """Find all .jsonl session files in ~/.claude/projects/ (recursive)."""
    claude_dir = Path.home() / '.claude' / 'projects'
    if not claude_dir.exists():
        return []

    session_files: list[Path] = []
    for project_dir in claude_dir.iterdir():
        if project_dir.is_dir():
            # Use recursive glob to include subagent files in subagents/ subdirectory
            session_files.extend(project_dir.glob('**/*.jsonl'))

    return sorted(session_files)


def validate_session_file(session_file: Path, *, fast: bool = False) -> FileValidationResult:
    """Validate a single session file and return statistics."""
    results: FileValidationResult = {
        'file': str(session_file),
        'total_records': 0,
        'valid_records': 0,
        'invalid_records': 0,
        'record_types': Counter(),
        'errors': [],
        'enriched_errors': [],
        'unknown_types': set(),
        'unknown_content_types': set(),
        'missing_fields': defaultdict(list),
        'fallbacks': [],
    }

    session_filename = Path(session_file).name

    # Track tool_use_id -> tool_name for cross-record correlation
    tool_use_map: dict[str, str] = {}

    with open(session_file, 'rb') as f:
        for line_num, line in enumerate(f, 1):
            if not line or line == b'\n':
                continue

            results['total_records'] += 1
            record_type = 'UNKNOWN'

            try:
                record_data = orjson.loads(line)
                record_type = record_data.get('type', 'UNKNOWN')
                results['record_types'][record_type] += 1

                # Validate using type-dispatch (avoids 17-member union scan)
                record = validate_session_record(record_data)
                results['valid_records'] += 1

                if not fast:
                    # Extract tool uses from AssistantRecords for correlation
                    if isinstance(record, AssistantRecord) and record.message:
                        if isinstance(record.message.content, list):
                            for block in record.message.content:
                                if isinstance(block, ToolUseContent):
                                    tool_use_map[block.id] = block.name

                    # Check for PermissiveModel fallbacks (only tool-containing records)
                    if _may_contain_fallbacks(record):
                        reclassified_as_invalid = False
                        for path, fb_type, tool_name, extra_fields in find_fallbacks(record):
                            # For MCPToolResult, check if it's actually a Claude Code tool
                            if fb_type == 'MCPToolResult' and isinstance(record, UserRecord):
                                actual_tool_name = resolve_tool_name_for_result(record, tool_use_map)
                                if actual_tool_name and not actual_tool_name.startswith('mcp__'):
                                    if not reclassified_as_invalid:
                                        # Claude Code tool fell through - reclassify once
                                        results['invalid_records'] += 1
                                        results['valid_records'] -= 1
                                        reclassified_as_invalid = True
                                    error_msg = (
                                        f'Line {line_num} ({record_type}): ⚠️  Claude Code tool '
                                        f"'{actual_tool_name}' result fell through to MCPToolResult. "
                                        f'Fields: {list(extra_fields.keys())}'
                                    )
                                    results['errors'].append(error_msg)
                                    continue  # Don't also add to fallbacks

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
                results['invalid_records'] += 1

                if fast:
                    results['errors'].append(f'Line {line_num} ({record_type}): {len(e.errors())} validation errors')
                else:
                    # Check if this is a validator error about unmodeled tools
                    is_validator_error = False
                    for error in e.errors():
                        error_msg_str = str(error.get('ctx', {}).get('error', ''))
                        if 'fell through to MCPToolInput' in error_msg_str:
                            is_validator_error = True
                            error_msg = f'Line {line_num} ({record_type}): ⚠️  VALIDATOR ERROR: {error_msg_str}'
                            results['errors'].append(error_msg)
                            break

                    if not is_validator_error:
                        # Regular validation error - extract enriched info
                        error_msg = f'Line {line_num} ({record_type}): {len(e.errors())} validation errors'
                        results['errors'].append(error_msg)

                        for err in e.errors():
                            loc = err['loc']
                            value = extract_value_at_path(record_data, loc)

                            results['enriched_errors'].append(
                                {
                                    'file': session_filename,
                                    'file_path': session_file,
                                    'line': line_num,
                                    'record_type': record_type,
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

            except Exception as e:
                results['invalid_records'] += 1
                error_msg = f'Line {line_num} ({record_type}): {str(e)[:500]}'
                results['errors'].append(error_msg)

                if not fast:
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
                                'custom-title',
                                'progress',
                                'pr-link',
                                'saved_hook_context',
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
                                            'tool_reference',
                                            'image',
                                            'document',
                                        ]
                                        if item_type not in known_content_types:
                                            results['unknown_content_types'].add(item_type)

                        # Track missing fields
                        if 'Validation error' in str(e) or 'Field required' in str(e):
                            results['missing_fields'][record_type].append(str(e)[:150])
                    except Exception:
                        pass  # Ignore errors in error handling

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
    lines.append(f'  {Colors.DIM}Example file:{Colors.RESET} {example["file"]}:{example["line"]}')
    lines.append(f'  {Colors.DIM}Record type:{Colors.RESET} {example["record_type"]}')
    lines.append(f'  {Colors.DIM}Example path:{Colors.RESET} {example["normalized_loc"]}')
    lines.append(f'  {Colors.DIM}Error message:{Colors.RESET} {example["msg"]}')
    lines.append(f'  {Colors.DIM}Example value:{Colors.RESET}')

    # Format value with indentation
    value_str = truncate_value(example['value'], full=full)
    lines.extend(f'    {line}' for line in value_str.split('\n'))

    # Affected files
    all_files = sorted({f'{e["file"]}:{e["line"]}' for e in group['errors']})
    if len(all_files) == 1:
        pass  # Already shown as example
    elif len(all_files) <= 10 or full:
        lines.append(f'  {Colors.DIM}Affected locations ({len(all_files)}):{Colors.RESET}')
        lines.extend(f'    - {f}' for f in all_files)
    else:
        lines.append(f'  {Colors.DIM}Affected locations ({len(all_files)}):{Colors.RESET}')
        lines.extend(f'    - {f}' for f in all_files[:10])
        lines.append(f'    ... and {len(all_files) - 10} more')

    return '\n'.join(lines)


def print_errors_mode(all_results: list[FileValidationResult], full: bool = False) -> None:
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
    all_results: list[FileValidationResult],
    total_stats: TotalStats,
) -> None:
    """Print output in summary mode (default): stats first, brief error list."""

    print('SUMMARY')
    print('-' * 80)
    print(f'Total files processed: {total_stats["files"]}')
    print(f'Total records: {total_stats["total_records"]}')

    valid_pct = _format_pct(total_stats['valid_records'], total_stats['total_records'], floor=True)
    invalid_pct = _format_pct(total_stats['invalid_records'], total_stats['total_records'], floor=False)
    print(f'Valid records: {total_stats["valid_records"]} ({valid_pct:.2f}%)')
    print(f'Invalid records: {total_stats["invalid_records"]} ({invalid_pct:.2f}%)')
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

        if total_stats['invalid_records'] > 0:
            print()
            print(f'{Colors.DIM}Tip: Use --errors for detailed error information{Colors.RESET}')

    # Print PermissiveModel fallback usage (MCP tools)
    if total_stats['fallbacks']:
        print()
        print('PERMISSIVE MODEL FALLBACKS')
        print('-' * 80)
        print(f'{Colors.YELLOW}⚠ {len(total_stats["fallbacks"])} records used fallback typing:{Colors.RESET}')

        # Group fallbacks by type
        fallback_by_type: dict[str, list[FallbackUsage]] = defaultdict(list)
        for fb in total_stats['fallbacks']:
            fallback_by_type[fb['fallback_type']].append(fb)

        for fb_type, usages in sorted(fallback_by_type.items(), key=lambda x: -len(x[1])):
            print(f'\n  {Colors.CYAN}{fb_type}{Colors.RESET}: {len(usages)} instance(s)')

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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Validate session schemas against Claude Code session files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Summary mode (default)
  %(prog)s --errors             Debugging mode with grouped errors
  %(prog)s --errors --full      Full values for AI/detailed analysis
  %(prog)s -ef                  Short form of --errors --full
  %(prog)s --fast               Quick pass/fail validation
  %(prog)s --fast -j 4          Quick validation, 4 workers
  %(prog)s -j 1                 Sequential (no parallelism)
  %(prog)s -e path/to/file.jsonl   Investigate specific file
        """,
    )

    parser.add_argument(
        'file',
        nargs='?',
        type=Path,
        default=None,
        help='Specific .jsonl file to validate (default: all files in ~/.claude/projects/)',
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

    parser.add_argument(
        '--fast',
        action='store_true',
        help='Fast mode: skip fallback detection and error enrichment (just validate counts)',
    )

    parser.add_argument(
        '-j',
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: min(cpu_count, 8), use 1 for sequential)',
    )

    args = parser.parse_args()

    if args.fast and (args.errors or args.full):
        parser.error('--fast cannot be combined with --errors or --full')

    return args


# ==============================================================================
# Main
# ==============================================================================


def _aggregate_result(total_stats: TotalStats, results: FileValidationResult) -> None:
    """Aggregate a single file's results into total stats."""
    total_stats['files'] += 1
    total_stats['total_records'] += results['total_records']
    total_stats['valid_records'] += results['valid_records']
    total_stats['invalid_records'] += results['invalid_records']
    total_stats['record_types'].update(results['record_types'])
    total_stats['unknown_types'].update(results['unknown_types'])
    total_stats['unknown_content_types'].update(results['unknown_content_types'])
    total_stats['fallbacks'].extend(results['fallbacks'])


def main() -> None:
    args = parse_args()

    # --full implies --errors
    if args.full:
        args.errors = True

    print('=' * 80)
    print('Claude Code Session Model Validation')
    print('=' * 80)
    print()

    # Find session files
    if args.file:
        if not args.file.exists():
            print(f'Error: File not found: {args.file}')
            sys.exit(1)
        session_files = [args.file]
        print(f'Validating single file: {args.file}')
    else:
        session_files = find_all_session_files()
        if not session_files:
            print('No session files found in ~/.claude/projects/*/')
            return
        print(f'Found {len(session_files)} session files')
    print()

    # Validate all files
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

    max_workers = args.workers if args.workers is not None else min(os.cpu_count() or 4, 8)

    if args.file or len(session_files) <= 1 or max_workers <= 1:
        # Sequential validation (single file, or -j 1)
        for session_file in session_files:
            results = validate_session_file(session_file, fast=args.fast)
            all_results.append(results)
            _aggregate_result(total_stats, results)
    else:
        # Parallel validation across files
        completed = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(validate_session_file, f, fast=args.fast): f for f in session_files}

            for future in as_completed(future_to_file):
                results = future.result()
                all_results.append(results)
                _aggregate_result(total_stats, results)

                completed += 1
                if sys.stdout.isatty():
                    print(
                        f'\r  Validated {completed}/{len(session_files)} files...',
                        end='',
                        flush=True,
                    )

        if sys.stdout.isatty():
            print()  # Newline after progress

        # Sort for deterministic output (as_completed returns in completion order)
        all_results.sort(key=lambda r: r['file'])

    # Output based on mode
    if args.fast:
        # Minimal output in fast mode
        print('SUMMARY')
        print('-' * 80)
        print(f'Total files processed: {total_stats["files"]}')
        print(f'Total records: {total_stats["total_records"]}')
        valid_pct = _format_pct(total_stats['valid_records'], total_stats['total_records'], floor=True)
        invalid_pct = _format_pct(total_stats['invalid_records'], total_stats['total_records'], floor=False)
        print(f'Valid records: {total_stats["valid_records"]} ({valid_pct:.2f}%)')
        print(f'Invalid records: {total_stats["invalid_records"]} ({invalid_pct:.2f}%)')
    elif args.errors:
        print_errors_mode(all_results, full=args.full)
        print()
        # Also print fallback info in errors mode
        if total_stats['fallbacks']:
            print('PERMISSIVE MODEL FALLBACKS')
            print('-' * 80)
            print(f'{Colors.YELLOW}⚠ {len(total_stats["fallbacks"])} records used fallback typing:{Colors.RESET}')

            fallback_by_type: dict[str, list[FallbackUsage]] = defaultdict(list)
            for fb in total_stats['fallbacks']:
                fallback_by_type[fb['fallback_type']].append(fb)

            for fb_type, usages in sorted(fallback_by_type.items(), key=lambda x: -len(x[1])):
                print(f'\n  {Colors.CYAN}{fb_type}{Colors.RESET}: {len(usages)} instance(s)')
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
    else:
        print_summary_mode(all_results, total_stats)

    # Exit code based on validation success
    if total_stats['invalid_records'] == 0:
        print()
        print(f'{Colors.GREEN}✓ All records validated successfully!{Colors.RESET}')
        sys.exit(0)
    else:
        print()
        print(f'{Colors.RED}✗ Validation failed for {total_stats["invalid_records"]} records{Colors.RESET}')
        sys.exit(1)


def _format_pct(numerator: int, denominator: int, *, floor: bool) -> float:
    """Format a percentage with floor (never overstate) or ceil (never understate)."""
    if denominator == 0:
        return 0.0
    raw = numerator / denominator * 10000
    return math.floor(raw) / 100 if floor else math.ceil(raw) / 100


if __name__ == '__main__':
    main()
