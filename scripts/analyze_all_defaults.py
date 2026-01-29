#!/usr/bin/env -S uv run
"""Comprehensive analysis of ALL = None defaults across ALL nested objects.

Recursively analyzes every nested structure in session files to find:
1. Fields defined with = None that NEVER appear in JSON
2. Fields that appear but are ALWAYS null
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

CLAUDE_PROJECTS_DIR = Path.home() / '.claude' / 'projects'
MODELS_FILE = Path(__file__).parent.parent / 'src' / 'schemas' / 'session' / 'models.py'


def get_all_none_defaults() -> dict[str, list[str]]:
    """Parse models.py to find ALL fields with = None default."""
    content = MODELS_FILE.read_text()

    model_fields: dict[str, list[str]] = defaultdict(list)
    current_class = None

    for line in content.splitlines():
        class_match = re.match(r'^class (\w+)\(', line)
        if class_match:
            current_class = class_match.group(1)
            continue

        if current_class and '= None' in line:
            field_match = re.match(r'\s+(\w+):\s*.+= None', line)
            if field_match:
                model_fields[current_class].append(field_match.group(1))

    return dict(model_fields)


def analyze_object(
    obj: Any,
    path: str,
    field_present: dict[str, dict[str, int]],
    field_null: dict[str, dict[str, int]],
    type_counts: dict[str, int],
) -> None:
    """Recursively analyze an object and all its nested structures."""
    if not isinstance(obj, dict):
        return

    # Determine the object type based on path or content
    obj_type = path.split('.')[-1] if '.' in path else path

    # Special handling for known nested types
    if 'role' in obj:
        obj_type = f'Message:{obj.get("role", "unknown")}'
    elif 'type' in obj and path != 'record':
        # Content blocks, tool results, etc.
        obj_type = f'ContentBlock:{obj.get("type")}'
    elif path == 'record':
        obj_type = f'Record:{obj.get("type", "unknown")}'

    type_counts[obj_type] += 1

    # Track all fields in this object
    for field, value in obj.items():
        field_present[obj_type][field] += 1
        if value is None:
            field_null[obj_type][field] += 1

        # Recurse into nested objects
        if isinstance(value, dict):
            nested_path = f'{path}.{field}'
            analyze_object(value, nested_path, field_present, field_null, type_counts)

        # Recurse into lists of objects
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    nested_path = f'{path}.{field}[]'
                    analyze_object(item, nested_path, field_present, field_null, type_counts)


def main() -> None:
    model_fields = get_all_none_defaults()
    total_none_fields = sum(len(v) for v in model_fields.values())

    print(f'Found {total_none_fields} fields with = None across {len(model_fields)} models')
    print()

    # Comprehensive analysis of all JSON structures
    field_present: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    field_null: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    type_counts: dict[str, int] = defaultdict(int)

    total_records = 0

    for project_dir in CLAUDE_PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue

        for session_file in project_dir.glob('*.jsonl'):
            try:
                for line in session_file.read_text().splitlines():
                    if not line.strip():
                        continue

                    record = json.loads(line)
                    total_records += 1
                    analyze_object(record, 'record', field_present, field_null, type_counts)

            except Exception:
                pass

    print(f'Analyzed {total_records:,} records')
    print(f'Found {len(type_counts)} distinct object types')
    print()

    # Find fields that are NEVER present
    print('=' * 80)
    print('FIELDS NEVER PRESENT IN JSON')
    print('(Defined with = None but field key never appears)')
    print('=' * 80)
    print()

    never_present_by_model: dict[str, list[tuple[str, str, int]]] = defaultdict(list)

    # Map JSON object types to model names (partial mapping, will miss some)
    json_to_model = {
        'Record:user': ['UserRecord'],
        'Record:assistant': [
            '_AssistantRecordBase',
            'NormalMainAssistantRecord',
            'ErrorMainAssistantRecord',
            'NormalAgentAssistantRecord',
            'ErrorAgentAssistantRecord',
        ],
        'Record:summary': ['SummaryRecord'],
        'Record:progress': ['ProgressRecord'],
        'Record:queue-operation': ['QueueOperationRecord'],
        'Record:file-history-snapshot': ['FileHistorySnapshotRecord'],
        'Record:system': ['SystemRecord', 'ApiErrorSystemRecord', 'CompactBoundarySystemRecord'],
        'Message:assistant': ['AssistantMessage'],
        'Message:user': ['UserMessage'],
        'ContentBlock:text': ['TextContent'],
        'ContentBlock:tool_use': ['ToolUseContent'],
        'ContentBlock:tool_result': ['ToolResultContent'],
        'ContentBlock:thinking': ['ThinkingContent'],
        'ContentBlock:image': ['ImageContent'],
        'ContentBlock:document': ['DocumentContent'],
        'record.message.usage': ['TokenUsage'],
        'record.message.usage.cache_creation': ['CacheCreation'],
        'record.message.usage.server_tool_use': ['ServerToolUse'],
        'record.message.context_management': ['ContextManagement'],
        'record.thinkingMetadata': ['ThinkingMetadata'],
        'record.mcpMeta': ['McpMeta'],
    }

    for json_type, model_names in json_to_model.items():
        for model_name in model_names:
            if model_name not in model_fields:
                continue

            fields = model_fields[model_name]
            present_fields = field_present.get(json_type, {})
            count = type_counts.get(json_type, 0)

            if count == 0:
                continue

            for field in fields:
                if field not in present_fields:
                    never_present_by_model[model_name].append((field, json_type, count))

    if never_present_by_model:
        for model_name, field_tuples in sorted(never_present_by_model.items()):
            print(f'{model_name}:')
            for field_name, json_type, count in field_tuples:
                print(f'  .{field_name} - NEVER in JSON ({count:,} {json_type} objects)')
            print()
    else:
        print('None found')
        print()

    # Find fields that are ALWAYS null when present
    print('=' * 80)
    print('FIELDS ALWAYS NULL WHEN PRESENT')
    print('(Field key exists but value is always null)')
    print('=' * 80)
    print()

    always_null_by_model: dict[str, list[tuple[str, str, int, int, float]]] = defaultdict(list)

    for json_type, model_names in json_to_model.items():
        for model_name in model_names:
            if model_name not in model_fields:
                continue

            fields = model_fields[model_name]
            present_fields = field_present.get(json_type, {})
            null_fields = field_null.get(json_type, {})
            count = type_counts.get(json_type, 0)

            if count == 0:
                continue

            for field in fields:
                present = present_fields.get(field, 0)
                null = null_fields.get(field, 0)

                if present > 0 and null == present:
                    pct = present / count * 100
                    always_null_by_model[model_name].append((field, json_type, present, count, pct))

    if always_null_by_model:
        for model_name, null_tuples in sorted(always_null_by_model.items()):
            print(f'{model_name}:')
            for field_name, json_type, present, count, pct in sorted(null_tuples, key=lambda x: -x[4]):
                print(f'  .{field_name} - {present:,}/{count:,} ({pct:.1f}%) ALWAYS NULL')
            print()
    else:
        print('None found')
        print()

    # Find fields that are ALWAYS PRESENT (= None default is unnecessary)
    print('=' * 80)
    print('FIELDS ALWAYS PRESENT (could remove = None default)')
    print('(Field has = None but is present in 100% of JSON objects)')
    print('=' * 80)
    print()

    always_present_by_model: dict[str, list[tuple[str, str, int]]] = defaultdict(list)

    for json_type, model_names in json_to_model.items():
        for model_name in model_names:
            if model_name not in model_fields:
                continue

            fields = model_fields[model_name]
            present_fields = field_present.get(json_type, {})
            count = type_counts.get(json_type, 0)

            if count < 100:  # Need significant sample
                continue

            for field in fields:
                present = present_fields.get(field, 0)
                if present == count:
                    always_present_by_model[model_name].append((field, json_type, count))

    if always_present_by_model:
        for model_name, present_tuples in sorted(always_present_by_model.items()):
            print(f'{model_name}:')
            for field_name, json_type, count in present_tuples:
                # Check if also always non-null
                null_count = field_null.get(json_type, {}).get(field_name, 0)
                if null_count == 0:
                    print(f'  .{field_name} - 100% present AND always non-null ({count:,} objects)')
                    print('      → ACTION: Remove = None, make required')
                else:
                    pct_null = null_count / count * 100
                    print(f'  .{field_name} - 100% present but {pct_null:.1f}% null ({count:,} objects)')
                    print('      → Keep = None (value can be null)')
            print()
    else:
        print('None found - all optional fields are genuinely optional')
        print()

    # Summary of all object types found (for debugging/verification)
    print('=' * 80)
    print('ALL OBJECT TYPES FOUND (for reference)')
    print('=' * 80)
    print()

    for obj_type, count in sorted(type_counts.items(), key=lambda x: -x[1])[:30]:
        print(f'  {obj_type}: {count:,}')


if __name__ == '__main__':
    main()
