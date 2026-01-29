#!/usr/bin/env -S uv run
"""Find fields with = None that could potentially be removed.

Two categories:
1. NEVER PRESENT: Field defined in model but never appears in JSON
   → Can remove field entirely OR remove = None default

2. ALWAYS NULL WHEN PRESENT: Field appears in JSON but value is always null
   → Needs migration to clean JSON, then can remove field

This helps identify "hallucinated" schema fields that don't reflect reality.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

CLAUDE_PROJECTS_DIR = Path.home() / '.claude' / 'projects'
MODELS_FILE = Path(__file__).parent.parent / 'src' / 'schemas' / 'session' / 'models.py'


def get_model_fields_with_none_default() -> dict[str, list[str]]:
    """Parse models.py to find all fields with = None default."""
    content = MODELS_FILE.read_text()

    # Find class definitions and their fields
    model_fields: dict[str, list[str]] = defaultdict(list)
    current_class = None

    for line in content.splitlines():
        # Match class definition
        class_match = re.match(r'^class (\w+)\(', line)
        if class_match:
            current_class = class_match.group(1)
            continue

        # Match field with = None default
        if current_class and '= None' in line:
            # Extract field name
            field_match = re.match(r'\s+(\w+):\s*.+= None', line)
            if field_match:
                field_name = field_match.group(1)
                model_fields[current_class].append(field_name)

    return dict(model_fields)


def analyze_json_presence() -> tuple[
    dict[str, dict[str, int]],  # record_type -> field -> present_count
    dict[str, dict[str, int]],  # record_type -> field -> null_count
    dict[str, int],  # record_type -> total_count
]:
    """Analyze actual JSON to find field presence and null rates."""
    field_present: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    field_null: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    record_counts: dict[str, int] = defaultdict(int)

    for project_dir in CLAUDE_PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue

        for session_file in project_dir.glob('*.jsonl'):
            try:
                for line in session_file.read_text().splitlines():
                    if not line.strip():
                        continue

                    record = json.loads(line)
                    record_type = record.get('type', 'unknown')
                    record_counts[record_type] += 1

                    # Analyze top-level fields
                    for field, value in record.items():
                        field_present[record_type][field] += 1
                        if value is None:
                            field_null[record_type][field] += 1

                    # Analyze nested message fields
                    if 'message' in record and isinstance(record['message'], dict):
                        msg = record['message']
                        msg_type = f'message:{msg.get("role", "unknown")}'
                        record_counts[msg_type] += 1

                        for field, value in msg.items():
                            field_present[msg_type][field] += 1
                            if value is None:
                                field_null[msg_type][field] += 1

                        # Analyze usage fields
                        if 'usage' in msg and isinstance(msg['usage'], dict):
                            usage = msg['usage']
                            record_counts['TokenUsage'] += 1
                            for field, value in usage.items():
                                field_present['TokenUsage'][field] += 1
                                if value is None:
                                    field_null['TokenUsage'][field] += 1

                    # Analyze tool_use input fields
                    if record_type == 'assistant' and 'message' in record:
                        msg = record.get('message', {})
                        content = msg.get('content', [])
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get('type') == 'tool_use':
                                    tool_name = block.get('name', 'unknown')
                                    tool_input = block.get('input', {})
                                    if isinstance(tool_input, dict):
                                        tool_type = f'ToolInput:{tool_name}'
                                        record_counts[tool_type] += 1
                                        for field, value in tool_input.items():
                                            field_present[tool_type][field] += 1
                                            if value is None:
                                                field_null[tool_type][field] += 1

            except Exception:
                pass

    return dict(field_present), dict(field_null), dict(record_counts)


def main() -> None:
    print('Analyzing fields with = None defaults...')
    print()

    # Get fields defined with = None in models.py
    model_fields = get_model_fields_with_none_default()

    print(f'Found {sum(len(v) for v in model_fields.values())} fields with = None across {len(model_fields)} models')
    print()

    # Get actual JSON presence data
    field_present, field_null, record_counts = analyze_json_presence()

    print('=' * 80)
    print('CATEGORY 1: FIELDS NEVER PRESENT IN JSON')
    print('(Model defines field with = None, but JSON never has this field)')
    print('→ Can remove field entirely from model')
    print('=' * 80)
    print()

    never_present: list[tuple[str, str, int]] = []

    # Map model names to JSON record types
    type_mapping = {
        'UserRecord': 'user',
        'AssistantMessage': 'message:assistant',
        'UserMessage': 'message:user',
        'TokenUsage': 'TokenUsage',
        'SummaryRecord': 'summary',
        'ProgressRecord': 'progress',
        'QueueOperationRecord': 'queue-operation',
        'FileHistorySnapshotRecord': 'file-history-snapshot',
        # Assistant record variants
        'NormalMainAssistantRecord': 'assistant',
        'ErrorMainAssistantRecord': 'assistant',
        'NormalAgentAssistantRecord': 'assistant',
        'ErrorAgentAssistantRecord': 'assistant',
        '_AssistantRecordBase': 'assistant',
    }

    for model_name, fields in sorted(model_fields.items()):
        json_type = type_mapping.get(model_name)
        if not json_type:
            continue

        present_fields = field_present.get(json_type, {})
        total = record_counts.get(json_type, 0)

        if total == 0:
            continue

        never_present.extend((model_name, field, total) for field in fields if field not in present_fields)

    if never_present:
        for model, field, total in sorted(never_present):
            print(f'{model}.{field}')
            print(f'  → NEVER in JSON ({total:,} records checked)')
            print('  → Action: Remove field from model OR make required')
            print()
    else:
        print('None found - all fields with = None appear in JSON at least sometimes')
        print()

    print('=' * 80)
    print('CATEGORY 2: FIELDS ALWAYS NULL WHEN PRESENT')
    print('(Field appears in JSON, but value is always null)')
    print('→ Need migration to remove from JSON, then remove from model')
    print('=' * 80)
    print()

    always_null = []

    for model_name, fields in sorted(model_fields.items()):
        json_type = type_mapping.get(model_name)
        if not json_type:
            continue

        present_fields = field_present.get(json_type, {})
        null_fields = field_null.get(json_type, {})
        total = record_counts.get(json_type, 0)

        if total == 0:
            continue

        for field in fields:
            present = present_fields.get(field, 0)
            null = null_fields.get(field, 0)

            # Field is present AND always null
            if present > 0 and null == present:
                pct = present / total * 100
                always_null.append((model_name, field, present, total, pct))

    if always_null:
        for model, field, present, total, pct in sorted(always_null, key=lambda x: -x[4]):
            print(f'{model}.{field}')
            print(f'  → Present in {present:,}/{total:,} records ({pct:.1f}%)')
            print('  → ALWAYS NULL when present')
            print('  → Action: Add migration, then remove field')
            print()
    else:
        print('None found')
        print()

    print('=' * 80)
    print('CATEGORY 3: LEGITIMATE OPTIONAL FIELDS')
    print('(Field sometimes present with non-null values - keep = None)')
    print('=' * 80)
    print()

    legitimate = []

    for model_name, fields in sorted(model_fields.items()):
        json_type = type_mapping.get(model_name)
        if not json_type:
            continue

        present_fields = field_present.get(json_type, {})
        null_fields = field_null.get(json_type, {})
        total = record_counts.get(json_type, 0)

        if total == 0:
            continue

        for field in fields:
            present = present_fields.get(field, 0)
            null = null_fields.get(field, 0)
            non_null = present - null

            # Has non-null values AND is not always present
            if non_null > 0 and present < total:
                pct_present = present / total * 100
                pct_non_null = non_null / present * 100 if present > 0 else 0
                legitimate.append((model_name, field, present, total, pct_present, non_null, pct_non_null))

    for model, field, present, total, pct_present, non_null, pct_non_null in sorted(legitimate, key=lambda x: x[4]):
        print(f'{model}.{field}: {pct_present:.1f}% present, {pct_non_null:.0f}% non-null when present')

    # Now analyze TOOL INPUT models
    print()
    print('=' * 80)
    print('TOOL INPUT ANALYSIS')
    print('=' * 80)
    print()

    # Map tool names to model names
    tool_to_model = {
        'Read': 'ReadToolInput',
        'Edit': 'EditToolInput',
        'Write': 'WriteToolInput',
        'Bash': 'BashToolInput',
        'Glob': 'GlobToolInput',
        'Grep': 'GrepToolInput',
        'Task': 'TaskToolInput',
        'TaskOutput': 'TaskOutputToolInput',
        'WebFetch': 'WebFetchToolInput',
        'WebSearch': 'WebSearchToolInput',
        'AskUserQuestion': 'AskUserQuestionToolInput',
        'NotebookEdit': 'NotebookEditToolInput',
        'ToolSearch': 'ToolSearchToolInput',
        'MCPSearch': 'MCPSearchToolInput',
        'TodoRead': 'TodoReadToolInput',
        'TodoWrite': 'TodoWriteToolInput',
        'Skill': 'SkillToolInput',
        'EnterPlanMode': 'EnterPlanModeToolInput',
        'ExitPlanMode': 'ExitPlanModeToolInput',
        'ListMcpResourcesTool': 'ListMcpResourcesToolInput',
        'ReadMcpResourceTool': 'ReadMcpResourceToolInput',
        'TaskCreate': 'TaskCreateToolInput',
        'TaskUpdate': 'TaskUpdateToolInput',
        'TaskGet': 'TaskGetToolInput',
        'TaskList': 'TaskListToolInput',
    }

    tool_never_present = []
    tool_always_null = []

    for tool_name, model_name in tool_to_model.items():
        if model_name not in model_fields:
            continue

        fields = model_fields[model_name]
        json_type = f'ToolInput:{tool_name}'
        present_fields = field_present.get(json_type, {})
        null_fields = field_null.get(json_type, {})
        total = record_counts.get(json_type, 0)

        if total < 10:  # Skip rare tools
            continue

        for field in fields:
            present = present_fields.get(field, 0)
            null = null_fields.get(field, 0)

            if present == 0:
                tool_never_present.append((model_name, field, total))
            elif null == present and present > 0:
                pct = present / total * 100
                tool_always_null.append((model_name, field, present, total, pct))

    if tool_never_present:
        print('TOOL INPUT FIELDS NEVER PRESENT:')
        print('(Defined with = None but never used in actual tool calls)')
        print()
        for model, field, total in sorted(tool_never_present):
            print(f'  {model}.{field} - never used ({total:,} calls checked)')
        print()

    if tool_always_null:
        print('TOOL INPUT FIELDS ALWAYS NULL WHEN PRESENT:')
        print('(Field sent but always null)')
        print()
        for model, field, present, total, pct in sorted(tool_always_null, key=lambda x: -x[4]):
            print(f'  {model}.{field} - {present}/{total} ({pct:.1f}%) always null')
        print()

    if not tool_never_present and not tool_always_null:
        print('All tool input fields with = None are legitimately optional')


if __name__ == '__main__':
    main()
