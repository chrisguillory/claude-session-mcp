#!/usr/bin/env -S uv run
"""Analyze tool input fields to find lazy defaults that could be tightened."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

CLAUDE_PROJECTS_DIR = Path.home() / '.claude' / 'projects'


def main() -> None:
    # Track field presence per tool
    tool_fields: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {'present': 0, 'null': 0, 'non_null': 0})
    )
    tool_counts: dict[str, int] = defaultdict(int)

    for project_dir in CLAUDE_PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue

        for session_file in project_dir.glob('*.jsonl'):
            try:
                for line in session_file.read_text().splitlines():
                    if not line.strip():
                        continue

                    record = json.loads(line)
                    if record.get('type') != 'assistant':
                        continue

                    msg = record.get('message', {})
                    content = msg.get('content', [])
                    if not isinstance(content, list):
                        continue

                    for block in content:
                        if block.get('type') != 'tool_use':
                            continue

                        tool_name = block.get('name', '')
                        tool_input = block.get('input', {})
                        if not isinstance(tool_input, dict):
                            continue

                        tool_counts[tool_name] += 1

                        for field, value in tool_input.items():
                            stats = tool_fields[tool_name][field]
                            stats['present'] += 1
                            if value is None:
                                stats['null'] += 1
                            else:
                                stats['non_null'] += 1

            except Exception:
                pass

    # Report fields that are always present and always non-null (could be required)
    print('TOOL INPUT FIELDS THAT COULD BE REQUIRED')
    print('=' * 80)
    print('(Fields always present and always non-null in actual usage)')
    print()

    for tool_name in sorted(tool_counts.keys()):
        count = tool_counts[tool_name]
        if count < 10:  # Skip rare tools
            continue

        always_present = []
        for field, stats in tool_fields[tool_name].items():
            if stats['present'] == count and stats['null'] == 0:
                always_present.append(field)

        if always_present:
            print(f'{tool_name} ({count} uses):')
            for field in sorted(always_present):
                print(f'  {field}: always present, always non-null')
            print()

    # Report fields that are always null when present
    print()
    print('TOOL INPUT FIELDS ALWAYS NULL WHEN PRESENT')
    print('=' * 80)
    print('(Lazy defaults that waste JSON bytes)')
    print()

    for tool_name in sorted(tool_counts.keys()):
        count = tool_counts[tool_name]
        if count < 10:
            continue

        always_null = []
        for field, stats in tool_fields[tool_name].items():
            if stats['present'] > 0 and stats['null'] == stats['present']:
                always_null.append((field, stats['present']))

        if always_null:
            print(f'{tool_name} ({count} uses):')
            for field, present in sorted(always_null, key=lambda x: -x[1]):
                pct = present / count * 100
                print(f'  {field}: {present} present ({pct:.1f}%), ALWAYS NULL')
            print()


if __name__ == '__main__':
    main()
