#!/usr/bin/env -S uv run --no-project
# /// script
# dependencies = []
# ///

"""
Check for corrupted JSONL session files in ~/.claude/projects.

This script validates that all session files contain valid JSON records.
Useful for detecting file corruption before running the MCP server.
"""

import json
import sys
from pathlib import Path


def check_file(jsonl_file: Path) -> tuple[bool, int, str | None]:
    """
    Check a single JSONL file for corruption.

    Returns:
        (is_valid, total_lines, error_message)
    """
    total_lines = 0

    try:
        with open(jsonl_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                total_lines += 1

                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    return (False, total_lines, f'Line {line_num}: {e}')

        return (True, total_lines, None)

    except Exception as e:
        return (False, 0, f'Failed to read file: {e}')


def main():
    print('=' * 80)
    print('Claude Code Session Corruption Check')
    print('=' * 80)
    print()

    claude_dir = Path.home() / '.claude' / 'projects'

    if not claude_dir.exists():
        print(f'Directory not found: {claude_dir}')
        sys.exit(1)

    all_files = []
    for project_dir in claude_dir.iterdir():
        if not project_dir.is_dir():
            continue

        for jsonl_file in project_dir.glob('*.jsonl'):
            all_files.append(jsonl_file)

    all_files.sort()

    print(f'Found {len(all_files)} session files')
    print()

    corrupted_files = []
    total_records = 0

    for jsonl_file in all_files:
        is_valid, record_count, error = check_file(jsonl_file)

        if not is_valid:
            corrupted_files.append((jsonl_file, error))
            print(f'✗ CORRUPTED: {jsonl_file.name}')
            print(f'  Project: {jsonl_file.parent.name}')
            print(f'  Error: {error}')
            print()
        else:
            total_records += record_count

    print()
    print('SUMMARY')
    print('-' * 80)
    print(f'Total files checked: {len(all_files)}')
    print(f'Valid files: {len(all_files) - len(corrupted_files)}')
    print(f'Corrupted files: {len(corrupted_files)}')
    print(f'Total valid records: {total_records:,}')
    print()

    if corrupted_files:
        print('CORRUPTED FILES:')
        for file, error in corrupted_files:
            print(f'  {file}')
            print(f'    {error}')
        print()
        print('✗ Found corrupted files!')
        sys.exit(1)
    else:
        print('✓ All session files are valid JSON!')
        sys.exit(0)


if __name__ == '__main__':
    main()
