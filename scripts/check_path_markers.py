#!/usr/bin/env -S uv run --no-project
# /// script
# dependencies = ["pydantic>=2.0.0"]
# ///

"""
Check that all path fields are marked with # PATH TO TRANSLATE comments.
"""

from __future__ import annotations

import re
from pathlib import Path


def check_path_markers() -> bool:
    """Check all path fields have translation markers."""

    models_file = (Path(__file__).parent.parent / 'src' / 'schemas' / 'session' / 'models.py').resolve(strict=True)
    content = models_file.read_text()

    # Fields that should have path markers
    path_fields = [
        'cwd',
        'file_path',
        'filePath',
        'projectPaths',
        'oldString',  # Edit tool
        'newString',  # Edit tool
    ]

    print('=' * 80)
    print('Path Translation Marker Check')
    print('=' * 80)
    print()

    issues = []
    marked_correctly = []

    for field in path_fields:
        # Find all occurrences of this field
        pattern = rf'^\s*{re.escape(field)}:\s*(.+?)$'
        matches = list(re.finditer(pattern, content, re.MULTILINE))

        for match in matches:
            line_text = match.group(0)

            # Check if it has PATH TO TRANSLATE comment
            if '# PATH TO TRANSLATE' in line_text:
                marked_correctly.append((field, line_text.strip()))
            else:
                # Check if it's in a comment or docstring (skip these)
                line_num = content[: match.start()].count('\n') + 1

                # Get context
                lines = content.split('\n')
                context_start = max(0, line_num - 3)
                context = '\n'.join(lines[context_start : line_num + 1])

                # Skip if in docstring or comment
                in_docstring = '"""' in context or "'''" in context
                in_comment = line_text.strip().startswith('#')

                if not in_docstring and not in_comment:
                    issues.append((field, line_num, line_text.strip()))

    print('✓ CORRECTLY MARKED PATH FIELDS:')
    for field, line in marked_correctly:
        print(f'  {field}: {line}')
    print()

    if issues:
        print('⚠ MISSING PATH MARKERS:')
        for field, line_num, line in issues:
            print(f'  Line {line_num}: {line}')
            print('    → Should have: # PATH TO TRANSLATE comment')
        print()
        return False
    else:
        print('✓ All path fields correctly marked!')
        print()
        return True


def find_potential_path_fields() -> None:
    """Find fields that might contain paths but aren't marked."""

    models_file = (Path(__file__).parent.parent / 'src' / 'schemas' / 'session' / 'models.py').resolve(strict=True)
    content = models_file.read_text()

    # Look for str fields that might be paths
    potential_patterns = [
        r'^\s*(\w*[Pp]ath\w*):\s*str',
        r'^\s*(\w*[Dd]ir\w*):\s*str',
        r'^\s*(\w*[Ff]ile\w*):\s*str',
        r'^\s*(cwd):\s*str',
    ]

    print('🔍 POTENTIAL PATH FIELDS (should review):')
    found_any = False

    for pattern in potential_patterns:
        matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
        for match in matches:
            field_name = match.group(1)
            line = match.group(0).strip()
            line_num = content[: match.start()].count('\n') + 1

            # Skip if already has marker
            if '# PATH TO TRANSLATE' not in line:
                # Get context
                lines = content.split('\n')
                context_start = max(0, line_num - 2)
                context_end = min(len(lines), line_num + 1)
                context = '\n'.join(f'  {i + 1}: {lines[i]}' for i in range(context_start, context_end))

                print(f'\nLine {line_num}: {field_name}')
                print(context)
                found_any = True

    if not found_any:
        print('  None found - all look good!')
    print()


if __name__ == '__main__':
    all_marked = check_path_markers()
    find_potential_path_fields()

    if not all_marked:
        exit(1)
