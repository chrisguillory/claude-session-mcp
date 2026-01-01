#!/usr/bin/env -S uv run --no-project
# /// script
# dependencies = ["pydantic>=2.0.0"]
# ///

"""
Export JSON Schema from Pydantic models.

This generates a JSON Schema document that other tools can consume:
- TypeScript type generation
- OpenAPI documentation
- Cross-language validation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    CLAUDE_CODE_MAX_VERSION,
    CLAUDE_CODE_MIN_VERSION,
    LAST_VALIDATED,
    SCHEMA_VERSION,
    VALIDATION_RECORD_COUNT,
    SessionRecordAdapter,
)


def export_schema(output_path: str = 'session-schema.json'):
    """Export complete JSON Schema for SessionRecord."""

    print('=' * 80)
    print('JSON Schema Export')
    print('=' * 80)
    print()

    # Generate schema
    schema = SessionRecordAdapter.json_schema()

    # Add metadata
    schema['$schema'] = 'https://json-schema.org/draft/2020-12/schema'
    schema['$id'] = 'https://github.com/your-org/claude-session-mcp/session-schema.json'
    schema['title'] = 'Claude Code Session Record'
    schema['description'] = 'Complete schema for Claude Code session JSONL records'

    # Add version info
    schema['x-schema-version'] = SCHEMA_VERSION
    schema['x-claude-code-compatibility'] = {
        'min': CLAUDE_CODE_MIN_VERSION,
        'max': CLAUDE_CODE_MAX_VERSION,
        'last_validated': LAST_VALIDATED,
        'validation_record_count': VALIDATION_RECORD_COUNT,
    }

    # Write to file
    output_file = Path(output_path)
    with open(output_file, 'w') as f:
        json.dump(schema, f, indent=2)

    print(f'✓ Exported JSON Schema to: {output_file}')
    print(f'  Schema version: {SCHEMA_VERSION}')
    print(f'  Claude Code compatibility: {CLAUDE_CODE_MIN_VERSION} - {CLAUDE_CODE_MAX_VERSION}')
    print(f'  Size: {output_file.stat().st_size:,} bytes')
    print()

    # Show some stats
    definitions = schema.get('$defs', {})
    print('Schema contains:')
    print(f'  {len(definitions)} model definitions')

    # Count discriminated unions
    discriminated_unions = sum(1 for def_name, def_schema in definitions.items() if 'discriminator' in def_schema)
    print(f'  {discriminated_unions} discriminated unions')

    return schema


if __name__ == '__main__':
    output_path = sys.argv[1] if len(sys.argv) > 1 else 'session-schema.json'
    export_schema(output_path)
