"""
Tests for edge case fixtures.

These tests validate that all fixtures in the fixtures/ directory
pass our Pydantic model validation. This serves multiple purposes:

1. Regression testing - ensures model changes don't break edge cases
2. Documentation - fixtures demonstrate real-world edge cases
3. CI integration - can run in CI without access to user session files
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.schemas.session.models import SessionRecordAdapter

# Path to fixtures directory (relative to repo root)
FIXTURES_DIR = Path(__file__).parent.parent / 'fixtures'
EDGE_CASES_DIR = FIXTURES_DIR / 'edge_cases'


def iter_fixture_records(fixture_path: Path) -> list[tuple[int, dict[str, object]]]:
    """Load all records from a JSONL fixture file.

    Returns list of (line_number, record_dict) tuples.
    """
    records = []
    with open(fixture_path) as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if line:  # Skip empty lines
                records.append((i, json.loads(line)))
    return records


def get_edge_case_fixtures() -> list[Path]:
    """Get all edge case fixture files."""
    if not EDGE_CASES_DIR.exists():
        return []
    return list(EDGE_CASES_DIR.glob('*.jsonl'))


@pytest.mark.parametrize(
    'fixture_path',
    get_edge_case_fixtures(),
    ids=lambda p: p.name,
)
def test_edge_case_fixture_validates(fixture_path: Path) -> None:
    """Each edge case fixture must validate against our session models.

    This ensures that:
    1. The models we created for edge cases actually work
    2. Future model changes don't accidentally break edge case handling
    """
    records = iter_fixture_records(fixture_path)
    assert records, f'Fixture {fixture_path.name} is empty'

    errors = []
    for line_num, record in records:
        try:
            SessionRecordAdapter.validate_python(record)
        except Exception as e:
            errors.append(f'Line {line_num}: {e}')

    if errors:
        pytest.fail(f'Fixture {fixture_path.name} validation failed:\n' + '\n'.join(errors))


def test_fixtures_directory_exists() -> None:
    """Verify fixtures directory structure exists."""
    assert FIXTURES_DIR.exists(), 'fixtures/ directory not found'
    assert EDGE_CASES_DIR.exists(), 'fixtures/edge_cases/ directory not found'


def test_edge_cases_have_manifest() -> None:
    """Verify edge_cases has a manifest.json documenting the fixtures."""
    manifest_path = EDGE_CASES_DIR / 'manifest.json'
    assert manifest_path.exists(), 'fixtures/edge_cases/manifest.json not found'

    with open(manifest_path) as f:
        manifest = json.load(f)

    assert 'fixtures' in manifest, 'manifest.json missing "fixtures" key'

    # Verify each fixture in the directory is documented in manifest
    fixture_files = {p.name for p in get_edge_case_fixtures()}
    documented_fixtures = set(manifest['fixtures'].keys())

    undocumented = fixture_files - documented_fixtures
    assert not undocumented, f'Fixtures not documented in manifest: {undocumented}'
