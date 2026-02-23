#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# ///
"""Fix broken `claude --resume` caused by orphan parentUuid pointers.

Problem 1 - Orphan pointers:
    Claude Code's prompt_suggestion agents write turn_duration and other
    records into the main session JSONL with parentUuid pointing into agent
    sidechain files (agent-aprompt_suggestion-*.jsonl). When one of these
    records is the last entry in the session, `claude --resume` walks the
    parentUuid chain from the tail, immediately hits a dangling pointer,
    and shows no conversation history.

    See: https://github.com/anthropics/claude-code/issues/23375

Problem 2 - Duplicate UUIDs from saved_hook_context:
    saved_hook_context records reuse the same UUID across multiple entries,
    updating parentUuid after each turn. The uuid_index (last-write-wins)
    creates artificial cycles: the last occurrence points to a conversation
    record which points back to the same UUID. Fixed by excluding duplicate
    UUIDs from the index — affected records become orphans and get rewired
    to the previous valid record in file order.

Problem 3 - Stale parent after resume:
    When a session is resumed, the new user record's parentUuid sometimes
    latches onto a record from an old compaction segment instead of the
    latest one. The chain from tail is structurally intact (no broken
    pointers) but skips newer compact_boundary segments containing the
    actual recent conversation. Detected but not auto-fixed.

Fix:
    Rewires orphan parentUuid pointers to the previous record in the main
    session file, reconnecting the linked list so --resume can walk the
    full chain back to root. Stale-parent issues require manual rewiring.

Prevention:
    Set CLAUDE_CODE_ENABLE_PROMPT_SUGGESTION=false in ~/.claude/settings.json
    to prevent prompt_suggestion agents from creating orphan records.

Usage:
    fix_broken_session_resume.py scan              # Check all sessions
    fix_broken_session_resume.py check <id>        # Diagnose one session
    fix_broken_session_resume.py fix <id|--all>    # Fix session(s)
    fix_broken_session_resume.py restore <id>      # Undo a fix

Exit codes:
    0 - Healthy / success
    1 - Broken or stale, fixable
    2 - Broken, unfixable
    3 - Error
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BACKUP_DIR = Path.home() / '.claude-session-mcp' / 'chain-backups'

EXIT_HEALTHY = 0
EXIT_FIXABLE = 1
EXIT_UNFIXABLE = 2
EXIT_ERROR = 3


@dataclass
class Orphan:
    line: int
    type: str
    old_parent: str
    new_parent: str | None
    new_parent_line: int | None
    new_parent_type: str | None
    source_agent: str | None = None


@dataclass
class StaleSegment:
    """A compact_boundary segment unreachable from the tail chain.

    Indicates the resume created a parentUuid pointing to an older compaction
    segment, skipping newer segments that contain the actual recent conversation.
    """

    root_line: int
    root_uuid: str
    deepest_line: int  # upper bound: line before next boundary or EOF


@dataclass
class AnalysisResult:
    file_path: Path
    status: str  # "healthy", "fixable", "unfixable", "error"
    error_message: str | None = None
    total_records: int = 0
    total_uuids: int = 0
    orphans: list[Orphan] = field(default_factory=list)
    chain_steps: int = 0
    chain_break_uuid: str | None = None
    last_line: int | None = None
    fixed_steps: int | None = None
    stale_segments: list[StaleSegment] = field(default_factory=list)
    duplicate_uuids: dict[str, int] = field(default_factory=dict)  # uuid -> occurrence count

    @property
    def prompt_suggestion_count(self) -> int:
        return sum(1 for o in self.orphans if o.source_agent and 'prompt_suggestion' in o.source_agent)

    @property
    def other_agent_count(self) -> int:
        return sum(1 for o in self.orphans if o.source_agent and 'prompt_suggestion' not in o.source_agent)

    @property
    def duplicate_uuid_orphan_count(self) -> int:
        return sum(1 for o in self.orphans if o.old_parent in self.duplicate_uuids)

    @property
    def unattributed_count(self) -> int:
        return sum(1 for o in self.orphans if not o.source_agent and o.old_parent not in self.duplicate_uuids)

    @property
    def exit_code(self) -> int:
        return {
            'healthy': EXIT_HEALTHY,
            'fixable': EXIT_FIXABLE,
            'stale': EXIT_FIXABLE,
            'unfixable': EXIT_UNFIXABLE,
            'error': EXIT_ERROR,
        }[self.status]


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------


def find_all_session_files() -> list[Path]:
    """Find all session JSONL files under ~/.claude/projects/."""
    projects_dir = Path.home() / '.claude' / 'projects'
    if not projects_dir.exists():
        return []
    results: list[Path] = []
    for d in projects_dir.iterdir():
        if not d.is_dir():
            continue
        results.extend(f for f in d.iterdir() if f.suffix == '.jsonl' and not f.name.startswith('agent-'))
    return sorted(results)


def find_session_file(session_id: str) -> Path | None:
    """Find a session JSONL file by ID or prefix under ~/.claude/projects/."""
    matches = [f for f in find_all_session_files() if f.stem.startswith(session_id)]

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        for m in matches:
            if m.stem == session_id:
                return m
        print(f"Error: ambiguous prefix '{session_id}', matches {len(matches)} files:", file=sys.stderr)
        for m in matches:
            print(f'  {m}', file=sys.stderr)
        return None
    return None


# ---------------------------------------------------------------------------
# Analysis engine
# ---------------------------------------------------------------------------


def attribute_orphans(file_path: Path, orphan_uuids: set[str]) -> dict[str, str]:
    """Search sidechain agent files for orphan UUIDs. Returns uuid -> agent filename."""
    if not orphan_uuids:
        return {}

    session_id = file_path.stem
    project_dir = file_path.parent
    attribution: dict[str, str] = {}

    agent_files: list[Path] = []
    # Nested subagents
    subagents_dir = project_dir / session_id / 'subagents'
    if subagents_dir.exists():
        agent_files.extend(subagents_dir.glob('agent-*.jsonl'))
    # Clone agents at project root
    agent_files.extend(project_dir.glob(f'agent-*-clone-{session_id[:8]}*.jsonl'))
    # Non-clone agents at project root (older format)
    agent_files.extend(project_dir.glob('agent-*.jsonl'))

    remaining = set(orphan_uuids)
    for af in agent_files:
        if not remaining:
            break
        with open(af) as f:
            for line in f:
                rec = json.loads(line)
                uuid = rec.get('uuid', '')
                if uuid in remaining:
                    attribution[uuid] = af.name
                    remaining.discard(uuid)
                    if not remaining:
                        break

    return attribution


def analyze_session(file_path: Path) -> AnalysisResult:
    """Analyze a session file for chain integrity. Returns structured result."""
    if not file_path.exists():
        return AnalysisResult(file_path=file_path, status='error', error_message='file not found')

    try:
        with open(file_path) as f:
            raw_lines = f.readlines()
    except Exception as e:
        return AnalysisResult(file_path=file_path, status='error', error_message=str(e))

    if not raw_lines:
        return AnalysisResult(file_path=file_path, status='error', error_message='empty file')

    # Parse records (skip blank lines but preserve line numbering)
    uuid_index: dict[str, int] = {}
    uuid_counts: dict[str, int] = {}
    records: list[dict[str, Any] | None] = []  # None for blank lines
    for i, line in enumerate(raw_lines):
        if not line.strip():
            records.append(None)
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            return AnalysisResult(
                file_path=file_path, status='error', error_message=f'JSON parse error line {i + 1}: {e}'
            )
        records.append(rec)
        uuid = rec.get('uuid')
        if uuid:
            uuid_counts[uuid] = uuid_counts.get(uuid, 0) + 1
            uuid_index[uuid] = i

    # Exclude duplicate UUIDs — they can't be reliably resolved (e.g. saved_hook_context
    # records reuse the same UUID across entries, creating artificial cycles).
    dup_uuids = {u: c for u, c in uuid_counts.items() if c > 1}
    for u in dup_uuids:
        del uuid_index[u]

    # Find orphans
    orphans: list[Orphan] = []
    for i, rec in enumerate(records):
        if rec is None:  # Skip blank lines
            continue
        parent = rec.get('parentUuid')
        if parent and parent not in uuid_index:
            rec_type = rec.get('type', '?')
            subtype = rec.get('subtype', '')
            label = rec_type + (f'/{subtype}' if subtype else '')

            new_parent = None
            new_parent_line = None
            new_parent_type = None
            for j in range(i - 1, -1, -1):
                prev_rec = records[j]
                if prev_rec is None:  # Skip blank lines
                    continue
                prev_uuid = prev_rec.get('uuid')
                if prev_uuid and prev_uuid in uuid_index:
                    new_parent = prev_uuid
                    new_parent_line = j + 1
                    pt = prev_rec.get('type', '?')
                    ps = prev_rec.get('subtype', '')
                    new_parent_type = pt + (f'/{ps}' if ps else '')
                    break

            orphans.append(
                Orphan(
                    line=i + 1,
                    type=label,
                    old_parent=parent,
                    new_parent=new_parent,
                    new_parent_line=new_parent_line,
                    new_parent_type=new_parent_type,
                )
            )

    # Attribute orphans to sidechain files
    orphan_uuids = {o.old_parent for o in orphans}
    attribution = attribute_orphans(file_path, orphan_uuids)
    for o in orphans:
        o.source_agent = attribution.get(o.old_parent)

    # Find tail (last record with uuid)
    last_uuid = None
    last_line = None
    for i in range(len(records) - 1, -1, -1):
        tail_rec = records[i]
        if tail_rec is None:  # Skip blank lines
            continue
        if tail_rec.get('uuid'):
            last_uuid = tail_rec['uuid']
            last_line = i + 1
            break

    if not last_uuid:
        return AnalysisResult(file_path=file_path, status='error', error_message='no records with uuid')

    # Walk unpatched chain (with cycle detection)
    current = last_uuid
    chain_steps = 0
    chain_break_uuid = None
    chain_visited: set[str] = set()
    while current:
        if current in chain_visited:
            # Cycle — treat as broken
            chain_break_uuid = current
            break
        chain_visited.add(current)
        if current in uuid_index:
            chain_rec = records[uuid_index[current]]
            assert chain_rec is not None  # Records in uuid_index are never blank lines
            current = chain_rec.get('parentUuid') or None
        else:
            chain_break_uuid = current
            break
        chain_steps += 1

    chain_healthy = current is None

    if chain_healthy:
        # Detect stale-parent resume: chain reaches root but skips newer segments.
        # Find compact_boundary roots not reachable from the tail chain.
        chain_uuids: set[str] = set()
        walk = last_uuid
        while walk and walk in uuid_index and walk not in chain_uuids:
            chain_uuids.add(walk)
            rec = records[uuid_index[walk]]
            assert rec is not None
            walk = rec.get('parentUuid') or None

        chain_lines = {uuid_index[u] for u in chain_uuids}
        chain_min_line = min(chain_lines) if chain_lines else 0

        # Find compact_boundary roots newer than the chain but not in it.
        # Use the next boundary (or EOF) as a cheap upper-bound for the segment
        # extent — avoids an expensive BFS over potentially thousands of records.
        all_boundaries = [
            (i, rec)
            for i, rec in enumerate(records)
            if rec is not None
            and rec.get('type') == 'system'
            and rec.get('subtype') == 'compact_boundary'
            and rec.get('uuid')
        ]

        stale_segments: list[StaleSegment] = []
        for bi, (i, rec) in enumerate(all_boundaries):
            if rec['uuid'] in chain_uuids or i <= chain_min_line:
                continue
            # Segment extends from this boundary to the next (or EOF)
            if bi + 1 < len(all_boundaries):
                deepest = all_boundaries[bi + 1][0] - 1
            else:
                deepest = len(records) - 1
            stale_segments.append(StaleSegment(root_line=i + 1, root_uuid=rec['uuid'], deepest_line=deepest + 1))

        status = 'stale' if stale_segments else 'healthy'
        return AnalysisResult(
            file_path=file_path,
            status=status,
            total_records=sum(1 for r in records if r is not None),
            total_uuids=len(uuid_index),
            orphans=orphans,
            chain_steps=chain_steps,
            last_line=last_line,
            stale_segments=stale_segments,
            duplicate_uuids=dup_uuids,
        )

    # Simulate fix
    parent_map: dict[str, str | None] = {}
    for i, rec in enumerate(records):
        if rec is None:  # Skip blank lines
            continue
        uuid = rec.get('uuid')
        if uuid:
            parent_map[uuid] = rec.get('parentUuid') or None

    for orphan in orphans:
        rec = records[orphan.line - 1]
        if rec is None:  # Should never happen for orphan lines, but defensive
            continue
        uuid = rec.get('uuid')
        if uuid and orphan.new_parent is not None:
            parent_map[uuid] = orphan.new_parent

    current = last_uuid
    fixed_steps = 0
    fix_visited: set[str] = set()
    while current:
        if current in fix_visited:
            break  # Cycle
        fix_visited.add(current)
        if current in uuid_index:
            current = parent_map.get(current) or None
        else:
            break
        fixed_steps += 1

    fix_works = current is None
    status = 'fixable' if fix_works else 'unfixable'

    return AnalysisResult(
        file_path=file_path,
        status=status,
        total_records=sum(1 for r in records if r is not None),
        total_uuids=len(uuid_index),
        orphans=orphans,
        chain_steps=chain_steps,
        chain_break_uuid=chain_break_uuid,
        last_line=last_line,
        fixed_steps=fixed_steps if fix_works else None,
        duplicate_uuids=dup_uuids,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_check(r: AnalysisResult) -> None:
    """Print detailed check output for one session."""
    if r.status == 'error':
        print(f'Error: {r.error_message}: {r.file_path}', file=sys.stderr)
        return

    print(f'Session: {r.file_path}')
    dup_str = f' | Dup UUIDs: {len(r.duplicate_uuids)}' if r.duplicate_uuids else ''
    print(f'Records: {r.total_records} | UUIDs: {r.total_uuids} | Orphans: {len(r.orphans)}{dup_str}')

    if r.status == 'healthy':
        print(f'Chain from tail (L{r.last_line}): HEALTHY ({r.chain_steps} steps to root)')
        if r.orphans:
            print(f'Note: {len(r.orphans)} orphans exist on dead branches (not affecting active chain)')
        print('Status: HEALTHY')
        return

    if r.status == 'stale':
        print(f'Chain from tail (L{r.last_line}): STALE ({r.chain_steps} steps to root)')
        print(f'\nResume latched onto an old compaction segment, skipping {len(r.stale_segments)} newer segment(s):')
        for seg in r.stale_segments:
            print(f'  Segment root L{seg.root_line} (uuid={seg.root_uuid[:16]}...) extends to L{seg.deepest_line}')
        print('\nThe tail parentUuid points to a valid but outdated record. Newer conversation')
        print('content exists in unreachable segments. Manual rewiring required.')
        print('Status: STALE')
        return

    orphan_id = r.chain_break_uuid[:16] if r.chain_break_uuid else '?'
    print(f'Chain from tail (L{r.last_line}): BROKEN at step {r.chain_steps} (orphan: {orphan_id}...)')

    if r.status == 'fixable':
        print(f'After rewiring: CLEAN ({r.fixed_steps} steps to root)')
        print('Status: FIXABLE')
    else:
        print('After rewiring: STILL BROKEN')
        print('Status: UNFIXABLE')

    # Cause summary
    causes = []
    if r.duplicate_uuid_orphan_count:
        causes.append(f'{r.duplicate_uuid_orphan_count} duplicate_uuid')
    if r.prompt_suggestion_count:
        causes.append(f'{r.prompt_suggestion_count} prompt_suggestion')
    if r.other_agent_count:
        causes.append(f'{r.other_agent_count} other agent')
    if r.unattributed_count:
        causes.append(f'{r.unattributed_count} unattributed')
    print(f'\nCause: {", ".join(causes)}')

    print('\nOrphans:')
    for o in r.orphans:
        source = f'  [from {o.source_agent}]' if o.source_agent else ''
        if o.new_parent:
            print(f'  L{o.line:<5d}  {o.type:<25s}  -> rewire to L{o.new_parent_line} ({o.new_parent_type}){source}')
        else:
            print(f'  L{o.line:<5d}  {o.type:<25s}  -> NO CANDIDATE{source}')


# ---------------------------------------------------------------------------
# Fix engine
# ---------------------------------------------------------------------------


def apply_fix(file_path: Path, orphans: list[Orphan]) -> Path:
    """Apply orphan rewiring to a session file. Returns backup path.

    Raises:
        FileExistsError: If backup already exists (prevents overwriting)
    """
    session_id = file_path.stem

    # Create backup
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / f'{session_id}.jsonl'
    meta_path = BACKUP_DIR / f'{session_id}.meta.json'

    # Check for existing backup
    if backup_path.exists():
        raise FileExistsError(
            f'Backup already exists: {backup_path}\n'
            f"Run 'fix_broken_session_resume.py restore {session_id}' first, or manually remove the backup."
        )

    shutil.copy2(file_path, backup_path)

    meta = {
        'session_id': session_id,
        'original_path': str(file_path),
        'backup_path': str(backup_path),
        'fixed_at': datetime.now(UTC).isoformat(),
        'orphans_fixed': len(orphans),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + '\n')

    # Build line index of orphans that have a rewire target
    # line number (1-indexed) -> new parentUuid
    rewire_map: dict[int, str] = {}
    for o in orphans:
        if o.new_parent is not None:
            rewire_map[o.line] = o.new_parent

    # Read, patch, write
    with open(file_path) as f:
        lines = f.readlines()

    for line_num, new_parent in rewire_map.items():
        idx = line_num - 1
        rec = json.loads(lines[idx])
        rec['parentUuid'] = new_parent
        lines[idx] = json.dumps(rec) + '\n'

    with open(file_path, 'w') as f:
        f.writelines(lines)

    return backup_path


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_scan(args: argparse.Namespace) -> int:
    """Scan all sessions and print summary."""
    files = find_all_session_files()
    if not files:
        print('No session files found under ~/.claude/projects/')
        return EXIT_ERROR

    print(f'Scanning {len(files)} session files...\n')

    healthy: list[AnalysisResult] = []
    stale: list[AnalysisResult] = []
    fixable: list[AnalysisResult] = []
    unfixable: list[AnalysisResult] = []
    errors: list[AnalysisResult] = []

    for f in files:
        r = analyze_session(f)
        if r.status == 'healthy':
            healthy.append(r)
        elif r.status == 'stale':
            stale.append(r)
        elif r.status == 'fixable':
            fixable.append(r)
        elif r.status == 'unfixable':
            unfixable.append(r)
        else:
            errors.append(r)

    print(f'Healthy:    {len(healthy):>4d}')
    print(f'Stale:      {len(stale):>4d}  (resume skipped newer segments)')
    print(f'Fixable:    {len(fixable):>4d}')
    print(f'Unfixable:  {len(unfixable):>4d}')
    print(f'Errors:     {len(errors):>4d}  (empty/stub files)')

    if stale:
        print('\nStale sessions (resume skipped newer compaction segments):')
        for r in stale:
            proj = r.file_path.parent.name
            segs = len(r.stale_segments)
            print(f'  {r.file_path.stem}  {segs} skipped segment{"s" if segs != 1 else ""}  [{proj}]')
        print('\nStale sessions require manual rewiring (see `check <id>` for details).')

    if fixable:
        # Summarize causes
        total_dup = sum(r.duplicate_uuid_orphan_count for r in fixable)
        total_ps = sum(r.prompt_suggestion_count for r in fixable)
        total_other = sum(r.other_agent_count for r in fixable)
        total_unattr = sum(r.unattributed_count for r in fixable)
        causes = []
        if total_dup:
            causes.append(f'{total_dup} duplicate_uuid')
        if total_ps:
            causes.append(f'{total_ps} prompt_suggestion')
        if total_other:
            causes.append(f'{total_other} other agent')
        if total_unattr:
            causes.append(f'{total_unattr} unattributed')

        print(f'\nFixable sessions ({", ".join(causes)}):')
        for r in fixable:
            proj = r.file_path.parent.name
            print(f'  {r.file_path.stem}  {len(r.orphans):>2d} orphans  [{proj}]')

    if unfixable:
        print('\nUnfixable sessions:')
        for r in unfixable:
            proj = r.file_path.parent.name
            print(f'  {r.file_path.stem}  {len(r.orphans):>2d} orphans  [{proj}]  ({r.total_records} records)')

    if fixable:
        print(f"\nRun 'fix_broken_session_resume.py fix --all' to fix all {len(fixable)} fixable sessions.")

    # Return exit code based on findings
    if unfixable:
        return EXIT_UNFIXABLE
    if fixable or stale:
        return EXIT_FIXABLE
    return EXIT_HEALTHY


def cmd_check(args: argparse.Namespace) -> int:
    """Check a specific session."""
    if args.file:
        file_path = args.file
    else:
        file_path = find_session_file(args.session_id)
        if not file_path:
            print(f"Error: session '{args.session_id}' not found", file=sys.stderr)
            return EXIT_ERROR

    r = analyze_session(file_path)
    print_check(r)
    return r.exit_code


def cmd_fix(args: argparse.Namespace) -> int:
    """Fix session(s) by rewiring orphan parentUuids."""
    if args.all:
        files = find_all_session_files()
        targets: list[AnalysisResult] = []
        for f in files:
            r = analyze_session(f)
            if r.status == 'fixable':
                targets.append(r)

        if not targets:
            print('No fixable sessions found.')
            return EXIT_HEALTHY

        print(f'Fixing {len(targets)} sessions...\n')
        failed_count = 0
        for r in targets:
            try:
                backup = apply_fix(r.file_path, r.orphans)
            except FileExistsError as e:
                print(f'  [SKIP] {r.file_path.stem}  {e}', file=sys.stderr)
                failed_count += 1
                continue

            # Verify
            verify = analyze_session(r.file_path)
            if verify.status == 'healthy':
                print(f'  [OK] {r.file_path.stem}  {len(r.orphans)} orphans rewired  backup={backup.name}')
            else:
                # Rollback on verification failure
                shutil.copy2(backup, r.file_path)
                # Clean up stale backup so re-fix is possible
                backup.unlink(missing_ok=True)
                meta_path = BACKUP_DIR / f'{r.file_path.stem}.meta.json'
                meta_path.unlink(missing_ok=True)
                print(
                    f'  [FAILED] {r.file_path.stem}  verification failed, rolled back  status={verify.status}',
                    file=sys.stderr,
                )
                failed_count += 1

        success_count = len(targets) - failed_count
        if success_count > 0:
            print(f'\nBackups saved to {BACKUP_DIR}/')
            print("Run 'fix_broken_session_resume.py restore <session-id>' to undo.")

        if failed_count:
            print(f'\nWarning: {failed_count} sessions could not be fixed (rolled back)', file=sys.stderr)
            return EXIT_ERROR
        return EXIT_HEALTHY

    # Single session
    file_path = find_session_file(args.session_id)
    if not file_path:
        print(f"Error: session '{args.session_id}' not found", file=sys.stderr)
        return EXIT_ERROR

    r = analyze_session(file_path)
    if r.status == 'healthy':
        print(f'Session {file_path.stem} is already healthy, nothing to fix.')
        return EXIT_HEALTHY
    if r.status == 'stale':
        print(f'Session {file_path.stem} has stale-parent resume (not auto-fixable).')
        print_check(r)
        return EXIT_FIXABLE
    if r.status == 'unfixable':
        print(f"Session {file_path.stem} is unfixable (rewiring doesn't produce a clean chain).")
        print_check(r)
        return EXIT_UNFIXABLE
    if r.status == 'error':
        print(f'Error analyzing session: {r.error_message}', file=sys.stderr)
        return EXIT_ERROR

    try:
        backup = apply_fix(file_path, r.orphans)
    except FileExistsError as e:
        print(f'Error: {e}', file=sys.stderr)
        return EXIT_ERROR

    verify = analyze_session(file_path)
    if verify.status == 'healthy':
        print(f'Fixed {file_path.stem}: {len(r.orphans)} orphans rewired')
        print(f'Backup: {backup}')
    else:
        print(f'WARNING: Fix applied but verification failed (status={verify.status})', file=sys.stderr)
        print('Restoring from backup...', file=sys.stderr)
        shutil.copy2(backup, file_path)
        # Clean up stale backup so re-fix is possible
        backup.unlink(missing_ok=True)
        meta_path = BACKUP_DIR / f'{file_path.stem}.meta.json'
        meta_path.unlink(missing_ok=True)
        print('Restored. Original file unchanged.', file=sys.stderr)
        return EXIT_ERROR

    return EXIT_HEALTHY


def cmd_restore(args: argparse.Namespace) -> int:
    """Restore a session from backup."""
    if args.list:
        if not BACKUP_DIR.exists():
            print('No backups found.')
            return EXIT_HEALTHY

        metas = sorted(BACKUP_DIR.glob('*.meta.json'))
        if not metas:
            print('No backups found.')
            return EXIT_HEALTHY

        print(f'Available backups ({BACKUP_DIR}):\n')
        for meta_path in metas:
            meta = json.loads(meta_path.read_text())
            print(f'  {meta["session_id"]}')
            print(f'    Fixed:    {meta["fixed_at"]}')
            print(f'    Orphans:  {meta["orphans_fixed"]}')
            print(f'    Original: {meta["original_path"]}')
            print()
        return EXIT_HEALTHY

    if not args.session_id:
        print('Error: provide a session ID or use --list', file=sys.stderr)
        return EXIT_ERROR

    # Find backup by prefix
    if not BACKUP_DIR.exists():
        print(f'Error: no backups found in {BACKUP_DIR}', file=sys.stderr)
        return EXIT_ERROR

    matches = list(BACKUP_DIR.glob(f'{args.session_id}*.meta.json'))
    if not matches:
        print(f"Error: no backup found for '{args.session_id}'", file=sys.stderr)
        return EXIT_ERROR
    if len(matches) > 1:
        print(f"Error: ambiguous prefix '{args.session_id}', matches:", file=sys.stderr)
        for m in matches:
            print(f'  {m.stem.removesuffix(".meta")}', file=sys.stderr)
        return EXIT_ERROR

    meta = json.loads(matches[0].read_text())
    session_id = meta['session_id']
    original_path = Path(meta['original_path'])
    backup_path = BACKUP_DIR / f'{session_id}.jsonl'

    if not backup_path.exists():
        print(f'Error: backup file missing: {backup_path}', file=sys.stderr)
        return EXIT_ERROR

    shutil.copy2(backup_path, original_path)

    # Verify restore before cleaning up backup
    if not original_path.exists() or original_path.stat().st_size != backup_path.stat().st_size:
        print(f'Error: restore verification failed, backup preserved at {backup_path}', file=sys.stderr)
        return EXIT_ERROR

    # Clean up backup only after successful verification
    backup_path.unlink()
    matches[0].unlink()

    print(f'Restored {session_id} from backup')
    print(f'  -> {original_path}')
    return EXIT_HEALTHY


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Diagnose and repair parentUuid chain integrity in Claude Code session files.',
    )
    subparsers = parser.add_subparsers(dest='command')

    # scan
    subparsers.add_parser('scan', help='Assess all sessions, print summary')

    # check
    check_p = subparsers.add_parser('check', help='Detailed diagnosis of one session')
    check_g = check_p.add_mutually_exclusive_group(required=True)
    check_g.add_argument('session_id', nargs='?', help='Session ID or prefix')
    check_g.add_argument('--file', '-f', type=Path, help='Direct path to session JSONL')

    # fix
    fix_p = subparsers.add_parser('fix', help='Fix session(s) by rewiring orphan parentUuids')
    fix_p.add_argument('session_id', nargs='?', help='Session ID or prefix')
    fix_p.add_argument('--all', action='store_true', help='Fix all fixable sessions')

    # restore
    restore_p = subparsers.add_parser('restore', help='Restore a session from backup')
    restore_p.add_argument('session_id', nargs='?', help='Session ID or prefix')
    restore_p.add_argument('--list', action='store_true', help='List available backups')

    args = parser.parse_args()

    if args.command == 'scan':
        return cmd_scan(args)
    elif args.command == 'check':
        return cmd_check(args)
    elif args.command == 'fix':
        if not args.all and not args.session_id:
            fix_p.error('provide a session ID or use --all')
        return cmd_fix(args)
    elif args.command == 'restore':
        return cmd_restore(args)
    else:
        parser.print_help()
        return EXIT_ERROR


if __name__ == '__main__':
    sys.exit(main())
