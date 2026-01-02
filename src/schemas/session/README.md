# Session Schema

Pydantic models for Claude Code session JSONL records (schema v0.1.9).

## Current State

All models are in `models.py` (~1,100 lines). This works but will eventually be split for better navigability.

## Planned Split

When the file becomes unwieldy or we need granular imports:

```
session/
├── __init__.py       # Re-exports (unchanged interface)
├── records.py        # UserRecord, AssistantRecord, SystemRecord variants
├── content.py        # MessageContent types (text, thinking, tool_use, etc.)
├── message.py        # Message, TokenUsage, CacheCreation
├── tools/
│   ├── inputs.py     # Tool input models (ReadToolInput, WriteToolInput, etc.)
│   └── results.py    # Tool result models (BashToolResult, etc.)
├── types.py          # ModelId, StopReason, etc. (move from schemas/types.py)
└── markers.py        # PathMarker, PathField (already split)
```

## Validation

```bash
uv run scripts/validate_models.py
```

Expects 100% pass rate across all session files.

## Key Design Patterns

- **Discriminated unions**: `SessionRecord` uses `union_mode='left_to_right'` because multiple records share `type='system'`
- **Frozen models**: All `StrictModel` subclasses are immutable; use `validated_copy()` for modifications
- **PathMarker annotations**: Fields marked with `PathField` are translated during cross-machine restore
- **Reserved fields**: Fields typed as `None` (not `Any | None`) indicate Claude API schema reserves but doesn't populate them
