# Claude Session MCP - Development Guide

## Session File Analysis

Session files are JSONL at `~/.claude/projects/<encoded-path>/<session-id>.jsonl`.

### Checking Token Usage

Token counts are in `message.usage` on assistant records:

```python
total_tokens = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
```

This is the **message context** only. Full context includes overhead:

| Component | Typical Size | Notes |
|-----------|--------------|-------|
| System prompt | ~3k | Base Claude Code instructions |
| System tools | ~16k | Built-in tools |
| MCP tools | 0-50k+ | Varies by configured MCPs |
| Memory files | varies | CLAUDE.md files |
| Messages | varies | Conversation history |

Run `/context` in Claude Code for authoritative breakdown.

### Finding Compaction Events

```bash
rg -n "compact_boundary" <session-file>.jsonl
```

Each has `compactMetadata.trigger` (usually "manual") and `compactMetadata.preTokens`.

### "Prompt is too long" Errors

When the API rejects a prompt, Claude Code logs a synthetic response:

```json
{
  "type": "assistant",
  "message": {
    "model": "<synthetic>",
    "content": [{"type": "text", "text": "Prompt is too long"}]
  }
}
```

This indicates context exceeded ~200k. Check preceding records for actual token counts.

### Cache Invalidation

`cache_read_input_tokens` drops significantly when:
- User edits a previous message (branches conversation)
- User goes back to retry from earlier point
- Session structure changes

This is NOT compaction - the tokens are still there, just not cached.

## Validation

```bash
uv run ./scripts/validate_models.py
```

Validates all session files against Pydantic models. 100% pass rate expected.

## Analyzing Session Token Usage

**Always use Pydantic models** for session analysis - never raw `json.loads()` without validation. The models provide type safety, IDE autocomplete, and catch schema changes early:

```python
from pathlib import Path
from pydantic import TypeAdapter, BaseModel
from src.models import SessionRecord, UserRecord, AssistantRecord, CompactBoundarySystemRecord
import json

adapter = TypeAdapter(SessionRecord)

# Parse session with Pydantic
records = []
with open(session_path) as f:
    for line in f:
        record = adapter.validate_python(json.loads(line))
        records.append(record)

# Find last compact_boundary
last_compact_idx = None
for i, record in enumerate(records):
    if isinstance(record, CompactBoundarySystemRecord):
        last_compact_idx = i
        print(f"Compact at {i}: preTokens={record.compactMetadata.preTokens}")

# Analyze records after compact
for record in records[last_compact_idx + 1:]:
    if isinstance(record, UserRecord) and record.message:
        for block in record.message.content:
            if block.type == 'tool_result':
                # block.content is the tool output
                pass
            elif block.type == 'text':
                # block.text is user message
                pass

    elif isinstance(record, AssistantRecord) and record.message:
        for block in record.message.content:
            if block.type == 'thinking':
                # block.thinking - NOT counted in context
                pass
            elif block.type == 'text':
                # block.text - assistant response
                pass
            elif block.type == 'tool_use':
                # block.input - Pydantic model, use .model_dump_json()
                pass
```

### Preserving Context Before Compact

Before running `/compact`, ask Claude to generate a handoff document:

> "Make a self-contained handoff artifact that the next AI can use to continue. Include executive summary, current state, pending work, and key decisions."

This produces a structured document (~12k chars) that's **larger and more useful** than the auto-generated compact summary (~9k chars). The handoff survives in session history as additional context.

The compact summary is stored in the first user message after `compact_boundary` as `message.content` (string, not list).

### Estimating Message Tokens

Claude's tokenizer differs from tiktoken, but rough estimates:

| Content Type | In Context? | Estimate |
|--------------|-------------|----------|
| Thinking blocks | NO | Generated output, not input |
| Tool results | ~45% | Truncated/compressed |
| User/assistant text | ~100% | Full content |
| Tool use inputs | ~100% | Full content |

## Operational Patterns

### Finding Session Files

Locate a session file by ID prefix:

```bash
find ~/.claude/projects -name "019b342b*.jsonl"
```

### One-off Dependencies

Use `uv run --with` for temporary dependencies without modifying pyproject.toml:

```bash
uv run --with tiktoken python3 << 'EOF'
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
# ...
EOF
```

### Large Outputs to Files

When inline response is too limited (~16k chars), write to a file instead:

```python
# Use Write tool to create handoff-<session-id>.md
# Can produce 30k+ char documents vs ~16k inline limit
```

### Comparing Handoff vs Compact

Measure preservation strategy effectiveness:

```python
# Find handoff document (large assistant text before compact)
# Find compact summary (first user message after compact_boundary, stored as string)
# Compare char lengths - handoff typically ~25% larger
```

## Schema Updates

When Claude Code updates break validation:
1. Run validate_models.py to find failures
2. Check field paths in error messages
3. Update `src/models.py` with new fields/types
4. Bump schema version in models.py header