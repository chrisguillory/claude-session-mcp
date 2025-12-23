# Extended Thinking: Research Findings

Research notes on Claude Code's extended thinking system, compiled from session file analysis, GitHub issues, and community findings (December 2025).

## ThinkingMetadata Structure

Session files record thinking configuration when users activate extended thinking:

```json
{
  "thinkingMetadata": {
    "level": "high",
    "disabled": false,
    "triggers": [
      {
        "start": 53,
        "end": 63,
        "text": "ultrathink"
      }
    ]
  }
}
```

**Key fields:**
- `level`: Thinking intensity (`none`, `low`, `medium`, `high`)
- `disabled`: Whether thinking is globally disabled for the session
- `triggers`: Array of trigger keywords with position information

This structure is **not officially documented** by Anthropic and may change in future versions.

## Position Tracking

The `start` and `end` fields record character positions where trigger keywords appeared in the user's original message. This exists because:

1. **Keywords are stripped before API calls** - The preprocessing layer removes trigger words before constructing the API request
2. **Original prompt reconstruction** - Position data allows rebuilding what the user actually typed
3. **Future UI features** - Infrastructure for potential highlighting/visualization
4. **Debug support** - Helps identify trigger detection edge cases

The model never sees trigger keywords - they're client-side commands only.

## Trigger Keywords

Only four trigger keywords have ever existed:

| Keyword | Token Budget | Status (Dec 2025) |
|---------|--------------|-------------------|
| `think` | ~4,000 | Deprecated |
| `think hard` | ~10,000 | Deprecated |
| `think harder` | ~20,000-24,000 | Deprecated |
| `ultrathink` | 31,999 (default max) | Active |

Token budgets are approximations from community testing, not official specs. The `MAX_THINKING_TOKENS` environment variable can override the default maximum.

### Why Keywords Were Deprecated

The intermediate keywords (`think`, `think hard`, `think harder`) were deprecated due to false positive detection in natural language:

- Users typing "I think we should..." unintentionally triggered extended thinking
- This caused unexpected latency and token costs
- GitHub issues #5702, #658, #883 documented the problem
- Fix in v1.0.123 improved negation detection ("don't think" no longer triggers)

`ultrathink` was retained because it's unlikely to appear in natural language and signals clear user intent.

### Version History

- **Pre-October 2025 (V1)**: All four keywords functional
- **October 2025+ (V2)**: Only `ultrathink` reliably supported
- **V2 introduced**: Shift+Tab toggle as primary thinking control mechanism

If analyzing historical session files, older sessions may contain triggers for the deprecated keywords.

## Platform Exclusivity

**Trigger keywords work ONLY in Claude Code CLI.**

They do not function in:
- VS Code extension (uses toggle only)
- claude.ai web interface (uses UI toggle)
- Anthropic API (uses `thinking` parameter directly)

This is architectural - Claude Code CLI has a preprocessing layer that intercepts keywords before constructing API requests. Other interfaces lack this layer.

## Thinking Levels and Token Budgets

The `level` field maps approximately to token budgets:

| Level | Estimated Tokens | Typical Use Case |
|-------|------------------|------------------|
| `high` | 20,000-31,999 | Complex architecture, novel problems |
| `medium` | 8,000-16,000 | Multi-step reasoning |
| `low` | 2,000-6,000 | Moderate complexity |
| `none` | 0 | Thinking disabled |

Community testing suggests diminishing returns above ~10,000 tokens for most coding tasks. Reserve `ultrathink` for genuinely complex problems.

## Case Sensitivity

Trigger detection is case-insensitive for activation, but original casing is preserved in session records. Both `ultrathink` and `UltraThink` may appear in session files.

## Known Documentation Gaps

Open GitHub issues (as of December 2025):

| Issue | Description |
|-------|-------------|
| #9072 | Interaction between Shift+Tab toggle and `ultrathink` keyword undocumented |
| #14321 | Trigger keywords don't work in subagent sessions |
| #8360 | Request for guidance on when to use different thinking levels |

## Configuration Methods

Extended thinking can be configured via:

1. **Keyword in prompt**: `ultrathink: your prompt here`
2. **Keyboard toggle**: Shift+Tab (V2+)
3. **Environment variable**: `MAX_THINKING_TOKENS` sets global maximum
4. **Session config**: `/config` command toggles thinking on/off

## Analyzing Thinking Usage

When examining session files for thinking patterns:

```python
from src.models import SessionRecord, UserRecord
from pydantic import TypeAdapter
import json

adapter = TypeAdapter(SessionRecord)

with open(session_file) as f:
    for line in f:
        record = adapter.validate_python(json.loads(line))
        if isinstance(record, UserRecord) and record.thinkingMetadata:
            meta = record.thinkingMetadata
            print(f"Level: {meta.level}")
            for trigger in meta.triggers:
                if isinstance(trigger, dict):
                    print(f"  Trigger: {trigger['text']} at {trigger['start']}-{trigger['end']}")
```

## References

- [Extended thinking docs](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
- [Claude Code best practices](https://www.anthropic.com/engineering/claude-code-best-practices)
- [GitHub issues](https://github.com/anthropics/claude-code/issues?q=thinking)