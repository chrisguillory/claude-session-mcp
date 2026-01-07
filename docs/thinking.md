# Extended Thinking: Research Findings

Research notes on Claude Code's extended thinking system, compiled from session file analysis, API traffic capture, and community findings.

> **January 2026 Update**: API traffic analysis confirmed that `ultrathink` no longer changes API behavior. Boris Cherny (Claude Code creator) stated on X: "Thinking is on by default everywhere, ULTRATHINK doesn't really do anything anymore." See [Verification](#verification-january-2026) section for methodology.

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

The `start` and `end` fields record character positions where trigger keywords appeared in the user's original message.

**Purpose of position tracking:**
1. **Rainbow animation** - Triggers the colorful "ultrathink" visual effect in the CLI
2. **Session audit trail** - Records what activated thinking mode
3. **Future UI features** - Infrastructure for potential highlighting/visualization

> **Correction (January 2026)**: Earlier versions of this document claimed keywords are stripped before API calls. API traffic capture proved this is **false**—the keyword is sent verbatim to the model. The `thinkingMetadata` structure is stored locally only and is not transmitted to Anthropic's servers.

## Trigger Keywords

Only four trigger keywords have ever existed:

| Keyword | Token Budget | Status (Jan 2026) |
|---------|--------------|-------------------|
| `think` | ~4,000 | Deprecated |
| `think hard` | ~10,000 | Deprecated |
| `think harder` | ~20,000-24,000 | Deprecated |
| `ultrathink` | 31,999 (default max) | **No effect** (thinking is default) |

Token budgets are approximations from community testing, not official specs. The `MAX_THINKING_TOKENS` environment variable can override the default maximum.

> **Current behavior (January 2026)**: Extended thinking with `budget_tokens: 31999` is enabled by default for all Opus 4.5 requests in Claude Code. The `ultrathink` keyword is detected (for UI animation) but does not change API parameters since they're already at maximum.

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

**Trigger keywords are only detected in Claude Code CLI.**

They do not function in:
- VS Code extension (uses toggle only)
- claude.ai web interface (uses UI toggle)
- Anthropic API (uses `thinking` parameter directly)

The CLI detects keywords for UI effects (rainbow animation) but does not modify the API request based on them—thinking is already enabled by default.

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

## Verification (January 2026)

On January 7, 2026, we conducted definitive API traffic analysis to verify whether `ultrathink` affects Claude Code behavior.

### Methodology

1. **Proxy capture**: Used mitmproxy to intercept Claude Code API traffic
2. **Three test conditions**: Same prompt ("Discover yourself the formula for 10! And prove it.")
   - Baseline: No prefix
   - `ultrathink:` prefix
   - `ultra think:` prefix (space—control for pattern matching)
3. **Data sources**: Both captured API requests AND local session JSONL files

### Findings

| Test | Local `thinkingMetadata.level` | Local `triggers` | API `thinking.budget_tokens` |
|------|-------------------------------|------------------|------------------------------|
| Baseline | `high` | `[]` | `31999` |
| `ultrathink:` | `high` | `[{text: "ultrathink", start: 0, end: 10}]` | `31999` |
| `ultra think:` | `high` | `[]` | `31999` |

**Key observations:**
- All three tests sent **identical** `thinking` parameters to the API
- The `ultrathink` keyword was detected locally (ThinkingTrigger recorded) but did **not** change API params
- The keyword was **not stripped**—it appeared verbatim in the API request: `"ultrathink: Discover yourself..."`
- `thinkingMetadata` is stored locally only; it is not transmitted to Anthropic's servers

### Confirmation

Boris Cherny (Claude Code creator) confirmed on X (January 4, 2026):
> "Thinking is on by default everywhere, ULTRATHINK doesn't really do anything anymore"

When asked about the rainbow animation, he replied:
> "Yes, we've been dragging our feet on [removing the colors] for sentimental reasons"

### What ultrathink Still Does

1. ✅ Triggers client-side detection (ThinkingTrigger recorded)
2. ✅ Activates rainbow animation in CLI
3. ❌ Does NOT modify API parameters (already at max)
4. ❌ Is NOT stripped from prompt (sent verbatim to model)

## References

- [Extended thinking docs](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
- [Claude Code best practices](https://www.anthropic.com/engineering/claude-code-best-practices)
- [GitHub issues](https://github.com/anthropics/claude-code/issues?q=thinking)
- [Boris Cherny's confirmation](https://x.com/bcherny/status/2007892431031988385) (January 4, 2026)