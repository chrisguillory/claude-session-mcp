# Session Memory: Claude Code's Background Summarization System

Session memory is an experimental Claude Code feature that automatically maintains
structured notes about the current session. These notes can then replace traditional
compaction, making context management faster and more informative.

## Overview

| Aspect | Session Memory | Auto-Memory (MEMORY.md) | CLAUDE.md |
|--------|---------------|------------------------|-----------|
| Scope | Per-session | Per-project | Per-project |
| Author | Claude (background) | Claude (via Edit/Write) | User-authored |
| Survives session? | On disk, may be recalled | Always loaded | Always loaded |
| Content | What happened THIS session | Cross-session learnings | Instructions & conventions |
| Size limit | 12,000 tokens total | 200 lines loaded | No hard limit |
| Storage | `<sid>/session-memory/summary.md` | `memory/MEMORY.md` | `.claude/CLAUDE.md` |

Session memory is stored per-session under the project directory:

```
~/.claude/projects/<encoded-path>/
  <session-id>/
    session-memory/
      summary.md          # Structured session notes
```

## How It Works

### The Two Feature Flags

Session memory is controlled by two independent Statsig feature flags:

| Flag | Purpose | Effect |
|------|---------|--------|
| `tengu_session_memory` | Background note-taking | Writes summary.md incrementally |
| `tengu_sm_compact` | Summary-based compaction | Uses summary.md for instant compaction |

Both default to `false`. They can be enabled via environment variables (see
[Enabling Session Memory](#enabling-session-memory)).

**Flag 1 alone** (`tengu_session_memory`): Claude Code runs background model calls
to maintain session notes. No effect on compaction — the notes are purely
informational.

**Both flags together**: Compaction uses the pre-built summary instead of running
an expensive re-summarization. Recent messages are preserved verbatim.

### Background Note-Taking (`tengu_session_memory`)

After each conversation turn, Claude Code checks whether the session has crossed
token and tool-call thresholds:

| Threshold | Value | Purpose |
|-----------|-------|---------|
| `minimumMessageTokensToInit` | ~140,000 | Don't start until 70% of context used |
| `minimumTokensBetweenUpdate` | ~10,000 | Update cadence (tokens) |
| `toolCallsBetweenUpdates` | ~5 | Update cadence (tool calls) |

These values are remotely configurable via Statsig experiments. The 140K init
threshold means session memory only starts building notes when you're deep into
a long session — it's designed as a pre-compaction optimization.

When thresholds are crossed:

1. A **background model call** runs (separate from the main conversation)
2. The model receives the current `summary.md` contents
3. It uses the **Edit tool** to update specific sections incrementally
4. Each section is capped at ~2,000 tokens; the total file at ~12,000 tokens

The model does NOT rewrite the whole file each time — it surgically updates
sections that changed, making updates efficient.

### Summary-Based Compaction (`tengu_sm_compact`)

When both flags are enabled, auto-compact works differently:

**Traditional compaction:**
1. Context hits ~200K limit
2. Expensive model call to summarize ALL messages
3. ALL messages discarded, replaced with summary
4. Model continues with just the summary (~5-10K tokens)

**SM compaction:**
1. Context hits limit
2. Claude Code reads the pre-built `summary.md` (instant, no model call)
3. Messages BEFORE a cut point are replaced with the summary
4. Messages AFTER the cut point (recent work) are preserved **verbatim**
5. Model continues with summary + recent context

The preserved window is configurable:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `minTokens` | 10,000 | Minimum tokens to keep verbatim |
| `minTextBlockMessages` | 5 | Minimum text messages to keep |
| `maxTokens` | 40,000 | Maximum tokens in preserved window |

This means after SM compaction, you retain both a structured summary of earlier
work AND the full verbatim context of your most recent 10-40K tokens of work.

## The Summary Template

Session memory uses a fixed 10-section markdown template:

```markdown
# Session Title
_A short and distinctive 5-10 word descriptive title for the session._

# Current State
_What is actively being worked on right now? Pending tasks not yet completed._

# Task Specification
_What did the user ask to build? Design decisions or explanatory context._

# Files and Functions
_Important files, what they contain, and why they're relevant._

# Workflow
_Bash commands usually run and in what order._

# Errors & Corrections
_Errors encountered and how they were fixed. Failed approaches._

# Codebase and System Documentation
_Important system components. How they work/fit together._

# Learnings
_What has worked well? What has not? What to avoid?_

# Key Results
_Specific outputs: answers, tables, documents._

# Worklog
_Step by step, what was attempted and done. Very terse._
```

Each section header includes an italicized description line that acts as a
template instruction for the background model.

### Custom Templates

You can override the default template by creating files at:

```
~/.claude/session-memory/config/
  template.md    # Custom section structure
  prompt.md      # Custom generation instructions
```

The `prompt.md` file supports placeholders:
- `{{notesPath}}` — path to the summary.md file
- `{{currentNotes}}` — current contents of summary.md

These files are optional. If absent, the built-in defaults are used.

**Note**: These config directories do NOT auto-create when the feature is enabled.
Create them manually if you want customization.

## Enabling Session Memory

### Three Gates, Not Two

Session memory extraction has three prerequisites that must ALL be satisfied:

| Gate                       | What It Controls                                | How to Enable                                           |
|----------------------------|-------------------------------------------------|---------------------------------------------------------|
| **Autocompact enabled**    | Hook registration at startup (`FY7()` → `uL()`) | Remove `DISABLE_AUTO_COMPACT` from settings.json env    |
| **`tengu_session_memory`** | Per-turn extraction check (`Lw4()`)             | tweakcc `--patches "session-memory"` or Statsig rollout |
| **Token/tool thresholds**  | When extraction actually fires (`Sw4()`)        | Automatic (140K tokens + 5 tool calls)                  |

The most common blocker: **`DISABLE_AUTO_COMPACT=true` in settings.json**
silently prevents session memory hook registration. The extraction hook
(`FY7()`) calls `uL()` at startup, which returns `false` if autocompact is
disabled. The hook is never registered, so extraction never fires — regardless
of whether the Statsig gate is patched.

### The Producer-Consumer Gap

There are two components that must BOTH be active for the full pipeline:

| Component                 | What It Does                                   | Control Mechanism                                        |
|---------------------------|------------------------------------------------|----------------------------------------------------------|
| **Producer** (extraction) | Background model calls that write `summary.md` | Autocompact gate + `tengu_session_memory` Statsig flag   |
| **Consumer** (SM compact) | Uses `summary.md` for instant compaction       | `ENABLE_CLAUDE_CODE_SM_COMPACT` env var OR Statsig flags |

**The env var only enables the consumer.** Setting `ENABLE_CLAUDE_CODE_SM_COMPACT=true`
tells Claude Code "if a summary.md exists, use it for compaction" — but nothing
creates the summary.md because the background extraction has no env var bypass.

If SM compact is enabled but no summary.md exists, Claude Code emits
`tengu_sm_compact_no_session_memory` and falls back to traditional compaction.

### Auto-Compact Threshold Configuration

If you enable autocompact (required for session memory) but don't want it
triggering prematurely, these env vars control the thresholds:

| Env Var                               | Purpose                        | Default                                        | Notes                                                                             |
|---------------------------------------|--------------------------------|------------------------------------------------|-----------------------------------------------------------------------------------|
| `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`     | Auto-compact trigger threshold | 92% (from Statsig `tengu_auto_compact_config`) | Clamped to range (0, 100]. At 100%, threshold equals the effective context window |
| `CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE` | Hard API blocking limit        | ~197K (window - 3000)                          | Set higher to allow the API to reject before auto-compact fires                   |
| `DISABLE_AUTO_COMPACT`                | Kill switch for autocompact    | `false`                                        | **Also kills session memory extraction**                                          |
| `DISABLE_COMPACT`                     | Kill switch for ALL compaction | `false`                                        | Disables both auto and manual `/compact`                                          |

**How `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` works** (from binary analysis):

```javascript
function qiT(model) {
    let effectiveWindow = n4T(model);         // ~180K for 200K context
    let defaultThreshold = effectiveWindow - 13000;  // ~167K
    let pctOverride = process.env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE;
    if (pctOverride) {
        let pct = parseFloat(pctOverride);
        if (!isNaN(pct) && pct > 0 && pct <= 100) {   // ← CAPPED AT 100
            let scaled = Math.floor(effectiveWindow * (pct / 100));
            return Math.min(scaled, defaultThreshold);  // ← CLAMPED to default
        }
    }
    return defaultThreshold;
}
```

At 100%, the `Math.min` clamp means the threshold equals the default (~167K).
You cannot set it above 100% to prevent auto-compact entirely. However,
combined with `CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE`, you can push the API
blocking limit far beyond 200K, allowing the API itself to reject the prompt
before auto-compact fires.

**Recommended configuration** for session memory with minimal auto-compact
interference:

```json
{
  "env": {
    "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "100",
    "CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE": "503000",
    "ENABLE_CLAUDE_CODE_SM_COMPACT": "true"
  }
}
```

This keeps autocompact enabled (so session memory works) with the threshold
at its maximum. The blocking limit override lets you go well past 200K before
Claude Code prevents API calls. Session memory builds its summary in the
background starting at 140K, ready for whenever compaction does happen.

### Environment Variable (Consumer Only)

```bash
# Enables SM compact — will use summary.md IF one exists
export ENABLE_CLAUDE_CODE_SM_COMPACT=true
```

This is useful in combination with other approaches that enable the producer.
On its own, it has no visible effect until `tengu_session_memory` is also enabled.

### tweakcc Session Memory Patch Details

[tweakcc](https://github.com/Piebald-AI/tweakcc) (MIT, 1,200+ stars) patches
Claude Code's compiled JavaScript to force-enable gated features.

```bash
npx tweakcc --apply --patches "session-memory"   # Targeted patch
npx tweakcc --apply                               # Reapply after update
npx tweakcc --restore                             # Revert all patches
```

The `session-memory` patch makes 4 changes:

| Change                              | What It Does                                         |
|-------------------------------------|------------------------------------------------------|
| `Lw4()` returns `true`              | Bypasses `tengu_session_memory` Statsig gate         |
| `tengu_coral_fern` gate → `true`    | Enables the `/remember` skill (past session search)  |
| SM thresholds → `CC_SM_*` env vars  | Allows overriding extraction thresholds              |
| SM file limits → `CC_SM_*` env vars | Allows overriding per-section and total token limits |

After patching, these env vars become available for threshold tuning:

| Env Var                            | Default | Purpose                        |
|------------------------------------|---------|--------------------------------|
| `CC_SM_MIN_TOKENS_TO_INIT`         | 140000  | Tokens before first extraction |
| `CC_SM_MIN_TOKENS_BETWEEN_UPDATE`  | 10000   | Tokens between updates         |
| `CC_SM_TOOL_CALLS_BETWEEN_UPDATES` | 5       | Tool calls between updates     |
| `CC_SM_PER_SECTION_LIMIT`          | 2000    | Max tokens per section         |
| `CC_SM_TOTAL_FILE_LIMIT`           | 12000   | Max tokens for entire summary  |

**Critical caveat**: The patch does NOT bypass the autocompact gate in `FY7()`.
You must have autocompact enabled (`DISABLE_AUTO_COMPACT` must NOT be set) for
the extraction hook to register at startup.

**Other considerations:**
- Patches survive Statsig server refreshes (code is patched, not cache)
- Patches break on every Claude Code update (reapply with `--apply`)
- Anthropic's Consumer ToS prohibits reverse engineering, though Anthropic has
  not acted against tweakcc in 7+ months despite its public profile

### Statsig Feature Flags

The flags `tengu_session_memory` and `tengu_sm_compact` are controlled server-side
by Anthropic via Statsig. The cache is at `~/.claude/statsig/statsig.cached.evaluations.<hash>`.

Gate names are hashed with **unsalted DJB2**:

```python
def djb2(s: str) -> str:
    h = 0
    for c in s:
        h = ((h << 5) - h + ord(c)) & 0xFFFFFFFF
    return str(h)

# djb2("tengu_session_memory") → "3695724478"
# djb2("tengu_sm_compact")     → "370447666"
```

Claude Code also maintains a parallel `cachedGrowthBookFeatures` object in
`~/.claude.json` with **plain-text flag names** — easier to inspect:

```bash
python3 -c "import json; d=json.load(open('$HOME/.claude.json')); print(json.dumps({k:v for k,v in d.get('cachedGrowthBookFeatures',{}).items() if 'session' in k or 'compact' in k}, indent=2))"
```

Both caches are refreshed from Anthropic's servers on every Claude Code startup,
so manual edits are overwritten immediately.

### Via `CLAUDE_CODE_SIMPLE`

Setting `CLAUDE_CODE_SIMPLE=true` **disables** session memory (among other
features like custom agents and skills). Avoid this if you want session memory.

### Wait for Official Rollout

Session memory is in gradual rollout. The `tengu_session_memory` flag shows a
`0.00:10` rollout rule (0% of the current cohort). Given that the feature has
been in development since Nov 2025 and Anthropic's documentation already
references it publicly, broader availability is likely in the near term.

## Version History

| Version                | Event                                                       |
|------------------------|-------------------------------------------------------------|
| ~v2.0.58 (Nov 2025)    | Infrastructure appears in binary                            |
| v2.0.64 (Dec 9, 2025)  | Feature flag enabled for some users                         |
| v2.0.65 (Dec 11, 2025) | Feature flag rolled back for most users                     |
| v2.1.50 (Feb 2026)     | `CLAUDE_CODE_SIMPLE` documented as disabling session memory |
| v2.1.59 (Feb 2026)     | Auto-memory (MEMORY.md) formally documented                 |
| v2.1.63                | Code fully present, gated by feature flags                  |
| v2.1.69 (current)      | Unchanged; autocompact dependency on extraction confirmed   |
| v2.1.70                | Fixed stale Statsig cache not refreshing on startup         |

## Known Issues

- **Race condition** (GitHub #15097): Auto-compact can trigger before session
  memory extraction finishes. The wait function checks a timestamp, but if
  extraction hasn't started yet (due to debouncing), the wait returns immediately
  and compaction proceeds with a stale summary.

- **Feature flag churn**: The feature has been in a prolonged A/B test. Users
  may see it enabled briefly then disabled without notice.

## Relationship to This Project

### Session Memory as an Artifact

Session memory files are project-specific, session-ID-keyed artifacts. As of
the `move_session` commit, all session operations handle them:

| Operation | Handling |
|-----------|---------|
| **Archive** | Captured in `SessionArchiveV2` (if present) |
| **Clone** | Copied to cloned session |
| **Restore** | Restored from archive |
| **Delete** | Backed up then deleted |
| **Move** | Relocated to target project |

The artifact module is at `src/services/artifacts/session_memory.py`.

### Implications for Session Analysis

Session memory summaries are high-value structured data. A 26KB summary can
capture the essential state of a session whose JSONL transcript is several MB.
Potential uses:

- **Session search**: Search summaries instead of raw JSONL (much less noise)
- **Session info display**: Show the "Current State" section as a session preview
- **Compaction analysis**: Compare traditional compact summaries vs session memory
  summaries for quality and completeness

## Telemetry Events

Claude Code emits these session-memory-related telemetry events:

| Event | Meaning |
|-------|---------|
| `tengu_session_memory` | Feature accessed |
| `tengu_session_memory_extraction` | Background extraction triggered |
| `tengu_session_memory_loaded` | Summary loaded into context |
| `tengu_sm_compact` | SM compact flow entered |
| `tengu_sm_compact_empty_template` | Bailed: summary is just empty template |
| `tengu_sm_compact_no_session_memory` | Bailed: no summary.md exists |
| `tengu_sm_compact_threshold_exceeded` | Post-compact tokens exceed threshold |
| `tengu_sm_compact_resumed_session` | SM compact on a resumed session |

## References

- [GitHub #13688](https://github.com/anthropics/claude-code/issues/13688) — Session memory files stopped being created (v2.0.65)
- [GitHub #15097](https://github.com/anthropics/claude-code/issues/15097) — Race condition between extraction and compaction
- [GitHub #14227](https://github.com/anthropics/claude-code/issues/14227) — Feature request: Claude Code starts every session with zero context
- [Piebald-AI system prompts](https://github.com/Piebald-AI/claude-code-system-prompts) — Community-extracted session memory template
- [Anthropic Cookbook](https://platform.claude.com/cookbook/misc-session-memory-compaction) — Session memory compaction pattern