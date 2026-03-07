# Claude Code Feature Flags & tweakcc

Research report on controlling Claude Code's feature flag system, with a focus
on enabling session memory. Produced March 2, 2026.

## Executive Summary

- **tweakcc** (MIT, 1,200+ stars, v4.0.10) patches Claude Code's compiled JS to
  force-enable gated features. It survives Statsig server refreshes but must be
  reapplied after Claude Code updates.
- Statsig gate names use **unsalted DJB2** hashing. We can now map every gate.
- `tengu_session_memory` is **false** across all 4 cached evaluations on this
  system (Dec 19 → Feb 3), with a **0% rollout** rule.
- Session memory files from Dec 9-10 predate our earliest cache (Dec 19), so we
  cannot directly confirm the flag was ever TRUE in cache — only infer from the
  files' existence and community reports.
- Anthropic's ToS prohibits reverse engineering. They DMCA'd deobfuscated source
  publication but have NOT acted against tweakcc (7+ months, 1,200+ stars).

## Feature Flag Architecture

Claude Code uses **two parallel flag systems**, both server-synced on startup:

| System | Storage | Key Format | Refresh |
|--------|---------|------------|---------|
| **Statsig** | `~/.claude/statsig/statsig.cached.evaluations.<hash>` | DJB2-hashed names | Every startup via `statsig.anthropic.com` |
| **GrowthBook** | `~/.claude.json` → `cachedGrowthBookFeatures` | Plain-text names | Every startup |

Both show the same values. GrowthBook is easier to inspect; Statsig is what the
code actually evaluates.

### DJB2 Hash Function

The Statsig SDK uses plain (unsalted) DJB2:

```python
def djb2(s: str) -> str:
    h = 0
    for c in s:
        h = ((h << 5) - h + ord(c)) & 0xFFFFFFFF
    return str(h)
```

Confirmed by matching computed hashes against cache entries. The cache file
itself declares `"hash_used": "djb2"`.

### Key Session Memory Flags

| Flag | DJB2 Hash | Current Value | Rollout Rule |
|------|-----------|--------------|--------------|
| `tengu_session_memory` | `3695724478` | **false** | `0.00:10` (0% rollout) |
| `tengu_sm_compact` | `370447666` | **false** | `default` |

Config: `tengu_sm_config` (hash `1120349011`):
```json
{
  "minimumMessageTokensToInit": 140000,
  "minimumTokensBetweenUpdate": 10000,
  "toolCallsBetweenUpdates": 5
}
```

## Historical Flag Analysis

Four Statsig cache files exist on this system:

| Cache | Received | Total Gates | TRUE Gates | `tengu_session_memory` |
|-------|----------|-------------|------------|----------------------|
| `86edf9d902` | Dec 19, 2025 | **66** | 23 | **false** (rule: `6UZagExcer2lI04UWb6p1i:0.00:3`) |
| `6b2b1873b6` | Jan 29, 2026 | 41 | 19 | **false** (rule: `56rWaA9UThrkHCJufAzHaa:0.00:10`) |
| `2af100616e` | Jan 29, 2026 | 41 | 20 | **false** (rule: `56rWaA9UThrkHCJufAzHaa:0.00:10`) |
| `04d9b7a7f7` | Feb 3, 2026 | 41 | 20 | **false** (rule: `56rWaA9UThrkHCJufAzHaa:0.00:10`) |

### Key Observations

1. **The Dec 19 cache has 66 gates vs 41 in later caches.** 25 gates were removed
   between Dec 19 and Jan 29 — likely temporary experiments from the session
   memory testing period.

2. **The rollout rule changed.** Dec 19 had rule `6UZagExcer2lI04UWb6p1i:0.00:3`
   (3 targeting rules). By Jan 29 it became `56rWaA9UThrkHCJufAzHaa:0.00:10`
   (10 targeting rules). More targeting rules suggests Anthropic is refining which
   user segments get the feature, even while the overall rollout is 0%.

3. **6 gates were TRUE in Dec but gone by Jan.** These unmapped gates
   (`402975144`, `2137706241`, `2271102501`, `2958380928`, `3068840225`,
   `3522193162`) could be temporary experiments active during the session memory
   testing window.

4. **Session memory files predate our earliest cache.** Files exist from Dec 9-10;
   earliest cache is Dec 19. We cannot directly confirm `tengu_session_memory` was
   TRUE during the file creation window. By Dec 19 it was already FALSE.

5. **53 of 66 Dec gates are unmapped.** These use non-`tengu_` naming conventions.
   We only extracted `tengu_*` strings from the binary. The unmapped gates likely
   use different internal naming.

### Timeline Reconstruction

```
Dec  9 14:45  First session-memory file created (underwriting-api)
Dec  9 15:03  Second session-memory file created
Dec 10 07:01  Third session-memory file (claude-session-mcp)
Dec 10 07:35  Fourth session-memory file
Dec 10 08:09  Fifth session-memory file
Dec 10 09:31  Sixth (last) session-memory file
Dec 11        GitHub #13688: "session-memory stopped being created in v2.0.65"
Dec 19 14:53  Earliest Statsig cache on disk — flag already FALSE (0% rollout)
Jan 29        Two cache snapshots — flag FALSE, new rule ID, 10 targeting rules
Feb  3        Most recent cache — flag FALSE, unchanged
```

## tweakcc

### What It Is

A TypeScript CLI tool that modifies Claude Code installations. For native binary
installs (macOS/Linux), it uses [node-lief](https://github.com/Piebald-AI/node-lief)
to extract the embedded JavaScript, apply patches, and repack the binary (with
ad-hoc code signing on Apple Silicon).

### How It Enables Gated Features

tweakcc does NOT manipulate Statsig cache files. Instead, it patches the JavaScript
that evaluates gates — replacing conditional checks like `if (gate('tengu_session_memory'))`
with `if (true)`. This approach:

- **Survives Statsig server refreshes** (the check is bypassed entirely)
- **Breaks on Claude Code updates** (binary changes, must reapply)
- Explicitly lists "Session memory unlock" as a feature (v4.0.0+)

### Installation and Usage

```bash
# Run without installing (interactive TUI)
npx tweakcc

# Also available via Homebrew
brew install tweakcc

# Reapply after Claude Code update
npx tweakcc --apply

# Ad-hoc patching (programmatic)
npx tweakcc adhoc-patch --string '"old"' '"new"'
npx tweakcc adhoc-patch --regex 'pattern' 'replacement'
npx tweakcc adhoc-patch --script 'return js.replace(/old/g, "new")'
```

Ad-hoc scripts run with Node.js `--experimental-permission` for sandboxing.

### Capabilities

| Category | Examples |
|----------|---------|
| Feature Flags | Session memory, multi-agent teams, scratch pad |
| System Prompts | Replace portions of Claude's system prompt |
| Themes | Custom colors, spinner animations |
| Toolsets | Control which built-in tools Claude can see |
| Model Routing | Configure model per subagent type |
| UI | Input box styling, expanded thinking by default |

### Maintenance

- **Latest release:** v4.0.10 (Feb 28, 2026)
- **Verified compatibility:** Claude Code 2.1.62
- **License:** MIT
- **Stars:** 1,200+, 87 forks
- **Open issues:** 31 (mostly feature requests)

## Legal and Risk Assessment

### Anthropic's Consumer ToS

Section 3 (Use Restrictions):
> "To decompile, reverse engineer, disassemble, or otherwise reduce our Services
> to human-readable form, except when these restrictions are prohibited by
> applicable law"

### Enforcement History

| Action | Outcome |
|--------|---------|
| Deobfuscated source published on GitHub | **DMCA takedown** (Apr 2025) |
| Third-party harnesses spoofing Claude Code | **Technical blocks** (Jan 2026) |
| tweakcc (1,200+ stars, 7+ months public) | **No action** |
| Binary analysis tools (claude-code-reverse) | **No action** |
| Modifying local config files | **No known enforcement** |

tweakcc was announced on the claude-code issues tracker (#4429, Jul 2025).
Auto-closed after 60 days with zero Anthropic engagement.

### Technical Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Flag overwritten by server sync | N/A for tweakcc (patches code, not cache) | Reapply after updates |
| Enabling unstable flags causes crashes | Medium (#19869: `tengu_scratch` froze CLI) | Only enable well-tested flags |
| `.claude.json` corruption from concurrent writes | Medium (#29003) | Don't edit while Claude runs |
| Telemetry reveals flag manipulation | Low | Events are diagnostic, not enforcement |
| Binary patch breaks on update | Certain | `npx tweakcc --apply` after each update |

## Alternative Approaches

| Approach | Enables Producer? | Survives Restart? | Risk |
|----------|------------------|-------------------|------|
| **tweakcc** | Yes (patches gate check) | Yes (until CC update) | Low-moderate |
| **Edit `~/.claude.json`** | Theoretically | No (overwritten on startup) | Low |
| **Edit Statsig cache** | Theoretically | No (overwritten + checksum issues) | Medium |
| **Freeze traffic + edit cache** | Yes | Yes (but freezes ALL flags) | High |
| **`ENABLE_CLAUDE_CODE_SM_COMPACT` env var** | No (consumer only) | Yes | None |
| **Wait for rollout** | Eventually | Yes | None |

## Inspecting Your Flags

### Quick Check (GrowthBook, plain text)

```bash
python3 -c "
import json
d = json.load(open('$HOME/.claude.json'))
flags = d.get('cachedGrowthBookFeatures', {})
for k in sorted(flags):
    if 'session' in k or 'compact' in k or 'memory' in k:
        print(f'  {k}: {flags[k]}')
"
```

### Full Statsig Audit

```python
def djb2(s: str) -> str:
    h = 0
    for c in s:
        h = ((h << 5) - h + ord(c)) & 0xFFFFFFFF
    return str(h)

# Compute hash for any flag name:
print(djb2("tengu_session_memory"))  # → 3695724478

# Then look it up in the cache:
import json
data = json.load(open("~/.claude/statsig/statsig.cached.evaluations.<hash>"))
inner = json.loads(data["data"])
gates = inner["feature_gates"]
print(gates.get("3695724478"))
```

## References

- [tweakcc](https://github.com/Piebald-AI/tweakcc) — MIT, v4.0.10
- [Statsig JS Client Hashing.ts](https://github.com/statsig-io/js-client-monorepo/blob/main/packages/client-core/src/Hashing.ts)
- [Anthropic Consumer Terms](https://www.anthropic.com/legal/consumer-terms)
- [DMCA Takedown (TechCrunch)](https://techcrunch.com/2025/04/25/anthropic-sent-a-takedown-notice-to-a-dev-trying-to-reverse-engineer-its-coding-tool/)
- [tweakcc Issue #4429](https://github.com/anthropics/claude-code/issues/4429)
- [Stale Cache Issue #28777](https://github.com/anthropics/claude-code/issues/28777)
- [tengu_scratch Crash #19869](https://github.com/anthropics/claude-code/issues/19869)
- [.claude.json Corruption #29003](https://github.com/anthropics/claude-code/issues/29003)
