# Thinking & Effort in Claude Code

Two independent API systems control reasoning depth. Understanding their interaction is essential for tuning Claude Code's behavior.

| System       | API Parameter          | Controls                                      | Default (Opus 4.6)      |
|--------------|------------------------|-----------------------------------------------|--------------------------|
| Thinking     | `thinking`             | Whether and how Claude reasons before response | Adaptive (model decides) |
| Effort       | `output_config.effort` | Soft guidance on overall reasoning depth       | Medium                   |

Both are sent in every API request. Setting one does not affect the other.

Verified against Claude Code **v2.1.80** via binary analysis, API documentation, and GitHub issue research. Last updated **March 2026**.

## Current Defaults

As of v2.1.68, Opus 4.6 defaults to **medium** effort with **adaptive** thinking. This reduction from the prior default of high effort prompted the restoration of the `ultrathink` keyword as a per-turn boost mechanism (see [Ultrathink](#ultrathink)).

| Model                                | Thinking Mode | Default Effort | Max Effort |
|--------------------------------------|---------------|----------------|------------|
| Opus 4.6                             | Adaptive      | Medium         | Max        |
| Sonnet 4.6                           | Adaptive      | Medium         | High       |
| Opus 4.5 / Sonnet 4.x / Haiku 4.x   | Fixed budget  | N/A            | N/A        |

## Thinking Modes

### Adaptive (Opus 4.6, Sonnet 4.6 default)

```json
{"thinking": {"type": "adaptive"}}
```

The model decides per-turn whether and how much to think. The `effort` parameter provides soft guidance. Adaptive mode automatically enables **interleaved thinking** — the model reasons between tool calls, not just at the start.

`budget_tokens` is deprecated on 4.6 models when using adaptive mode. Anthropic: it "will be removed in a future model release."

### Fixed Budget

```json
{"thinking": {"type": "enabled", "budget_tokens": 127999}}
```

Always thinks, up to the specified token ceiling. Deterministic per-turn cost. Default on pre-4.6 models. Force on 4.6 models with `CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1`.

Interleaved thinking availability with fixed budget:

| Model                                | Interleaved Thinking Available?                        |
|--------------------------------------|--------------------------------------------------------|
| Opus 4.6                             | **No.** Use adaptive mode for interleaved thinking.    |
| Sonnet 4.6                           | Yes, via `interleaved-thinking-2025-05-14` beta header |
| Pre-4.6 models                       | No                                                     |

### Disabled

```json
{"thinking": {"type": "disabled"}}
```

No internal reasoning. Lowest latency. Set via `CLAUDE_CODE_DISABLE_THINKING=1`.

## Effort Levels

Effort is a separate API parameter (`output_config.effort`) providing soft guidance on reasoning depth. It works with or without thinking enabled.

| Level    | Behavior                                       | Model Support                                        |
|----------|-------------------------------------------------|------------------------------------------------------|
| `low`    | May skip thinking for simple queries            | All effort-capable models                            |
| `medium` | Moderate thinking (current Opus 4.6 default)    | All effort-capable models                            |
| `high`   | Almost always thinks deeply                     | All effort-capable models                            |
| `max`    | Always thinks, no constraints on depth          | **Opus 4.6 only** (downgrades to `high` on others)  |

The API natively accepts `low`, `high`, `max`. Claude Code adds `medium` at the application layer.

### The "max" Persistence Problem

`"max"` cannot be saved to `settings.json`. The Zod schema only accepts `low`, `medium`, `high`. The settings serializer (`vK_`) returns `undefined` for `"max"`, silently dropping it. Binary-confirmed.

| Method                           | Accepts `max`?        | Persists?     |
|----------------------------------|-----------------------|---------------|
| `settings.json` `effortLevel`   | No (silently dropped) | N/A           |
| `/effort max`                    | Yes                   | Session only  |
| `CLAUDE_CODE_EFFORT_LEVEL=max`  | Yes (in code)         | N/A (env var) |
| `--effort max`                   | Rejected for Claude.ai subscribers | N/A |

15+ open GitHub issues about this. Zero Anthropic engineer responses as of March 2026.

## Ultrathink

**Restored in v2.1.68 (March 2026).** Previously cosmetic-only (January 2026), ultrathink was made functional again when the Opus 4.6 default effort was lowered to medium. Gated by `tengu_turtle_carbon` Statsig flag (default: `true`).

### Mechanism

1. Regex `/\bultrathink\b/i` matches in your prompt (word-boundary, case-insensitive)
2. Injects a system message: *"The user has requested reasoning effort level: high. Apply this to the current turn."*
3. Shows rainbow animation on the keyword text
4. Shows notification: "Effort set to high for this turn" (5 seconds)
5. Fires `tengu_ultrathink` telemetry event

The keyword is **not stripped** from the prompt — it is sent verbatim to the model alongside the system message.

### Limitations

- Sets effort to `high`, **not** `max` ([#34077](https://github.com/anthropics/claude-code/issues/34077))
- Does **not** directly set the API `effort` parameter — works via system message injection
- Does **not** modify `budget_tokens` or thinking config
- Does **not** persist beyond the current turn
- **Redundant** if your `effortLevel` is already `high`

### Other Keywords

`think`, `think hard`, and `think harder` are fully deprecated. They are treated as normal prompt text with no special handling.

### ThinkingMetadata in Session Files

Session files record keyword detection on user records. This is local-only metadata — not transmitted to the API:

```json
{
  "thinkingMetadata": {
    "level": "high",
    "disabled": false,
    "triggers": [{"start": 0, "end": 10, "text": "ultrathink"}]
  }
}
```

## Configuration Reference

### Environment Variables

| Variable                                  | Effect                                     | Default              |
|-------------------------------------------|--------------------------------------------|----------------------|
| `MAX_THINKING_TOKENS`                     | Fixed: sets `budget_tokens`. Adaptive: boolean (>0 = on). | 127,999 (Opus 4.6) |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS`           | Total output ceiling (thinking + response) | 64,000 (Opus 4.6)   |
| `CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING`   | `1` to force fixed budget on 4.6 models    | unset (adaptive)     |
| `CLAUDE_CODE_DISABLE_THINKING`            | `1` to disable thinking entirely           | unset                |
| `CLAUDE_CODE_EFFORT_LEVEL`                | Override effort level                      | unset                |
| `CLAUDE_CODE_ALWAYS_ENABLE_EFFORT`        | `1` to send effort for all models          | unset                |

`CLAUDE_CODE_EFFORT_LEVEL` is **not in the settings.json env allowlist**. It must be set as a real shell environment variable, not in the `env` block of settings.json.

### Settings

Top-level key in `settings.json`. Accepts `low`, `medium`, `high` only:

```json
{"effortLevel": "high"}
```

### Slash Commands

| Command         | Values                      | Persistence            |
|-----------------|-----------------------------|------------------------|
| `/effort low`   | `low`, `medium`, `high`     | Saved to settings.json |
| `/effort max`   | `max`                       | Session only           |
| `/effort auto`  | `auto` (alias: `unset`)     | Removes from settings  |

### Priority Order

```
CLAUDE_CODE_EFFORT_LEVEL env var    (highest — overrides everything)
  > /effort command                 (session-level)
  > effortLevel in settings.json   (persistent)
  > ultrathink system message      (per-turn, soft guidance via prompt)
  > Cp_() default                  ("medium" when tengu_turtle_carbon enabled)
```

## Token Limits

### Per-model limits (binary analysis, v2.1.80)

| Model                                | Default `max_tokens` | Upper Limit | Default `budget_tokens` |
|--------------------------------------|----------------------|-------------|-------------------------|
| Opus 4.6                             |               64,000 |     128,000 |                 127,999 |
| Sonnet 4.6                           |               32,000 |     128,000 |                 127,999 |
| Opus 4.5 / Sonnet 4.x / Haiku 4.x   |               32,000 |      64,000 |                  63,999 |
| Opus 4.1 / Opus 4                    |               32,000 |      32,000 |                  31,999 |
| Claude 3.7 Sonnet                    |               32,000 |      64,000 |                  63,999 |

### The -1 Offset

The API requires `budget_tokens < max_tokens` (strictly less than). The binary enforces this with `Math.min(max_tokens - 1, budget_tokens)` in three code paths (main query builder, retry function, side queries). The default budget function `Ch6(model)` builds this directly: `Io(model).upperLimit - 1`.

All limits are decimal (128,000 not 131,072). Values above the upper limit are silently capped.

## Configuration Recipes

### Maximum thinking — adaptive (recommended for Opus 4.6)

```json
{
  "effortLevel": "high",
  "env": {
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128000",
    "MAX_THINKING_TOKENS": "127999"
  }
}
```

Run `/effort max` at session start for unconstrained depth. `MAX_THINKING_TOKENS` acts as boolean only (>0 = enabled). Interleaved thinking enabled. `high` persists across sessions; `max` must be re-applied per session.

### Maximum thinking — fixed budget (deterministic)

```json
{
  "env": {
    "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING": "1",
    "MAX_THINKING_TOKENS": "127999",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128000"
  }
}
```

Guaranteed thinking on every turn. Hard 127,999 token ceiling. Loses interleaved thinking on Opus 4.6. Some users report better CLAUDE.md instruction-following.

### Balanced (default + high output ceiling)

```json
{
  "effortLevel": "high",
  "env": {
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128000"
  }
}
```

Adaptive thinking at high effort. Interleaved thinking enabled. Model almost always thinks deeply.

### Minimal (fast responses)

```json
{
  "effortLevel": "low"
}
```

Model may skip thinking for simple queries. Lower cost, faster responses.

## Adaptive vs Fixed Budget

| Consideration                    | Adaptive (default)                    | Fixed Budget                          |
|----------------------------------|---------------------------------------|---------------------------------------|
| Thinking depth                   | Model decides per-turn                | Always allocates up to `budget_tokens` |
| Interleaved thinking (Opus 4.6)  | Automatic                             | **Not available**                     |
| Interleaved thinking (Sonnet 4.6)| Automatic                             | Available via beta header             |
| Per-turn cost predictability     | Variable                              | Deterministic ceiling                 |
| Control mechanism                | Soft (`effort` parameter)             | Hard (`budget_tokens` cap)            |
| CLAUDE.md instruction following  | Reports of degradation (#23936)       | Users report better compliance        |
| Circular reasoning               | Reports of repetitive self-doubt      | Not reported                          |
| Deprecation status               | Current / recommended                 | Deprecated on 4.6 models              |

## Known Issues (March 2026)

| Issue   | Description                                                                |
|---------|----------------------------------------------------------------------------|
| [#35904](https://github.com/anthropics/claude-code/issues/35904) | `effortLevel` settings.json does not accept `"max"` — silently drops |
| [#34837](https://github.com/anthropics/claude-code/issues/34837) | Root cause: serializer only guards `low/medium/high`               |
| [#34077](https://github.com/anthropics/claude-code/issues/34077) | `ultrathink` sets effort to `high` instead of `max`                |
| [#23936](https://github.com/anthropics/claude-code/issues/23936) | Adaptive thinking + high effort deprioritizes CLAUDE.md rules      |
| [#23553](https://github.com/anthropics/claude-code/issues/23553) | Adaptive thinking produces circular reasoning                      |
| [#34633](https://github.com/anthropics/claude-code/issues/34633) | `/effort` and `/model` read different state stores                 |
| [#23606](https://github.com/anthropics/claude-code/issues/23606) | Feature request: expose `max` effort in UI                         |
| [#27429](https://github.com/anthropics/claude-code/issues/27429) | `MAX_THINKING_TOKENS` is global — crashes subagents                |
| [#33506](https://github.com/anthropics/claude-code/issues/33506) | Custom endpoints fail on `thinking.type: "adaptive"`               |

Zero Anthropic engineer responses found on any adaptive thinking or effort-level issue.

## Binary Analysis Reference

Key functions in the Claude Code binary (v2.1.80). Minified names change between versions — search by behavior, not name.

| Function | Purpose                                                                    |
|----------|----------------------------------------------------------------------------|
| `Io()`   | Returns `{default, upperLimit}` output token limits per model              |
| `mt()`   | Validates env var values with capping at `upperLimit`                      |
| `xU_()`  | Resolves effective `max_tokens` from env + model defaults                  |
| `Ch6()`  | Default thinking budget: `Io(model).upperLimit - 1`                        |
| `uYT()`  | Returns `true` for adaptive-capable models (opus-4-6, sonnet-4-6)          |
| `$J_()`  | Reads `MAX_THINKING_TOKENS`; returns `parseInt(value) > 0`                 |
| `OJ_()`  | Resolves effort; downgrades `max` to `high` for non-Opus-4.6              |
| `vK_()`  | Settings serializer — persists `low/medium/high` only                      |
| `yK_()`  | Reads `CLAUDE_CODE_EFFORT_LEVEL` env var                                   |
| `q_$()`  | Populates `output_config.effort` in API request                            |
| `Cp_()`  | Default effort per model (`medium` when `tengu_turtle_carbon` enabled)     |
| `Vp_()`  | Returns `true` for Opus 4.6 only (guards `max` effort)                     |
| `YV6()`  | Ultrathink regex detection: `/\bultrathink\b/i`                            |
| `eSR()`  | Creates ultrathink attachment `{type: "ultrathink_effort", level: "high"}` |
| `ZF()`   | Feature gate: `tengu_turtle_carbon` (default `true`)                       |

### Typical API request (Opus 4.6, adaptive, high effort)

```json
{
  "model":         "claude-opus-4-6-20260205",
  "max_tokens":    128000,
  "thinking":      {"type": "adaptive"},
  "output_config": {"effort": "high"},
  "messages":      [...]
}
```

### Typical API request (Opus 4.6, fixed budget)

```json
{
  "model":      "claude-opus-4-6-20260205",
  "max_tokens": 128000,
  "thinking":   {"type": "enabled", "budget_tokens": 127999},
  "messages":   [...]
}
```

## History

| Date        | Event                                                                                      |
|-------------|--------------------------------------------------------------------------------------------|
| ~2024       | Ultrathink introduced with tiered keywords: `think` (~4K), `think hard` (~10K), `think harder` (~20K), `ultrathink` (31,999) |
| Oct 2025    | Intermediate keywords deprecated (false positive issues). Only `ultrathink` retained.      |
| Jan 2026    | Thinking on by default at max budget. Boris Cherny: "ULTRATHINK doesn't really do anything anymore." Keyword cosmetic only. |
| Feb 5, 2026 | Opus 4.6 and Sonnet 4.6 released. Adaptive thinking default. Effort parameter GA. Fixed `budget_tokens` deprecated on 4.6. |
| v2.1.68     | **Opus 4.6 default effort lowered to medium.** Ultrathink restored as functional (sets effort to high per turn). |
| v2.1.72     | `max` removed from persistent settings schema. `/effort` command added.                    |
| v2.1.77     | Output token upper bound raised from 64K to 128K for Opus 4.6 and Sonnet 4.6.             |
| v2.1.80     | Current version. Effort frontmatter support for skills.                                    |