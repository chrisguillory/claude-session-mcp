# Potential Improvements & Measurements

Ideas for extending claude-session-mcp and the broader Claude Code tooling
ecosystem. Each item includes what to measure and how to know if it's working.

---

## 1. Context Management (Compaction & Eviction)

The highest-value area. Better compaction directly improves every long session.

### 1.1 Selective Context Eviction

**The dream**: instead of wholesale compaction (discard everything, summarize),
score individual message blocks by current relevance and evict only the
lowest-value ones.

**What it would take**:
- Token counting per message block (approximate, using `cl100k_base` or the
  actual API token counter endpoint)
- Relevance scoring: recency, reference count (is this message cited later?),
  content type (completed tool results vs. active work), size-to-information ratio
- Conversation threading awareness: can't evict a `tool_use` without its
  `tool_result`, can't orphan a `parentUuid` chain
- Would need to operate at the JSONL level before Claude Code sends to the API

**How to measure**:
- **Token savings**: how many tokens freed per eviction pass
- **Information preservation**: key entities (file paths, decisions, function names)
  that survive vs. are lost
- **Continuity**: can the model reference pre-eviction context without hallucinating?

**Difficulty**: High. This is essentially a custom compaction engine.
**Dependency**: Deep understanding of how Claude Code constructs the API payload
from JSONL records. The `compact_boundary` mechanism shows the seams.

### 1.2 Pre-Compaction Bloat Analysis

Before hitting 200K, identify what's consuming the most tokens for the least
value. Produce a report:

```
Token Budget Analysis (current session):
  System prompt + tools:     19,200 (10%)
  Memory files:               3,400 (2%)
  Conversation messages:     42,000 (22%)
  Tool results (active):     28,000 (15%)
  Tool results (stale):      61,000 (32%)  ← biggest opportunity
  Thinking blocks:                0 (not in context)
  Overhead:                  36,400 (19%)
```

**How to measure**:
- Parse JSONL, classify each record, estimate tokens
- Identify "stale" tool results (from tasks marked complete, files no longer
  being edited, search results already acted upon)
- Track how this distribution changes over a session's lifetime

**Difficulty**: Medium. Token estimation is approximate but actionable.
**Dependency**: Session record parsing (already have this via Pydantic models).

### 1.3 Session Memory Quality Comparison

Compare three compaction approaches on the same session data:

| Approach | Input | Output | Measurement |
|----------|-------|--------|-------------|
| Traditional compact | Full context | Model-generated summary | Quality, latency, tokens |
| SM compact | Pre-built summary.md | summary + verbatim recent | Quality, latency, tokens |
| Handoff document | User-requested artifact | Structured handoff | Quality, completeness |

**How to measure**:
- For sessions that have both session-memory summaries AND compact boundaries,
  compare the two summary texts
- Key entity extraction: parse both summaries for file paths, function names,
  decisions, pending work items
- Coverage ratio: what percentage of entities from the full session appear in
  each summary type?
- Char/token efficiency: information density per token

**Difficulty**: Medium. We have the data (6 session-memory files, many compact
boundaries). Need to build the comparison tooling.

### 1.4 Compaction Timing Optimization

Track when compaction happens relative to session activity:

- How many tokens before auto-compact triggers?
- How much context is lost at each compaction?
- Would earlier, smaller compactions preserve more information?
- Does the `minimumMessageTokensToInit: 140000` threshold for session memory
  start too late?

**How to measure**:
- Parse `compact_boundary` records: `compactMetadata.preTokens` shows the
  pre-compact token count
- Correlate with `usage` fields on surrounding assistant messages
- Build a timeline: token usage curve with compact events marked

**Difficulty**: Low. All data is already in JSONL files.

---

## 2. Session Intelligence

Making session data more accessible and useful.

### 2.1 Session Memory as Search Index

Session-memory summaries are structured, high-signal, and small (~5-40KB vs.
multi-MB JSONL transcripts). Use them as a first-pass search layer.

**How it would work**:
1. Search session-memory summaries first (fast, low noise)
2. If a match is found, optionally drill into the full JSONL for details
3. The 10-section structure enables section-targeted search (e.g., search only
   "Errors & Corrections" for debugging patterns)

**How to measure**:
- Precision: what percentage of summary search hits are actually relevant?
- Recall: what percentage of relevant sessions are found via summary search
  vs. missed (requiring full JSONL search)?
- Speed: summary search time vs. full JSONL search time

**Difficulty**: Low-Medium. Requires session memory to be enabled first.

### 2.2 Session Analytics

Aggregate statistics across sessions for a project:

- Token consumption per session (total, by type)
- Compaction frequency and quality
- Tool call distribution (which tools used most, error rates)
- Model distribution (which models used, when)
- Session duration and message counts
- Cost estimation (based on token usage and model pricing)

**How to measure**: Define metrics, parse JSONL files, produce reports.
Could be an MCP tool (`session_analytics`) or a CLI command.

**Difficulty**: Medium. Schema models make parsing reliable. The volume of data
(927K+ records across all sessions) requires efficient processing.

### 2.3 Cross-Session Topic Detection

Sessions about the same topic often span multiple session IDs (compaction
restarts, worktree switches, resume failures). Detect clusters:

- Extract key entities from each session (file paths, package names, error
  messages, custom titles)
- Group sessions by topic overlap
- Build a session graph showing continuation relationships

**How to measure**:
- Cluster coherence: do grouped sessions actually relate to the same work?
- Coverage: are all related sessions found?

**Difficulty**: High. NLP/embedding-based clustering, or simpler heuristics
based on `cwd`, slug, and custom title overlap.

---

## 3. Feature Flag Intelligence

Turning our binary analysis capabilities into ongoing monitoring.

### 3.1 Flag Rollout Tracking

Snapshot `cachedGrowthBookFeatures` (plain-text flag names in `~/.claude.json`)
at each session start. Store as time-series data.

**What this enables**:
- Detect when flags change (new feature rollouts, rollbacks)
- Correlate flag changes with Claude Code version updates
- Build a historical record of which flags were active when

**Implementation**: A SessionStart hook that reads `~/.claude.json` and appends
to a tracking file. Minimal code, high value.

**How to measure**:
- Number of flag changes detected per week
- Correlation between flag changes and observable behavior changes
- Lead time: how early do we detect a rollout before it's documented?

**Difficulty**: Low. Hook + JSON parsing + append to log.

### 3.2 Binary Diff on Version Updates

When Claude Code updates (detectable via version change in hooks), automatically:

1. Extract `tengu_*` strings from the new binary
2. Compare with the previous version's strings
3. Report new flags, removed flags, and changed constants

**How to measure**:
- New flags discovered per version
- Schema-relevant changes identified (new tool names, record types)
- Time savings vs. manual binary analysis

**Difficulty**: Medium. Needs binary access and string extraction automation.

### 3.3 Complete Gate Name Registry

We mapped 13 of 66 Dec gates and 13 of 41 current gates using `tengu_*` names.
The unmapped 53 gates in the Dec cache use different naming conventions.

**To resolve**:
- Extract ALL string literals from the binary (not just `tengu_*`)
- Hash each one with DJB2
- Match against the gate IDs in the Statsig cache
- This would give us a complete mapping of every feature flag

**How to measure**: Percentage of gates mapped (currently ~30%).

**Difficulty**: Medium. The binary has millions of strings; need smart filtering.

---

## 4. Operational Improvements

Enhancing existing session operations.

### 4.1 sessions-index.json Management

Claude Code maintains a per-project `sessions-index.json` cache for fast session
listing. After move/delete operations, this cache becomes stale.

**Options**:
- Remove the moved/deleted session's entry from the source project's index
- Add the moved session's entry to the target project's index
- Or just delete the index file (Claude Code regenerates it)

**How to measure**:
- Does `claude --resume` find the moved session without issues?
- Does the session list in Claude Code show correct data after operations?

**Difficulty**: Low. JSON file manipulation.

### 4.2 Tool Result Format Gap

Binary analysis found `.json` tool result files (image content blocks) alongside
the expected `.txt` files. Current code only handles `.txt`.

**Impact**: Tool results with images may be silently dropped during
archive/clone/restore/move/delete operations.

**How to measure**:
- Count `.json` vs `.txt` files in tool-results directories
- Verify all tool results survive round-trip operations

**Difficulty**: Low. Extend glob patterns and expected extensions.

### 4.3 Archive Compression Benchmarks

Archives support JSON and JSON+zstandard. Measure:

- Compression ratio by session size
- Compression/decompression speed
- Impact on Gist upload size (GitHub has size limits)

**Difficulty**: Low. Benchmark script against existing archives.

### 4.4 Path Translation Verification

After a move or cross-machine restore, verify that path translation worked:

- Extract all `PathField` values from the target JSONL
- Confirm none still reference the source project path
- Confirm all reference the target project path (or paths outside the project,
  which should be unchanged)

**How to measure**: Automated post-operation verification.

**Difficulty**: Low. Use the PathMarker introspection system.

---

## 5. The Controller Layer

A missing piece in the Claude Code tooling ecosystem.

### 5.1 Claude Code Configuration MCP

An MCP server (or extension of this one) that provides programmatic control over
Claude Code's runtime behavior:

| Tool | Purpose |
|------|---------|
| `get_feature_flags` | List all flags with current values |
| `get_settings` | Read settings.json |
| `update_settings` | Modify settings.json safely |
| `get_debug_log` | Read debug log for a session |
| `get_version` | Current Claude Code version |
| `configure_session_memory` | Create/edit custom templates |

**How to measure**:
- Time to diagnose configuration issues (with vs. without the tool)
- Number of manual file edits replaced by tool calls

**Difficulty**: Medium. Settings management needs careful locking (concurrent
write issues documented in #29003).

### 5.2 Feature Flag Toggle Tool

Integrate with tweakcc or implement independent flag manipulation:

- **Via tweakcc**: Shell out to `npx tweakcc adhoc-patch` for specific patches
- **Via cache manipulation**: Edit `cachedGrowthBookFeatures` + disable traffic
- **Via env vars**: Set `ENABLE_*` variables in settings.json env

**Considerations**:
- ToS implications (document but let user decide)
- Fragility (patches break on updates, cache edits overwritten)
- Scope (which flags are safe to toggle?)

**Difficulty**: Low (env var approach) to Medium (tweakcc integration).

### 5.3 Hook-Based Telemetry

Use claude-workspace hooks to capture operational data:

- SessionStart: snapshot flags, version, project, memory files
- SessionEnd: capture duration, token usage, compaction count
- Pre-prompt: capture context window state (via `/context` scraping)

Build a local analytics database from hook output.

**How to measure**:
- Insights per session (what did we learn that we wouldn't have otherwise?)
- Anomaly detection: flag changes, unusual token patterns, new record types

**Difficulty**: Medium. Hook infrastructure exists; analysis pipeline is new work.

---

## 6. Measurement Framework

How to measure any of the above systematically.

### 6.1 Token Estimation Library

A lightweight module for estimating token counts from session records:

```python
def estimate_tokens(record: SessionRecord) -> int:
    """Approximate token count for a session record."""
    ...
```

Uses `cl100k_base` tokenizer (via `tiktoken`) or the simpler chars/4 heuristic.
Critical for compaction analysis, bloat detection, and cost estimation.

**Validation**: Compare estimates against actual `usage` fields on assistant
records (which report real token counts from the API).

### 6.2 Information Preservation Score

A metric for compaction quality:

1. Extract "key entities" from pre-compact context (file paths, function names,
   error messages, user decisions, pending tasks)
2. Check how many appear in the post-compact summary
3. Score = entities_preserved / entities_total

Apply to traditional compact, SM compact, and handoff documents to compare.

### 6.3 Session Operation Fidelity Score

After any operation (clone, restore, move), verify:

- [ ] All JSONL records parseable by Pydantic models
- [ ] All PathField values correctly translated (no source paths remain)
- [ ] All auxiliary artifacts present (tool results, session memory, plans, etc.)
- [ ] Session ID preserved (for move) or correctly remapped (for clone)
- [ ] Session resumable from target location

Score = checks_passed / checks_total.

### 6.4 Regression Detection

Run `validate_models.py` before and after Claude Code updates. Track:

- New validation errors (schema drift)
- New record types or subtypes
- New tool input/output schemas
- Version delta between `CLAUDE_CODE_MAX_VERSION` and installed version

Automate via a post-update hook or periodic cron job.