# claude-session-mcp

Archive and restore Claude Code sessions across machines with full conversation history.

## ⚠️ Critical: Prevent Automatic Session Deletion

**Claude Code deletes session `.jsonl` files after 30 days by default.** Add this to `~/.claude/settings.json` **before using this tool**:

```json
{
  "cleanupPeriodDays": 99999
}
```

<details>
<summary><b>Why this matters and how it works</b></summary>

### The Problem

Claude Code silently deletes session transcripts from `~/.claude/projects/` after 30 days (based on last activity), running cleanup on every launch. This destroys the `.jsonl` files this project depends on, while leaving orphaned metadata in `~/.claude/todos/` and `~/.claude/debug/` - explaining "Session not found" errors even when `rg <session-id>` finds matches.

### Settings File Location

- **macOS/Linux**: `~/.claude/settings.json`
- **Windows**: `%USERPROFILE%\.claude\settings.json`

Create the file if it doesn't exist.

### How Cleanup Works

- **Default retention**: 30 days from last activity ([official docs](https://docs.anthropic.com/en/docs/claude-code/settings))
- **Cleanup timing**: Automatic on every Claude Code launch
- **No warning**: Files deleted silently without notification
- **Not recoverable**: Once deleted, transcripts cannot be restored

**⚠️ Warning**: Setting `cleanupPeriodDays: 0` immediately wipes ALL sessions.

### References

- [GitHub Issue #4172](https://github.com/anthropics/claude-code/issues/4172) - Auto-deletion disable request (resolved)
- [GitHub Issue #2543](https://github.com/anthropics/claude-code/issues/2543) - Documentation clarification
- [Official Settings Docs](https://docs.anthropic.com/en/docs/claude-code/settings) | [Data Usage Docs](https://docs.anthropic.com/en/docs/claude-code/data-usage)

</details>

## Installation

```bash
# Run without installation
uvx --from git+https://github.com/chrisguillory/claude-session-mcp claude-session --help

# Or install locally
git clone https://github.com/chrisguillory/claude-session-mcp
cd claude-session-mcp
uv sync
```

## Quick Start

```bash
# Clone a session directly and launch Claude
claude-session clone <session-id> --launch

# Or archive to local file
claude-session archive <session-id> output.json

# Archive to GitHub Gist (requires GITHUB_TOKEN)
export GITHUB_TOKEN=$(gh auth token)
claude-session archive <session-id> gist://

# Restore from file (with optional auto-launch)
claude-session restore archive.json --launch

# Restore from Gist
claude-session restore gist://<gist-id> --launch
```

## Features

- **Local storage**: Save sessions as JSON or compressed (zstd)
- **GitHub Gist**: Private/public Gist storage (100MB limit)
- **Path translation**: Automatically translates file paths when restoring to different directories
- **Agent sessions**: Captures main session + all agent sub-sessions
- **UUIDv7 session IDs**: Restored sessions use time-ordered UUIDv7 (vs Claude's random UUIDv4)

## Clone vs Native Fork

Claude Code 2.0.73+ supports `--fork-session`. Here's how it compares to our `clone` command:

| | Claude Native Fork | MCP Clone |
|---|---|---|
| **Command** | `claude --resume <id> --fork-session` | `claude-session clone <id>` |
| **Mechanism** | Compact (summary) | Full copy |
| **Session size** | ~34 KB | ~1.1 MB |
| **Context** | Summary blob | Complete history |
| **Fidelity** | Lossy | Lossless |
| **Agent history** | Not preserved | Preserved |
| **Cross-machine** | No | Yes |
| **Gist support** | No | Yes |
| **Path translation** | No | Yes |

**Key difference**: Native fork is `/compact` under the hood - it copies a summary + recent messages, not actual conversation history. Our clone preserves everything: all tool calls, thinking blocks, agent sub-sessions, and intermediate steps.

**Why compaction loses context**: Auto-compaction triggers at ~75% context utilization (not 90%+). Claude Code reserves 25% headroom for reasoning quality. When forking, you get the summary blob, not the rich history that led to decisions.

## Commands

### Clone

Direct session-to-session cloning without intermediate archive file:

```bash
claude-session clone <session-id>

# Arguments:
#   session-id     Session ID to clone (full UUID or prefix)

# Options:
#   --project, -p      Target project directory (default: current)
#   --no-translate     Don't translate file paths
#   --launch, -l       Launch Claude Code after clone
#   --verbose, -v      Verbose output
```

### Archive

```bash
claude-session archive <session-id> <output>

# Arguments:
#   session-id    Session ID to archive (find in ~/.claude/projects/)
#   output        File path or gist:// or gist://<gist-id>

# Options:
#   --format, -f           json or zst (auto-detected from extension)
#   --gist-token           GitHub token (or use GITHUB_TOKEN env)
#   --gist-visibility      public or secret (default: secret)
#   --gist-description     Gist description
#   --verbose, -v          Verbose output
```

### Restore

```bash
claude-session restore <archive>

# Arguments:
#   archive        File path or gist://<gist-id>

# Options:
#   --project, -p      Target project directory (default: current)
#   --no-translate     Don't translate file paths
#   --launch, -l       Launch Claude Code after restore
#   --gist-token       GitHub token for private gists
#   --verbose, -v      Verbose output
```

## MCP Server

The MCP server exposes archive/restore as Claude Code tools:

```json
{
  "mcpServers": {
    "claude-session": {
      "command": "uv",
      "args": [
        "run",
        "mcp-server.py"
      ],
      "cwd": "/path/to/claude-session-mcp"
    }
  }
}
```

**Tools:**

- `clone_session` - Clone a session directly (no archive file needed)
- `save_current_session` - Archive current session (local storage only)
- `restore_session` - Restore archived session (local files only)

## Technical Details

### Session Storage

Sessions are stored as JSONL files at `~/.claude/projects/<encoded-path>/<session-id>.jsonl`.

**Path Encoding**: Claude Code encodes working directory paths for filesystem safety:
- `/` → `-`
- `.` → `-`
- ` ` (space) → `-`
- `~` (tilde) → `-`

Example: `/Users/chris/My Project.app` → `-Users-chris-My-Project-app`

**Session IDs**: Claude Code uses UUIDv4 (random). This tool generates UUIDv7 (time-ordered) for cloned/restored sessions:

| Type | Source | Pattern | Example |
|------|--------|---------|---------|
| UUIDv4 | Claude Code native | `xxxxxxxx-xxxx-4xxx-...` | `a1b2c3d4-e5f6-4a7b-...` |
| UUIDv7 | Cloned/restored | `01xxxxxx-xxxx-7xxx-...` | `01934a2b-c3d4-7e5f-...` |

The version digit (4 or 7) is at position 15 (first digit of third group).

### Pydantic Models

914 lines, 56 classes covering 10 record types (user, assistant, system, local commands, compact boundaries, API errors, summaries, file history snapshots, queue operations). Validated on 78K+ records (100% valid).

### Compatibility

Claude Code 2.0.35 - 2.0.64

### Claude Code Architecture Reference

Understanding what gets captured in session files.

**Built-in Agents**:

| Agent | Model | Purpose | Creates Sidechain |
|-------|-------|---------|-------------------|
| Plan | Opus/Sonnet | Software architect (read-only) | Yes |
| Explore | Haiku | File search specialist | Yes |
| General-purpose | Sonnet | Complex multi-step tasks | Yes |

Agent invocations create separate `agent-{agentId}.jsonl` files, referenced via `isSidechain: true` in the main session. Our clone/archive captures all sidechains; native fork does not.

**Pre-flight Requests** (not stored in session files):

| Request | Model | Purpose |
|---------|-------|---------|
| Topic detection | Haiku | Extract 2-3 word session title |
| Agent warmup | Opus/Haiku | Pre-populate prompt cache |
| Token counting | N/A | ~100 `/count_tokens` calls (one per tool) |

**Extended Thinking**:

| Trigger | Budget | Notes |
|---------|--------|-------|
| `think` | Basic | Default for complex tasks |
| `think hard` | Increased | More reasoning depth |
| `think harder` | Significant | Extended analysis |
| `ultrathink` | Maximum (31,999 tokens) | Full reasoning budget |

- Stored in `thinking` content blocks with `signature` field
- NOT counted in context window (output tokens only)
- Configure cap via `MAX_THINKING_TOKENS` env var

**Prompt Cache** (explains `cache_read_input_tokens` field):

| Type | TTL | Write Premium | Read Savings |
|------|-----|---------------|--------------|
| Ephemeral (default) | 5 min | 25% | 90% |
| Extended | 1 hour | 100% | 90% |

Cache invalidates when: user edits/rewinds message, TTL expires, or content prefix changes. Minimum cacheable: ~1,024 tokens.

**Token Economics** (December 2025):

| Model | Base Input | Cache Write (5m) | Cache Read | Output |
|-------|------------|------------------|------------|--------|
| Opus 4.5 | $5/M | $6.25/M | $0.50/M | $15/M |
| Sonnet 4.5 | $3/M | $3.75/M | $0.30/M | $15/M |
| Haiku 4.5 | $1/M | $1.25/M | $0.10/M | $5/M |

**API-only Features** (not in Claude Code):

| Feature | Status | Notes |
|---------|--------|-------|
| Effort control | Beta API | Levels: high/medium/low. Medium = 76% fewer output tokens |
| Batch API | Production | 50% discount, async (incompatible with interactive CLI) |

**Session Startup Overhead**:

First response latency depends heavily on MCP configuration. With many MCPs:
- ~100 `/count_tokens` calls (one per tool)
- MCP OAuth handshakes (Linear, Sentry, Notion, etc.)
- Agent warmup requests (Plan/Explore)

Observed: ~5 minutes first response with 108 tools loaded. Reduce MCPs for faster startup.

## Claude Code Version Sources

Authoritative sources for checking Claude Code versions:

| Source | URL | Type | Update Speed | Best For |
|--------|-----|------|--------------|----------|
| **NPM Package** | https://www.npmjs.com/package/@anthropic-ai/claude-code | Distribution | Immediate | Definitive latest version |
| **NPM Versions Tab** | https://www.npmjs.com/package/@anthropic-ai/claude-code?activeTab=versions | History | Immediate | Full version history + downloads |
| **NPM JSON API** | https://registry.npmjs.org/@anthropic-ai/claude-code | API | Immediate | Programmatic version checks |
| **GitHub Releases** | https://github.com/anthropics/claude-code/releases | Releases | Same day | Release notes, binaries |
| **GitHub Releases API** | https://api.github.com/repos/anthropics/claude-code/releases/latest | API | Same day | Programmatic release info |
| **CHANGELOG.md** | https://github.com/anthropics/claude-code/blob/main/CHANGELOG.md | Docs | Same day | Detailed change history |
| **GitHub Repo** | https://github.com/anthropics/claude-code | Source | Continuous | Issues, source code |
| **Official Docs** | https://code.claude.com/docs/en/home | Docs | Release-aligned | Feature documentation |
| **Platform Release Notes** | https://platform.claude.com/docs/en/release-notes/overview | Docs | Major releases | API + Claude Code combined |
| **Support Release Notes** | https://support.claude.com/en/articles/12138966-release-notes | Support | Major releases | User-facing release notes |
| **ClaudeLog Changelog** | https://www.claudelog.com/claude-code-changelog/ | Community | ~Daily | Human-readable summaries |
| **ClaudeLog News** | https://www.claudelog.com/claude-news/ | Community | ~Daily | Announcement timeline |
| **Anthropic Newsroom** | https://www.anthropic.com/news | Announcements | Major only | Strategic announcements |
| **CLI** | `claude --version` | Local | N/A | Installed version |

> **Note**: The deprecated `claude-ai` NPM package has been superseded by `@anthropic-ai/claude-code`.

## Updating for New Claude Code Versions

When a new Claude Code version is released:

1. **Detect** - Run `uv run scripts/validate_models.py` to find schema gaps
2. **Analyze** - Run `uv run scripts/analyze_model_quality.py` to study new fields
3. **Update** - Add new Pydantic models/fields in `src/models.py` with version annotations
4. **Bump** - Update `SCHEMA_VERSION` and `CLAUDE_CODE_MAX_VERSION` in `src/models.py`
5. **Validate** - Re-run validation for 100% success rate
6. **Export** - Run `uv run scripts/export_schema.py` to regenerate `session-schema.json`
7. **Commit** - Format: `Update models for Claude Code X.X.X compatibility (schema vX.X.X)`

## License

MIT