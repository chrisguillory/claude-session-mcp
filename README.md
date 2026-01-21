# claude-session-mcp

Archive and restore Claude Code sessions across machines with full conversation history preserved.

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
# Install globally (recommended)
uv tool install git+https://github.com/chrisguillory/claude-session-mcp

# Run without installing
uvx --from git+https://github.com/chrisguillory/claude-session-mcp claude-session --help

# From local clone
git clone https://github.com/chrisguillory/claude-session-mcp
cd claude-session-mcp && uv sync
```

### Upgrading

```bash
uv tool upgrade claude-session-mcp
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

# Delete a cloned session (auto-backup created)
claude-session delete <session-id>

# Undo a delete
claude-session restore --in-place ~/.claude-session-mcp/deleted/<backup>.json
```

## Features

- **Lossless forking**: Clone sessions while preserving parent immutability
- **Complete artifact capture**: Session files, agent files, plan files, tool results, and todos
- **Lineage tracking**: Records parent-child relationships when sessions are cloned or restored
- **Local storage**: Save sessions as JSON or compressed (zstd)
- **GitHub Gist**: Private/public Gist storage (100MB limit)
- **Path translation**: Automatically translates file paths when restoring to different directories
- **Cross-machine tracking**: Archives include machine ID for detecting cross-machine restores
- **Delete with safety**: Auto-backup before deletion, native sessions require `--force`
- **Fail-fast checks**: Prevents partial writes if any output file already exists
- **UUIDv7 session IDs**: Cloned/restored sessions use time-ordered UUIDv7 (vs Claude's random UUIDv4)

## Understanding Claude Code Sessions

Claude Code session files (`~/.claude/projects/.../*.jsonl`) capture complete conversation history, but their format is **not officially documented** by Anthropic. This project provides typed Pydantic models derived from empirical analysis of real session records. See `src/schemas/session/models.py` for current schema version and validation coverage.

### What You Should Know

- **Unofficial schema**: The session file format is an internal implementation detail that may change without notice. We track compatibility through continuous validation against real session data.
- **CLI vs storage**: What you see in the Claude Code terminal is a visual "veneer" over the underlying data. Tool operations that appear as single units are actually stored as two linked records ([details below](#tool-useresult-pairing)).
- **Empirical coverage**: Our models cover observed patterns. New Claude Code versions may introduce unmodeled fields or record types—run `validate_models.py` to detect drift.

### Related Projects

Other projects working with Claude Code sessions:

| Project                                                                                  | Purpose                                                                                                      |
|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| [claude-code-transcripts](https://github.com/simonw/claude-code-transcripts)             | Convert sessions to browseable HTML (Simon Willison)                                                         |
| [claude-JSONL-browser](https://github.com/withLinda/claude-JSONL-browser)                | Web-based session log viewer                                                                                 |
| [claude-conversation-extractor](https://pypi.org/project/claude-conversation-extractor/) | Export conversations to Markdown                                                                             |
| [ccrider](https://github.com/neilberkman/ccrider)                                        | Session analysis with [schema research](https://github.com/neilberkman/ccrider/blob/main/research/schema.md) |
| [claude_code_session_client](https://github.com/randombet/claude_code_session_client)    | Python client for session file parsing                                                                       |

*Know of another project? Open an issue or PR.*

## Clone vs Native Fork

Claude Code 2.0.73+ supports `--fork-session`. Here's how it compares to our `clone` command:

|                         | Claude Native Fork                    | MCP Clone                   |
|-------------------------|---------------------------------------|-----------------------------|
| **Command**             | `claude --resume <id> --fork-session` | `claude-session clone <id>` |
| **Mechanism**           | Compact (summary)                     | Full copy                   |
| **Session size**        | ~34 KB                                | ~1.1 MB                     |
| **Context**             | Summary blob                          | Complete history            |
| **Fidelity**            | Lossy                                 | Lossless                    |
| **Agent history**       | Not preserved                         | Preserved                   |
| **Tool results**        | Not preserved                         | Preserved                   |
| **Todos**               | Not preserved                         | Preserved                   |
| **Plan files**          | Not preserved                         | Preserved                   |
| **Cross-machine**       | No                                    | Yes                         |
| **Gist support**        | No                                    | Yes                         |
| **Path translation**    | No                                    | Yes                         |
| **Parent immutability** | N/A (modifies original)               | Guaranteed                  |

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
#   --in-place         Restore with original session ID (verbatim restore)
#   --launch, -l       Launch Claude Code after restore
#   --gist-token       GitHub token for private gists
#   --verbose, -v      Verbose output
```

By default, restore generates a new UUIDv7 session ID and transforms all artifact IDs. Use `--in-place` for verbatim restoration with original IDs (useful for undoing a delete).

### Delete

```bash
claude-session delete <session-id>

# Arguments:
#   session-id     Session ID to delete

# Options:
#   --force, -f        Required to delete native (UUIDv4) sessions
#   --no-backup        Skip auto-backup before deletion
#   --dry-run          Preview what would be deleted
#   --project, -p      Project directory (default: current)
#   --verbose, -v      Verbose output
```

**Safety features:**
- By default, only cloned/restored sessions (UUIDv7) can be deleted
- Native Claude sessions (UUIDv4) require `--force` to prevent accidental deletion
- Auto-backup created at `~/.claude-session-mcp/deleted/` before deletion (unless `--no-backup`)
- Use `restore --in-place <backup>` to undo a delete

### Lineage

View parent-child relationships for cloned/restored sessions:

```bash
claude-session lineage <session-id>

# Arguments:
#   session-id     Session ID (full UUID or prefix)

# Options:
#   --format, -f       Output format: text, tree, or json (default: text)
```

**Examples:**

```bash
# View lineage for a session
claude-session lineage 019b53ff
# Output:
# Session: 019b53ff-4ef1-750a-9234-500b42ac818e
# Parent:  c3bac5a6-f519-4ef8-8dbc-7f7f030ebe5b
# Cloned:  2025-12-29 10:30:00+00:00
# Method:  clone
# Source:  /Users/alice/project
# Target:  /Users/bob/project
# Machine: bob@macbook.local

# View ancestry tree (multi-generational)
claude-session lineage 019b6b8f --format tree
# Output:
# c3bac5a6-f519-4ef8-8dbc-7f7f030ebe5b
#   └─ 019b53ff-4ef1-750a-9234-500b42ac818e
#     └─ 019b6b8f

# Export as JSON
claude-session lineage 019b53ff --format json
```

**Lineage tracking:**
- Lineage is stored in `~/.claude-session-mcp/lineage.json`
- Tracks: parent/child session IDs, timestamps, project paths, machine IDs
- Cross-machine restores are detected and highlighted
- Native (UUIDv4) sessions have no lineage entry

## MCP Server

Install the MCP server to use archive/restore directly from Claude Code:

```bash
# If installed globally (see Installation)
claude mcp add --scope user claude-session -- claude-session-mcp

# From GitHub
claude mcp add --scope user claude-session -- uvx --refresh --from git+https://github.com/chrisguillory/claude-session-mcp claude-session-mcp

# From local clone
claude mcp add --scope user claude-session -- uv run --project ~/claude-session-mcp claude-session-mcp
```

**Tools:**

MCP tools are session-aware - they automatically know the current session ID and project path.

| Tool                   | Description                                                           |
|------------------------|-----------------------------------------------------------------------|
| `save_current_session` | Archive current session (local JSON or zst)                           |
| `restore_session`      | Restore from archive with new UUIDv7 ID                               |
| `clone_session`        | Clone session directly. Defaults to current session ("fork yourself") |
| `delete_session`       | Delete session with auto-backup. Native sessions require `force=True` |
| `session_lineage`      | Get lineage info. Defaults to current session                         |

**Note**: Unlike CLI, MCP tools can't auto-launch Claude after operations.

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

| Type   | Source             | Pattern                  | Example                  |
|--------|--------------------|--------------------------|--------------------------|
| UUIDv4 | Claude Code native | `xxxxxxxx-xxxx-4xxx-...` | `a1b2c3d4-e5f6-4a7b-...` |
| UUIDv7 | Cloned/restored    | `01xxxxxx-xxxx-7xxx-...` | `01934a2b-c3d4-7e5f-...` |

The version digit (4 or 7) is at position 15 (first digit of third group).

### Archive Format Versions

| Version | Added Fields            | Description                        |
|---------|-------------------------|------------------------------------|
| 1.0     | `files`                 | Session JSONL files only           |
| 1.1     | `plan_files`            | Added plan file content            |
| 1.2     | `tool_results`, `todos` | Complete artifact capture          |
| 1.3     | `machine_id`            | Cross-machine lineage tracking     |

Archives are backwards compatible - older archives can be restored by newer versions.

### Session Artifacts

Complete inventory of session-specific data in `~/.claude/`:

| Artifact      | Location                                         | Transferred | Notes                                          |
|---------------|--------------------------------------------------|-------------|------------------------------------------------|
| Session JSONL | `projects/<enc-path>/<session-id>.jsonl`         | ✅           | Main conversation history                      |
| Agent JSONL   | `projects/<enc-path>/agent-<agent-id>.jsonl`     | ✅           | Sidechain conversations (via sessionId search) |
| Plan files    | `plans/<slug>.md`                                | ✅           | Plan mode documents (via slug extraction)      |
| Tool results  | `projects/<enc-path>/<session-id>/tool-results/` | ✅           | Large tool outputs                             |
| Todos         | `todos/<session-id>-agent-<agent-id>.json`       | ✅           | TodoWrite state                                |
| Session-env   | `session-env/<session-id>/`                      | ✅           | Currently empty; validated and recreated       |
| Debug logs    | `debug/<session-id>.txt`                         | ❌           | Ephemeral local debugging logs                 |
| File history  | `file-history/<session-id>/`                     | ❌           | Ephemeral local file version snapshots         |

**Why some artifacts aren't transferred**: Debug logs and file history are intentionally excluded - they're ephemeral local state for debugging and file rollback that aren't really needed for session continuity and can't be meaningfully used on a different machine.

**Delete behavior**: Currently removes transferred artifacts only. Debug logs and file history cleanup is a planned enhancement.

**Global data** (not session-specific, never modified): `history.jsonl`, `settings.json`, `statsig/`, `shell-snapshots/`

### Cloned Artifact Identification

When cloning, artifacts get new IDs to preserve parent immutability:

| Artifact   | Native                           | Cloned                                 |
|------------|----------------------------------|----------------------------------------|
| Session ID | `a1b2c3d4-...-4xxx-...` (UUIDv4) | `019xxxxx-...-7xxx-...` (UUIDv7)       |
| Plan slug  | `linked-twirling-tower`          | `linked-twirling-tower-clone-019b5232` |
| Agent ID   | `5271c147`                       | `5271c147-clone-019b51bd`              |

**Agent ID uses flat cloning**: When cloning a clone, the base hex is extracted and a new suffix applied (no accumulation of `-clone-` segments). For example, cloning `5271c147-clone-019b51bd` produces `5271c147-clone-019c1234`, not `5271c147-clone-019b51bd-clone-019c1234`.

### Pydantic Models

Strict typed models for all session record types, message content blocks, and tool inputs/results. See `src/schemas/session/models.py` header for current class count and validation coverage.

### Tool Use/Result Pairing

Claude Code's CLI presents tool operations as unified visual blocks:

```
⏺ Read config.json
  ⎿  {"setting": "value"}
```

However, these are stored as **two separate JSONL records**:

1. **AssistantRecord** with `message.content[].type = "tool_use"` — Claude's request
2. **UserRecord** with `message.content[].type = "tool_result"` — System's response

The records are linked via `tool_use.id` → `tool_result.tool_use_id`. Errors set `is_error: true` on the tool_result, and the error message appears both in `message.content[].content` and the top-level `toolUseResult` field.

This pattern follows the [Claude API tool use specification](https://docs.anthropic.com/en/docs/build-with-claude/tool-use), which is officially documented—unlike the session file format itself.

### Compatibility

See `CLAUDE_CODE_MIN_VERSION` and `CLAUDE_CODE_MAX_VERSION` in `src/schemas/session/models.py` for current supported range.

### Claude Code Architecture Reference

Understanding what gets captured in session files.

**Built-in Agents**:

| Agent           | Model       | Purpose                        | Creates Sidechain |
|-----------------|-------------|--------------------------------|-------------------|
| Plan            | Opus/Sonnet | Software architect (read-only) | Yes               |
| Explore         | Haiku       | File search specialist         | Yes               |
| General-purpose | Sonnet      | Complex multi-step tasks       | Yes               |

Agent invocations create separate `agent-{agentId}.jsonl` files, referenced via `isSidechain: true` in the main session. Our clone/archive captures all sidechains; native fork does not.

**Pre-flight Requests** (not stored in session files):

| Request         | Model      | Purpose                                   |
|-----------------|------------|-------------------------------------------|
| Topic detection | Haiku      | Extract 2-3 word session title            |
| Agent warmup    | Opus/Haiku | Pre-populate prompt cache                 |
| Token counting  | N/A        | ~100 `/count_tokens` calls (one per tool) |

**Extended Thinking**:

| Trigger      | Budget                  | Notes                  |
|--------------|-------------------------|------------------------|
| `ultrathink` | Maximum (31,999 tokens) | Full reasoning budget  |
| Shift+Tab    | Configurable            | Toggle thinking on/off |

- Stored in `thinking` content blocks with `signature` field
- NOT counted in context window (output tokens only)
- Configure cap via `MAX_THINKING_TOKENS` env var
- See [docs/thinking.md](docs/thinking.md) for deeper research on ThinkingMetadata structure

**Prompt Cache** (explains `cache_read_input_tokens` field):

| Type                | TTL    | Write Premium | Read Savings |
|---------------------|--------|---------------|--------------|
| Ephemeral (default) | 5 min  | 25%           | 90%          |
| Extended            | 1 hour | 100%          | 90%          |

Cache invalidates when: user edits/rewinds message, TTL expires, or content prefix changes. Minimum cacheable: ~1,024 tokens.

**Token Economics** (December 2025):

| Model      | Base Input | Cache Write (5m) | Cache Read | Output |
|------------|------------|------------------|------------|--------|
| Opus 4.5   | $5/M       | $6.25/M          | $0.50/M    | $15/M  |
| Sonnet 4.5 | $3/M       | $3.75/M          | $0.30/M    | $15/M  |
| Haiku 4.5  | $1/M       | $1.25/M          | $0.10/M    | $5/M   |

**API-only Features** (not in Claude Code):

| Feature        | Status     | Notes                                                     |
|----------------|------------|-----------------------------------------------------------|
| Effort control | Beta API   | Levels: high/medium/low. Medium = 76% fewer output tokens |
| Batch API      | Production | 50% discount, async (incompatible with interactive CLI)   |

**Session Startup Overhead**:

First response latency depends heavily on MCP configuration. With many MCPs:
- ~100 `/count_tokens` calls (one per tool)
- MCP OAuth handshakes (Linear, Sentry, Notion, etc.)
- Agent warmup requests (Plan/Explore)

Observed: ~5 minutes first response with 108 tools loaded. Reduce MCPs for faster startup.

## Claude Code Version Sources

Authoritative sources for checking Claude Code versions:

| Source                     | URL                                                                        | Type          | Update Speed    | Best For                         |
|----------------------------|----------------------------------------------------------------------------|---------------|-----------------|----------------------------------|
| **NPM Package**            | https://www.npmjs.com/package/@anthropic-ai/claude-code                    | Distribution  | Immediate       | Definitive latest version        |
| **NPM Versions Tab**       | https://www.npmjs.com/package/@anthropic-ai/claude-code?activeTab=versions | History       | Immediate       | Full version history + downloads |
| **NPM JSON API**           | https://registry.npmjs.org/@anthropic-ai/claude-code                       | API           | Immediate       | Programmatic version checks      |
| **GitHub Releases**        | https://github.com/anthropics/claude-code/releases                         | Releases      | Same day        | Release notes, binaries          |
| **GitHub Releases API**    | https://api.github.com/repos/anthropics/claude-code/releases/latest        | API           | Same day        | Programmatic release info        |
| **CHANGELOG.md**           | https://github.com/anthropics/claude-code/blob/main/CHANGELOG.md           | Docs          | Same day        | Detailed change history          |
| **GitHub Repo**            | https://github.com/anthropics/claude-code                                  | Source        | Continuous      | Issues, source code              |
| **Official Docs**          | https://code.claude.com/docs/en/home                                       | Docs          | Release-aligned | Feature documentation            |
| **Platform Release Notes** | https://platform.claude.com/docs/en/release-notes/overview                 | Docs          | Major releases  | API + Claude Code combined       |
| **Support Release Notes**  | https://support.claude.com/en/articles/12138966-release-notes              | Support       | Major releases  | User-facing release notes        |
| **ClaudeLog Changelog**    | https://www.claudelog.com/claude-code-changelog/                           | Community     | ~Daily          | Human-readable summaries         |
| **ClaudeLog News**         | https://www.claudelog.com/claude-news/                                     | Community     | ~Daily          | Announcement timeline            |
| **Anthropic Newsroom**     | https://www.anthropic.com/news                                             | Announcements | Major only      | Strategic announcements          |
| **CLI**                    | `claude --version`                                                         | Local         | N/A             | Installed version                |

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