# claude-session-mcp

Archive and restore Claude Code sessions across machines with full conversation history.

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

**Pydantic Models**: 914 lines, 56 classes covering 10 record types (user, assistant, system, local commands, compact boundaries, API errors, summaries, file history snapshots, queue operations). Validated on 78K+ records (100% valid).

**Compatibility**: Claude Code 2.0.35 - 2.0.47

## License

MIT