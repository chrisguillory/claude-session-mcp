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
# Archive to local file
claude-session archive <session-id> output.json

# Archive to GitHub Gist (requires GITHUB_TOKEN)
export GITHUB_TOKEN=$(gh auth token)
claude-session archive <session-id> gist://

# Restore from file
claude-session restore archive.json

# Restore from Gist
claude-session restore gist://<gist-id>

# Resume restored session
claude --resume <new-session-id>
```

## Features

- **Local storage**: Save sessions as JSON or compressed (zstd)
- **GitHub Gist**: Private/public Gist storage (100MB limit)
- **Path translation**: Automatically translates file paths when restoring to different directories
- **Agent sessions**: Captures main session + all agent sub-sessions
- **New session IDs**: Generates UUIDv7 for restored sessions (vs Claude's UUIDv4)

## Commands

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
      "args": ["run", "mcp-server.py"],
      "cwd": "/path/to/claude-session-mcp"
    }
  }
}
```

**Tools:**
- `save_current_session` - Archive current session (local storage only)
- `restore_session` - Restore archived session (local files only)

## Technical Details

### Models

- **935 lines**, 54 Pydantic classes covering all 6 record types
- **Compatible with**: Claude Code 2.0.35 - 2.0.47
- **Round-trip fidelity**: Uses `exclude_unset=True` to preserve original structure
- **Validation**: Tested on 74K+ records across 983 sessions (99.98% valid)

### Path Encoding

Claude encodes project paths in directory names:
```
/Users/chris/project      → -Users-chris-project
/Users/rushi.arumalla/app → -Users-rushi-arumalla-app
```

Both slashes and periods are replaced with dashes.

### Session ID Format

- **Claude sessions**: UUIDv4 (random)
- **Restored sessions**: UUIDv7 (time-ordered, identifiable)

## License

MIT