# Claude Session MCP - Development Guide

## Session File Analysis

Session files are JSONL at `~/.claude/projects/<encoded-path>/<session-id>.jsonl`.

> **Implementation Gap**: `_encode_path()` in `src/services/archive.py` only handles `/` and `.` encoding. Claude Code itself encodes all four characters (`/`, `.`, ` `, `~` → `-`). Paths with spaces or tildes may cause session discovery issues.

### Quick Reference: Record Types

Core record types found in session files (schema v0.1.7):

| Type | Purpose | Key Fields | Inherits BaseRecord |
|------|---------|------------|---------------------|
| `user` | User messages and inputs | `uuid`, `timestamp`, `sessionId`, `cwd`, `message`, `parentUuid` | Yes |
| `assistant` | Claude responses | `uuid`, `timestamp`, `sessionId`, `cwd`, `message`, `usage`, `model` | Yes |
| `summary` | Session summaries | `summary` (text), `leafUuid` | **No** (no uuid/timestamp/sessionId) |
| `system` | Internal events (see subtypes below) | `uuid`, `timestamp`, `sessionId`, `cwd`, `parentUuid`, `systemType` | Yes |
| `file-history-snapshot` | Document/file state tracking | `messageId`, `snapshot`, `isSnapshotUpdate` | **No** |
| `queue-operation` | Queue management events | `operation`, `timestamp`, `sessionId`, `content` | **No** (no uuid) |

**System Record Subtypes** (all have `type='system'`, differentiated by `subtype` field):

| Subtype | Purpose | Additional Fields |
|---------|---------|-------------------|
| `local_command` | Local shell/CLI operations | `content`, `level`, `slug`, `isMeta` |
| `compact_boundary` | Session compaction markers | `content`, `compactMetadata` (trigger, preTokens), `logicalParentUuid` |
| `api_error` | Claude API failures | `error`, `retryInMs`, `retryAttempt`, `maxRetries`, `cause` |
| `informational` | General system notifications | `content`, `level` |

### Quick Reference: Message Content Types

Content blocks within `message.content` arrays (discriminated by `type` field):

| Type | Description | Key Fields |
|------|-------------|------------|
| `thinking` | Extended thinking blocks (not in context) | `thinking` (str), `signature` (str \| None) |
| `text` | Plain text from user or assistant | `text` (str) |
| `tool_use` | Tool invocations from assistant | `id`, `name`, `input` (typed or dict) |
| `tool_result` | Tool execution results from user | `tool_use_id`, `content` (str \| list \| None), `is_error` (bool \| None) |
| `image` | Base64 images from user | `source` (ImageSource with base64 `data`, `media_type`) |
| `document` | PDF/document uploads from user | `source` (DocumentSource with base64 `data`, `media_type`) |

### Field Glossary

Common fields and their meanings:

- **`uuid`**: Unique record identifier (not present on summary/file-history-snapshot/queue-operation records)
- **`timestamp`**: ISO 8601 timestamp of record creation
- **`sessionId`**: Session identifier (format: `019xxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)
- **`parentUuid`**: Links to preceding message; establishes conversation thread (null for root records)
- **`cwd`**: Working directory at time of record (absolute path)
- **`isSidechain`**: Message from an agent or subprocess (references `agent-{agentId}.jsonl`)
- **`slug`**: Human-readable session identifier (e.g., "jiggly-churning-rabbit") - may change within session
- **`leafUuid`**: Most recent message in a conversation branch (used in summary records)
- **`compactMetadata`**: Session compression metadata (`trigger`: auto/manual, `preTokens`: token count before compaction)
- **`usage`**: Token consumption (`input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`)
- **`toolUseResult`**: Structured tool execution metadata (varies by tool type)

### Checking Token Usage

Token counts are in `message.usage` on assistant records:

```python
total_tokens = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
```

This is the **message context** only. Full context includes overhead:

| Component | Typical Size | Notes |
|-----------|--------------|-------|
| System prompt | ~3k | Base Claude Code instructions |
| System tools | ~16k | Built-in tools |
| MCP tools | 0-50k+ | Varies by configured MCPs |
| Memory files | varies | CLAUDE.md files |
| Messages | varies | Conversation history |

Run `/context` in Claude Code for authoritative breakdown.

### Finding Compaction Events

```bash
rg -n "compact_boundary" <session-file>.jsonl
```

Each has `compactMetadata.trigger` (usually "manual") and `compactMetadata.preTokens`.

### "Prompt is too long" Errors

When the API rejects a prompt, Claude Code logs a synthetic response:

```json
{
  "type": "assistant",
  "message": {
    "model": "<synthetic>",
    "content": [{"type": "text", "text": "Prompt is too long"}]
  }
}
```

This indicates context exceeded ~200k. Check preceding records for actual token counts.

### Cache Invalidation

`cache_read_input_tokens` drops significantly when:
- User edits a previous message (branches conversation)
- User goes back to retry from earlier point
- Session structure changes

This is NOT compaction - the tokens are still there, just not cached.

## Session Search

Searching session files is non-trivial due to content characteristics.

### Noise Sources

1. **Tool results**: Contain base64 images, previous search outputs, large file contents
2. **Self-referential**: Your search queries appear in session history as searchable content
3. **Nested JSON**: Content is JSON within JSONL - naive regex matches structural elements

A simple `rg "topic" ~/.claude/projects/` will match tool outputs, not just user intent.

### Layered Search Strategy

**Layer 1: Filter by record type** (eliminates 50-70% noise)
```bash
jq -r 'select(.type=="user") | .message.content' ~/.claude/projects/*/*.jsonl | rg -i "search term"
```

**Layer 2: Filter by content size** (exclude large pastes, base64)
```bash
jq -r 'select(.type=="user") | .message.content | select(type=="string" and length < 5000)' \
  ~/.claude/projects/*/*.jsonl | rg -i "search term"
```

**Layer 3: Session-level grouping** (examine matching files)
```bash
for file in $(rg -l "topic" ~/.claude/projects/*/*.jsonl 2>/dev/null); do
  echo "=== $file ==="
  jq -r 'select(.type=="user") | .message.content | if type=="string" then .[0:200] else .[0].text?[0:200] // "" end' "$file" 2>/dev/null | head -5
done
```

**Layer 4: Summary-first discovery** (fastest)
```bash
jq -r 'select(.type=="summary") | .summary' ~/.claude/projects/*/*.jsonl | rg -i "topic"
```

## Validation

```bash
uv run ./scripts/validate_models.py
```

Validates all session files against Pydantic models. 100% pass rate expected.

## Analyzing Session Token Usage

**Always use Pydantic models** for session analysis - never raw `json.loads()` without validation. The models provide type safety, IDE autocomplete, and catch schema changes early:

```python
from pathlib import Path
from pydantic import TypeAdapter, BaseModel
from src.models import SessionRecord, UserRecord, AssistantRecord, CompactBoundarySystemRecord
import json

adapter = TypeAdapter(SessionRecord)

# Parse session with Pydantic
records = []
with open(session_path) as f:
    for line in f:
        record = adapter.validate_python(json.loads(line))
        records.append(record)

# Find last compact_boundary
last_compact_idx = None
for i, record in enumerate(records):
    if isinstance(record, CompactBoundarySystemRecord):
        last_compact_idx = i
        print(f"Compact at {i}: preTokens={record.compactMetadata.preTokens}")

# Analyze records after compact
for record in records[last_compact_idx + 1:]:
    if isinstance(record, UserRecord) and record.message:
        for block in record.message.content:
            if block.type == 'tool_result':
                # block.content is the tool output
                pass
            elif block.type == 'text':
                # block.text is user message
                pass

    elif isinstance(record, AssistantRecord) and record.message:
        for block in record.message.content:
            if block.type == 'thinking':
                # block.thinking - NOT counted in context
                pass
            elif block.type == 'text':
                # block.text - assistant response
                pass
            elif block.type == 'tool_use':
                # block.input - Pydantic model, use .model_dump_json()
                pass
```

### Preserving Context Before Compact

Before running `/compact`, ask Claude to generate a handoff document:

> "Make a self-contained handoff artifact that the next AI can use to continue. Include executive summary, current state, pending work, and key decisions."

This produces a structured document (~12k chars) that's **larger and more useful** than the auto-generated compact summary (~9k chars). The handoff survives in session history as additional context.

The compact summary is stored in the first user message after `compact_boundary` as `message.content` (string, not list).

### Estimating Message Tokens

Claude's tokenizer differs from tiktoken, but rough estimates:

| Content Type | In Context? | Estimate |
|--------------|-------------|----------|
| Thinking blocks | NO | Generated output, not input |
| Tool results | ~45% | Truncated/compressed |
| User/assistant text | ~100% | Full content |
| Tool use inputs | ~100% | Full content |

## Operational Patterns

### Finding Session Files

Locate a session file by ID prefix:

```bash
find ~/.claude/projects -name "019b342b*.jsonl"
```

### One-off Dependencies

Use `uv run --with` for temporary dependencies without modifying pyproject.toml:

```bash
uv run --with tiktoken python3 << 'EOF'
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
# ...
EOF
```

### Large Outputs to Files

When inline response is too limited (~16k chars), write to a file instead:

```python
# Use Write tool to create handoff-<session-id>.md
# Can produce 30k+ char documents vs ~16k inline limit
```

### Comparing Handoff vs Compact

Measure preservation strategy effectiveness:

```python
# Find handoff document (large assistant text before compact)
# Find compact summary (first user message after compact_boundary, stored as string)
# Compare char lengths - handoff typically ~25% larger
```

## Intercepting Claude Code API Traffic

Use mitmproxy to see exactly what Claude Code sends to the API.

### Setup

```bash
# Install
brew install mitmproxy

# Create intercept script at /tmp/intercept_claude.py
cat > /tmp/intercept_claude.py << 'EOF'
"""Capture Claude Code API traffic."""
from mitmproxy import http
import json
from datetime import datetime

LOG_FILE = "/tmp/claude_traffic.log"

def request(flow: http.HTTPFlow):
    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"[{datetime.now().isoformat()}] REQUEST\n")
        f.write(f"Host: {flow.request.host}\n")
        f.write(f"URL: {flow.request.url}\n")
        f.write(f"Content-Length: {len(flow.request.content or b'')} bytes\n")

        if "anthropic.com" in flow.request.host and flow.request.content:
            try:
                body = json.loads(flow.request.content)
                path = flow.request.path.replace("/", "_").replace("?", "_")[:50]
                filename = f"/tmp/req_{path}_{datetime.now().strftime('%H%M%S%f')}.json"
                with open(filename, "w") as req_f:
                    json.dump(body, req_f, indent=2)
                f.write(f"Saved to: {filename}\n")
            except:
                pass

def response(flow: http.HTTPFlow):
    with open(LOG_FILE, "a") as f:
        f.write(f"RESPONSE: {flow.response.status_code} ({len(flow.response.content or b'')} bytes)\n")

    if "anthropic.com" in flow.request.host and flow.response.content:
        try:
            resp = json.loads(flow.response.content)
            path = flow.request.path.replace("/", "_").replace("?", "_")[:50]
            filename = f"/tmp/resp_{path}_{datetime.now().strftime('%H%M%S%f')}.json"
            with open(filename, "w") as resp_f:
                json.dump(resp, resp_f, indent=2)
        except:
            pass
EOF

# Start proxy
mitmdump -p 8080 -s /tmp/intercept_claude.py --quiet &

# Run Claude through proxy
HTTPS_PROXY=http://localhost:8080 NODE_TLS_REJECT_UNAUTHORIZED=0 claude
```

### Discovered Endpoints

| Endpoint | Purpose |
|----------|---------|
| `api.anthropic.com/v1/messages` | Main conversation API |
| `api.anthropic.com/v1/messages/count_tokens` | Token counting (called per tool!) |
| `api.anthropic.com/api/event_logging/batch` | Telemetry |
| `statsig.anthropic.com` | Feature flags |

### Analyzing Captured Requests

```bash
# View traffic log
cat /tmp/claude_traffic.log

# List captured request/response files
ls -la /tmp/req_*.json /tmp/resp_*.json

# Inspect a messages request
cat /tmp/req_*messages*.json | jq '{
  model,
  max_tokens,
  system_chars: (.system | tostring | length),
  messages_count: (.messages | length),
  tools_count: (.tools | length)
}'

# See actual user message
cat /tmp/req_*messages*.json | jq '.messages[2]'
```

### Key Finding: Tool Token Overhead

With 108 MCP tools configured, ~75% of context is tools:

| Component | Chars | Est. Tokens |
|-----------|-------|-------------|
| Tools | 130k | ~32k |
| System prompt | 15k | ~4k |
| Messages | varies | varies |

This is why `/context` shows 43k tokens for a simple "Hello" message.

## Schema Updates

When Claude Code updates break validation:
1. Run validate_models.py to find failures
2. Check field paths in error messages
3. Update `src/models.py` with new fields/types
4. Bump schema version in models.py header