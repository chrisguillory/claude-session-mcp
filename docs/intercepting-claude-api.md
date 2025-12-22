# Intercepting Claude Code API Traffic

## Goal

Monitor Claude Code's API requests to:
1. See actual token counts before hitting the 200k context limit
2. Detect when context is getting full (e.g., >150k tokens)
3. Preemptively generate a rich handoff document instead of relying on lossy auto-compact

## Key Insight: Two Approaches

### Approach 1: JSON Output (Easiest)

`claude --print --output-format json` returns token usage directly:

```json
{
  "usage": {
    "input_tokens": 3,
    "cache_creation_input_tokens": 29603,
    "cache_read_input_tokens": 13148,
    "output_tokens": 10
  },
  "contextWindow": 200000
}
```

**Total context = input_tokens + cache_creation_input_tokens + cache_read_input_tokens**

This is sufficient for monitoring context size without network interception.

### Approach 2: mitmproxy (Deeper Analysis)

Use network interception when you need to:
- See actual request payloads (system prompts, tool definitions)
- Understand what's consuming context
- Debug unexpected token usage

## The Problem with Auto-Compact

When Claude Code hits context limits, it runs `/compact` automatically, which produces a ~9k char summary. This is lossy - agent history, intermediate steps, and nuanced context are lost. By intercepting API traffic, we can:
- See the exact payload being sent
- Know how much room is left
- Generate a 30k+ char handoff document BEFORE compaction

## Setup: mitmproxy Interception

### 1. Install mitmproxy

```bash
brew install mitmproxy
```

### 2. Create the intercept script

Save to `/tmp/intercept_all.py`:

```python
"""Intercept ALL traffic to discover Claude's API endpoints."""
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
        f.write(f"Method: {flow.request.method}\n")
        f.write(f"Content-Length: {len(flow.request.content or b'')} bytes\n")

        if flow.request.content and len(flow.request.content) > 100:
            try:
                body = json.loads(flow.request.content)
                if "messages" in body or "model" in body:
                    filename = f"/tmp/request_{flow.request.host.replace('.', '_')}_{datetime.now().strftime('%H%M%S')}.json"
                    with open(filename, "w") as req_f:
                        json.dump(body, req_f, indent=2)
                    f.write(f"Saved full request to: {filename}\n")
                    f.write(f"Keys: {list(body.keys())}\n")
            except:
                pass

def response(flow: http.HTTPFlow):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] RESPONSE {flow.request.host}: {flow.response.status_code} ({len(flow.response.content or b'')} bytes)\n")
```

### 3. Start the proxy

```bash
# Kill any existing proxy
pkill -f mitmdump 2>/dev/null

# Clear old logs
rm -f /tmp/claude_traffic.log /tmp/request_*.json

# Start mitmdump
mitmdump -p 8080 -s /tmp/intercept_all.py --quiet &
```

### 4. Run Claude through the proxy

```bash
# For print mode (non-interactive)
HTTPS_PROXY=http://localhost:8080 NODE_TLS_REJECT_UNAUTHORIZED=0 \
  claude --print --output-format json "your prompt"

# For interactive mode
HTTPS_PROXY=http://localhost:8080 NODE_TLS_REJECT_UNAUTHORIZED=0 claude
```

### 5. Analyze captured traffic

```bash
# View traffic log
cat /tmp/claude_traffic.log

# View captured request bodies
ls -la /tmp/request_*.json
cat /tmp/request_api_anthropic_com_*.json | jq .
```

## Discovered API Endpoints

From interception, Claude Code calls:

| Endpoint | Purpose |
|----------|---------|
| `api.anthropic.com/v1/messages?beta=true` | Main conversation API |
| `api.anthropic.com/v1/messages/count_tokens?beta=true` | Token counting for tools |
| `api.anthropic.com/api/oauth/claude_cli/client_data` | OAuth/client data |
| `api.anthropic.com/api/eval/sdk-*` | SDK evaluation |
| `api.anthropic.com/api/event_logging/batch` | Event/telemetry logging |
| `api.anthropic.com/api/claude_code_grove` | Unknown (grove feature?) |
| `statsig.anthropic.com/v1/initialize` | Feature flags |
| `statsig.anthropic.com/v1/rgstr` | Statsig registration |

## Key Discovery: count_tokens Calls

Claude Code makes **many** `count_tokens` calls - one for each tool definition. This is how it calculates the context breakdown shown by `/context`. Each call contains:
- `model`: The model being used
- `messages`: The message(s) being counted
- `tools`: Tool definitions being counted

## Request Structure

Main `/v1/messages` request contains:
```json
{
  "model": "claude-opus-4-5-20251101",
  "max_tokens": 16000,
  "system": "...(system prompt)...",
  "messages": [...conversation...],
  "tools": [...tool definitions...],
  "metadata": {
    "user_id": "..."
  }
}
```

## Next Steps

1. **Parse count_tokens responses** to get actual token breakdown
2. **Monitor total context** across session
3. **Trigger handoff generation** at threshold (e.g., 150k tokens)
4. **Compare** intercepted counts with `/context` output for validation

## Alternative: Session File Analysis

Instead of network interception, you can also analyze token usage from session files:

```python
from pydantic import TypeAdapter
from src.models import SessionRecord, AssistantRecord
import json

adapter = TypeAdapter(SessionRecord)

# Last assistant message has usage stats
for line in reversed(open(session_file).readlines()):
    record = adapter.validate_python(json.loads(line))
    if isinstance(record, AssistantRecord) and record.message:
        usage = record.message.usage
        total = usage.input_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens
        print(f"Context tokens: {total}")
        break
```

This gives message context only - add ~55k for system overhead (prompt + tools + MCP).