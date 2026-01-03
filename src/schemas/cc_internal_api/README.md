# Claude Code Internal API Schemas

Pydantic models for **Claude Code's internal API usage patterns** - how Claude Code structures requests and handles responses when communicating with the Anthropic Messages API.

## What This Module Models

This module captures **how Claude Code uses the API**, which is distinct from:
- The public SDK types (documented, but CC may use differently)
- What Claude Code persists to session files (may differ from wire format)

Claude Code uses the **same Messages API** as the public SDK, but with specific patterns:
- Particular system prompt structures
- Caching strategies (cache_control directives)
- Tool definition formats
- Startup sequences and warmup calls

## Authentication Modes

Claude Code supports **two authentication modes** using the same API:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CLAUDE CODE AUTHENTICATION MODES                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OAUTH (SUBSCRIPTION)              │  API KEY (PER-TOKEN)                   │
│  ════════════════════              │  ═══════════════════                   │
│  • Default for Pro/Team/Max users  │  • User-configured API key             │
│  • Subscription covers usage       │  • Per-token billing applies           │
│  • Hard limits - blocked at cap    │  • No usage limits (just billing)      │
│  • No direct cost visibility       │  • Full cost visibility                │
│                                    │                                        │
│  SAME API ENDPOINT                 │  SAME API ENDPOINT                     │
│  api.anthropic.com/v1/messages     │  api.anthropic.com/v1/messages         │
│                                                                             │
│  THIS MODULE MODELS BOTH ◄─────────────────────────────────────────────────│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**When subscription limits are hit:**
- Usage is **blocked** (no automatic fallback)
- Options: upgrade subscription tier (also has limits), or configure an API key
- API key is a **separate manual configuration**, not automatic

**Cost implications:**
- With **API keys**: Per-token pricing applies directly - cache optimization saves real money
- With **OAuth**: Subscription covers usage up to tier limits
- Understanding CC's caching strategies matters for **API key users** especially

## Why Observation-Based?

While the Messages API is documented, Claude Code's **specific usage patterns** are not:
- Which fields does CC actually send?
- How does CC structure system blocks for caching?
- What warmup/preflight calls does CC make?
- Are there CC-specific fields or behaviors?

We cannot assume the public SDK documentation fully describes CC's behavior.

## Observed Behavior (via mitmproxy)

*Last captured: 2026-01-02, session ee8c7363, zero MCP tools*

### Traffic Breakdown

From a single "Hello!" interaction:
- **143 total requests** captured
- 89 requests (62%) to statsig.anthropic.com (feature flags)
- 47 requests to api.anthropic.com
- 6 requests to storage.googleapis.com (version check)

### Discovered Endpoints

**api.anthropic.com:**
| Endpoint | Count | Purpose |
|----------|-------|---------|
| `/v1/messages` | 6 | Main conversation API |
| `/v1/messages/count_tokens` | 14 | Token counting (per tool!) |
| `/api/hello` | 13 | Health check (~30s interval) |
| `/api/event_logging/batch` | 7 | Telemetry |
| `/api/oauth/claude_cli/client_data` | 1 | OAuth client data |
| `/api/oauth/account/settings` | 1 | Account settings |
| `/api/organization/{org_id}/claude_code_sonnet_1m_access` | 1 | Sonnet 1M access check |
| `/api/claude_code_grove` | 1 | Unknown (grove feature?) |
| `/api/eval/sdk-*` | 1 | SDK evaluation |

**statsig.anthropic.com:**
| Endpoint | Count | Purpose |
|----------|-------|---------|
| `/v1/rgstr` | 85 | Feature flag event tracking |
| `/v1/initialize` | 4 | Feature flag initialization |

**Other:**
| Host | Purpose |
|------|---------|
| `storage.googleapis.com` | Version check (claude-code-dist) |
| `http-intake.logs.us5.datadoghq.com` | External logging (Datadog) |

### Claude Code Startup Sequence

When Claude Code starts a conversation:

```
1. TOPIC DETECTION
   └─ Haiku preflight to extract conversation title
   └─ Uses JSON prefill technique

2. AGENT WARMUP
   └─ Sends messages to Plan and Explore agents
   └─ Primes caches for frequently-used agents

3. TOKEN COUNTING
   └─ ~100 /count_tokens calls (one per configured tool!)
   └─ Measures tool definition overhead

4. MAIN REQUEST
   └─ Full conversation request (~118KB with many tools)
   └─ Includes system prompt, tools, messages

5. SUGGESTION_MODE
   └─ Follow-up prompt for suggesting user's next input
   └─ Generates the "Suggestions:" at end of response
```

### Agent Behavior

| Agent                 | Default Model | Purpose                       | Gets Warmup? |
|-----------------------|---------------|-------------------------------|--------------|
| general-purpose       | sonnet        | Complex multi-step tasks      | No           |
| Explore               | haiku         | Fast codebase search          | **Yes**      |
| Plan                  | (inherits)    | Software architect            | **Yes**      |
| claude-code-guide     | haiku         | Documentation lookup          | No           |
| code-review-validated | -             | Code review with verification | No           |

### Tool Token Overhead

**Note:** The following was observed with an extensive MCP configuration (108 tools). With fewer or zero MCP tools, overhead will be significantly lower.

| Component        | Observed Size | Notes                                    |
|------------------|---------------|------------------------------------------|
| Tool definitions | ~32k tokens   | With 108 MCP tools (~75% of context)     |
| System prompt    | ~4k tokens    | Base CC instructions (~10% of context)   |
| Messages         | varies        | Conversation history                     |

With **zero MCP tools**, expect ~4-6k tokens overhead (system prompt only). This explains why `/context` shows large token counts even for simple messages when many tools are configured.

## Why Separate from Session Schemas?

Session files (`~/.claude/projects/.../*.jsonl`) represent what Claude Code **chooses to persist**, not necessarily what flows over the wire.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SESSION FILES vs API TRAFFIC                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Session JSONL (what CC persists)     API Traffic (what CC sends/receives)  │
│  ────────────────────────────────     ────────────────────────────────────  │
│  ✓ Message content                    ✓ Message content                     │
│  ✓ Tool calls and results             ✓ Tool calls and results              │
│  ✓ Token counts (in usage)            ✓ Token counts                        │
│  ✗ NO cache_control directives        ✓ cache_control on requests           │
│  ✗ NO tool definitions sent           ✓ Full tool JSON schemas              │
│  ✗ NO system prompt structure         ✓ System blocks with cache markers    │
│  ✗ NO request metadata                ✓ Request metadata, headers           │
│  + Local metadata (uuid, sessionId)   - No local metadata                   │
│                                                                             │
│  RESULT: Cannot understand full request context from session files alone    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

The CLI could also be:
- Stripping API-only fields before saving
- Transforming data formats
- Adding local-only metadata
- Omitting sensitive information

**We maintain separate schemas to allow independent evolution and document differences as discovered.**

## Session Schema Correspondence

Each type documents its corresponding session schema (if any):

```python
class ApiUsage(PermissiveModel):
    """
    Token usage in API response.

    CORRESPONDING SESSION TYPE: src.schemas.session.TokenUsage
    KNOWN DIFFERENCES:
    - Field names appear identical
    - Session version validated against 297k+ records
    - API version needs traffic validation
    """
```

| API Type          | Session Counterpart  | Notes                       |
|-------------------|----------------------|-----------------------------|
| `ApiUsage`        | `session.TokenUsage` | Fields appear identical     |
| `ApiMessage`      | `session.Message`    | Session adds wrapper fields |
| `CacheControl`    | *(none)*             | API-only, not persisted     |
| `SystemBlock`     | *(none)*             | API-only, not persisted     |
| `ToolDefinition`  | *(none)*             | API-only, not persisted     |
| `MessagesRequest` | *(none)*             | Requests not persisted      |

## API Reference (from Public Docs)

The following is from public documentation. Claude Code uses the same API, but may use it differently than documented patterns suggest.

### Cache Control
- Type: `{"type": "ephemeral"}`
- TTL options: `"5m"` (default) or `"1h"`
- Minimum 1,024 tokens for caching (2,048 for some models)
- Maximum 4 cache breakpoints per request
- Cache reads cost 10% of standard input rate

### Stop Reasons
```
end_turn                      - Natural completion
max_tokens                    - Hit token limit
stop_sequence                 - Custom stop sequence matched
tool_use                      - Tool invocation requested
pause_turn                    - Server tool pausing
model_context_window_exceeded - Context limit reached
refusal                       - Model declined to respond
```

### System Parameter
API accepts system as either:
- Simple string: `system: str`
- Array of blocks: `system: list[SystemBlock]` (with cache_control)

*Must verify Claude Code's actual usage pattern via traffic capture.*

### Pricing (API key users)

These prices apply when using API key authentication:

| Model      | Input | Output | Cache Write | Cache Read |
|------------|-------|--------|-------------|------------|
| Opus 4.5   | $5/M  | $25/M  | +25%        | 10%        |
| Sonnet 4.5 | $3/M  | $15/M  | +25%        | 10%        |
| Haiku 4.5  | $1/M  | $5/M   | +25%        | 10%        |

**Why cache optimization matters for everyone:**
- **API key users**: Direct cost savings
- **Subscription users**: Lower usage = more conversations within tier limits
- Tool overhead (varies by MCP configuration) impacts both cost and limits

## Observation-Based Development

### PermissiveModel Base

API schemas use `extra='ignore'` (not `extra='forbid'`) during observation:

```python
class PermissiveModel(BaseModel):
    """Allows unknown fields during traffic analysis."""
    model_config = ConfigDict(
        extra='ignore',   # Accept unknown fields
        strict=True,
        frozen=True,
    )
```

This lets us parse real traffic even with fields we haven't modeled yet.

### Validation Status

Each field should document its observation status:

| Status      | Meaning                                 |
|-------------|-----------------------------------------|
| `VALIDATED` | Observed in captured traffic            |
| `INFERRED`  | Derived from session files (may differ) |
| `REFERENCE` | From public docs (unverified for CC)    |

### Workflow

```
1. START PERMISSIVE
   PermissiveModel, loose types (dict[str, Any], str)
   Ensures we can parse any traffic

2. CAPTURE TRAFFIC
   mitmproxy (see CLAUDE.md for setup)
   Save to /tmp/req_*.json, /tmp/resp_*.json

3. ANALYZE
   Examine actual field values and structures
   Compare to session schemas and public API docs

4. TIGHTEN TYPES
   Replace dict[str, Any] → specific models
   Replace str → Literal[...] where constrained

5. DOCUMENT SOURCES
   Note observation date and capture context
   Track differences from session schemas

6. (Eventually) SWITCH TO STRICT
   Once confident, change to extra='forbid'
```

## Traffic Capture

See CLAUDE.md for full mitmproxy setup. Quick reference:

```bash
# Start proxy with intercept script
mitmdump -p 8080 -s /tmp/intercept_claude.py --quiet &

# Run Claude through proxy
HTTPS_PROXY=http://localhost:8080 NODE_TLS_REJECT_UNAUTHORIZED=0 claude
```

Captures save to `/tmp/req_*.json` and `/tmp/resp_*.json`.

## Module Structure

```
cc_internal_api/
├── __init__.py            # Re-exports (~80 types)
├── README.md              # This file
├── base.py                # PermissiveModel, FromSession, FromSdk markers
├── common.py              # Shared types (CacheControl, ApiUsage)
├── request.py             # Request schemas (MessagesRequest, SystemBlock, ToolDefinition)
├── response.py            # Response schemas (MessagesResponse)
├── streaming.py           # SSE event schemas (8 event types, discriminated unions)
├── rate_limits.py         # Rate limit header extraction (UnifiedRateLimit)
├── telemetry.py           # Telemetry schemas (TelemetryBatchRequest, TelemetryEnv)
└── internal_endpoints.py  # Claude Code internal APIs (count_tokens, grove, eval, oauth)
```

## Observed Request Structure

From captured traffic (2026-01-02, session ee8c7363):

```json
{
  "model": "claude-opus-4-5-20251101",
  "max_tokens": 32000,
  "stream": true,
  "system": [
    {"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}
  ],
  "messages": [
    {"role": "user", "content": [{"type": "text", "text": "..."}]}
  ],
  "tools": [
    {"name": "Task", "description": "...", "input_schema": {...}}
  ],
  "thinking": {"budget_tokens": 31999, "type": "enabled"},
  "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
  "metadata": {"user_id": "..."}
}
```

**Key observations:**
- `system` is array of text blocks (not string)
- `cache_control` is `{"type": "ephemeral"}` - no ttl field observed
- `thinking` has `budget_tokens` and `type` fields
- `context_management` has `edits` array with type-specific edit objects

## Observed Response Structure

```json
{
  "model": "claude-haiku-4-5-20251001",
  "id": "msg_01GSLf1wRbGx4zpF2X69BjZG",
  "type": "message",
  "role": "assistant",
  "content": [{"type": "text", "text": "..."}],
  "stop_reason": "max_tokens",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 8,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
    "output_tokens": 1,
    "service_tier": "standard"
  },
  "context_management": {"applied_edits": []}
}
```

**Key observations:**
- `usage.cache_creation` has nested `ephemeral_5m_input_tokens` and `ephemeral_1h_input_tokens`
- `context_management.applied_edits` in response (vs `edits` in request)
- Response uses streaming (SSE) for main requests; structure above from non-streaming quota check

## Current Validation Status

### Validated Against Traffic (2026-01-02/03)

**Request/Response (request.py, response.py):**
- `MessagesRequest` - Full request structure ✓
- `MessagesResponse` - Non-streaming response ✓
- `CacheControl`, `SystemBlock`, `ToolDefinition` - Request components ✓
- `ThinkingConfig`, `ContextManagement` - Request variants ✓

**SSE Streaming (streaming.py) - 224 events validated:**
- `MessageStartEvent`, `MessageStopEvent` - Lifecycle ✓
- `ContentBlockStartEvent`, `ContentBlockStopEvent` - Block lifecycle ✓
- `ContentBlockDeltaEvent` with `TextDelta` - Text streaming ✓
- `MessageDeltaEvent` - Final usage/stop_reason ✓
- `PingEvent` - Keepalive ✓
- `ThinkingDelta`, `SignatureDelta`, `InputJsonDelta` - INFERRED (not observed)

**Rate Limits (rate_limits.py) - 6/6 responses:**
- `UnifiedRateLimit` with dual window (5h/7d) ✓
- `from_headers()` extraction method ✓

**Telemetry (telemetry.py) - 99 events validated:**
- `TelemetryBatchRequest`, `TelemetryEvent` - Batch structure ✓
- `TelemetryEventData`, `TelemetryEnv` - Event payload ✓
- 45+ event names documented ✓

**Internal Endpoints (internal_endpoints.py) - 17 validated:**
- `CountTokensResponse` - Token counting ✓
- `GroveResponse` - Feature gating ✓
- `MetricsEnabledResponse` - Telemetry opt-in ✓
- `EvalRequest`, `EvalResponse` - Feature flags with experiments ✓
- `AccountSettingsResponse` - User preferences (~47 fields) ✓
- `ClientDataResponse` - Client config ✓
- `HelloResponse` - Health check (10 captures) ✓

### Inferred (Not Yet Observed)
- `thinking_delta` in SSE (extended thinking streaming)
- `signature_delta` in SSE (thinking signatures)
- `input_json_delta` in SSE (tool input streaming)
- `tool_use` content blocks in responses
- Error responses (4xx/5xx)

### Corresponds to Session Files
- `ApiUsage` ↔ `session.TokenUsage` (field names match)
- Response content blocks ↔ `session.MessageContent`

## See Also

- `schemas/session/` - Session JSONL schemas (validated against 297k+ records)
- `schemas/operations/` - Service operation result schemas
- `domain/` - Application-level models
- `CLAUDE.md` - Traffic capture setup, development guide