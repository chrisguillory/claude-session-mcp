# Handoff Document: Claude Code Internal API Schema Development

**Session ID:** 019b46e3-6bee-7f31-a3cf-4f4abed8ab32
**Date:** 2026-01-02
**Project:** /Users/chris/claude-session-mcp
**Status:** Traffic capture script ready, needs critical fixes before re-capture

---

## Executive Summary

This project reverse-engineers Claude Code CLI's internal API traffic to build validated Pydantic schemas. We use mitmproxy to capture all API traffic, then analyze the captured JSON files to build observation-based schemas.

**Key Principle:** CAPTURE everything at capture time, ANALYZE separately later. No filtering, no extraction, no analysis during capture.

**Current State:**
1. ✅ Capture script created (`scripts/intercept_traffic.py`)
2. ✅ Initial traffic captured (session e2e049ea) - 151 files
3. ✅ Comprehensive analysis completed via Perplexity Research
4. ⚠️ Script needs critical fixes (thread safety, atomic writes, lifecycle hooks)
5. ⏳ Re-capture needed with fixed script
6. ⏳ Schema building from captures

---

## Critical Fixes Needed for Capture Script

Based on two comprehensive Perplexity Research passes, these fixes are needed:

### 1. Thread-Safe Counter (CRITICAL)

**Problem:** mitmproxy can call hooks concurrently. The global `COUNTER` dict has race conditions.

**Current (broken):**
```python
COUNTER = {"n": 0, "errors": 0, "ws": 0}

def request(flow):
    COUNTER["n"] += 1  # RACE CONDITION
    n = COUNTER["n"]
```

**Fix:**
```python
import threading

class ThreadSafeCounter:
    def __init__(self):
        self._lock = threading.Lock()
        self._counters = {"n": 0, "errors": 0, "ws": 0}

    def increment(self, key: str) -> int:
        with self._lock:
            self._counters[key] += 1
            return self._counters[key]

    def get(self, key: str) -> int:
        with self._lock:
            return self._counters[key]

COUNTER = ThreadSafeCounter()
```

### 2. Atomic File Writes (CRITICAL)

**Problem:** If mitmproxy crashes mid-write, JSON files could be corrupted/truncated.

**Current (broken):**
```python
with open(filename, "w") as f:
    json.dump(capture, f, indent=2, default=str)
```

**Fix:**
```python
import tempfile
import os

def _save_json_atomic(filename: Path, data: dict) -> bool:
    """Write JSON atomically via temp file + rename."""
    try:
        # Create temp file in same directory (ensures same filesystem)
        fd, temp_path = tempfile.mkstemp(
            dir=filename.parent,
            prefix=f".tmp_{filename.stem}_",
            suffix=".json"
        )
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())  # Force to disk
        except Exception:
            os.close(fd)
            os.unlink(temp_path)
            raise

        # Atomic rename
        os.replace(temp_path, filename)
        return True
    except Exception as e:
        _log(f"Failed to save {filename}: {e}\n")
        return False
```

### 3. Lifecycle Hooks (CRITICAL)

**Problem:** No initialization (directory creation) or cleanup (flush logs).

**Fix:** Add `load()` and `done()` hooks:
```python
def load(loader):
    """Called when mitmproxy starts."""
    # Create captures directory
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize log file
    with open(LOG_FILE, "w") as f:
        f.write(f"=== Capture started at {datetime.now().isoformat()} ===\n")

def done():
    """Called when mitmproxy shuts down."""
    with open(LOG_FILE, "a") as f:
        f.write(f"\n=== Capture ended at {datetime.now().isoformat()} ===\n")
        f.write(f"Summary: {COUNTER.get('n')} requests, {COUNTER.get('errors')} errors, {COUNTER.get('ws')} websocket flows\n")
```

### 4. NOT Needed (Perplexity over-cautious)

- **Circular ref handling:** API JSON can't have circular refs by definition
- **requestheaders/responseheaders hooks:** Fire before body - we capture after full body
- **tls_clienthello hook:** Security research focused, we already capture TLS info from server_conn
- **Queue-based centralized logging:** Over-engineering for our use case

---

## The Complete Fixed Capture Script

Here is the COMPLETE script with all fixes applied:

```python
#!/usr/bin/env python3
"""
Intercept Claude Code API traffic via mitmproxy.

PURE CAPTURE - saves all traffic data to JSON files for later analysis.
No filtering, no extraction, no analysis at capture time.

Based on comprehensive mitmproxy research (Perplexity 2026-01-02).

Usage:
    # Kill any existing proxy
    pkill -f mitmdump 2>/dev/null

    # Clear old captures
    rm -f captures/*.json captures/traffic.log

    # Start the proxy (from repo root)
    mitmdump -p 8080 -s scripts/intercept_traffic.py --set stream=false

    # In another terminal, run Claude through the proxy
    HTTPS_PROXY=http://localhost:8080 NODE_TLS_REJECT_UNAUTHORIZED=0 claude

Output files:
    req_NNN_HOST_PATH.json   - Complete request data
    resp_NNN_HOST_PATH.json  - Complete response data
    error_NNN.json           - Connection failures
    ws_*.json                - WebSocket traffic (if any)
    traffic.log              - Summary log
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path

from mitmproxy import http

# ==============================================================================
# Configuration
# ==============================================================================

SCRIPT_DIR = Path(__file__).parent.parent
CAPTURES_DIR = SCRIPT_DIR / "captures"
LOG_FILE = CAPTURES_DIR / "traffic.log"


# ==============================================================================
# Thread-Safe Counter
# ==============================================================================

class ThreadSafeCounter:
    """Thread-safe counter for sequence numbers."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counters = {"n": 0, "errors": 0, "ws": 0}

    def increment(self, key: str) -> int:
        """Atomically increment and return new value."""
        with self._lock:
            self._counters[key] += 1
            return self._counters[key]

    def get(self, key: str) -> int:
        """Get current value."""
        with self._lock:
            return self._counters[key]


COUNTER = ThreadSafeCounter()


# ==============================================================================
# Utility Functions
# ==============================================================================

def _log(message: str) -> None:
    """Append to log file (thread-safe via OS-level append)."""
    with open(LOG_FILE, "a") as f:
        f.write(message)


def _safe_filename(host: str, path: str) -> str:
    """Convert host and path to safe filename component."""
    combined = f"{host}{path}"
    return combined.replace("/", "_").replace("?", "_").replace("=", "_").replace(".", "_")[:80]


def _headers_to_dict(headers) -> dict[str, str]:
    """Convert mitmproxy headers to dict, preserving all headers."""
    result = {}
    for key, value in headers.items():
        if key in result:
            result[key] = f"{result[key]}, {value}"  # HTTP standard for multiple
        else:
            result[key] = value
    return result


def _safe_decode(content: bytes) -> str:
    """Safely decode bytes with BOM handling."""
    # Strip UTF-8 BOM if present
    if content.startswith(b'\xef\xbb\xbf'):
        content = content[3:]

    for encoding in ["utf-8", "latin-1", "iso-8859-1"]:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _save_json_atomic(filename: Path, data: dict) -> bool:
    """Write JSON atomically via temp file + rename."""
    try:
        fd, temp_path = tempfile.mkstemp(
            dir=filename.parent,
            prefix=f".tmp_{filename.stem}_",
            suffix=".json"
        )
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            os.close(fd)
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

        os.replace(temp_path, filename)
        return True
    except Exception as e:
        _log(f"ERROR: Failed to save {filename}: {e}\n")
        return False


# ==============================================================================
# SSE Parsing (WHATWG HTML § 9.2.5 compliant)
# ==============================================================================

def _parse_sse_events(content: bytes) -> list[dict]:
    """
    Parse Server-Sent Events into structured data.

    Follows WHATWG HTML § 9.2.5:
    - Handles CR, LF, CRLF line endings
    - Preserves newlines between consecutive data: lines
    - Captures SSE comments
    """
    events = []
    text = _safe_decode(content)

    # Normalize all line endings to LF
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    current_event: dict = {}

    for line in text.split('\n'):
        if line.startswith(':'):  # Comment
            current_event.setdefault('comments', []).append(line[1:])
        elif line.startswith('event:'):
            current_event['type'] = line[6:].strip()
        elif line.startswith('data:'):
            value = line[5:]
            if value.startswith(' '):
                value = value[1:]  # Strip single leading space per spec
            current_event.setdefault('data', []).append(value)
        elif line.startswith('id:'):
            current_event['id'] = line[3:].strip()
        elif line.startswith('retry:'):
            current_event['retry'] = line[6:].strip()
        elif line == '':  # Blank line terminates event
            if current_event:
                if 'data' in current_event:
                    joined = '\n'.join(current_event['data'])  # Preserve newlines
                    try:
                        current_event['parsed_data'] = json.loads(joined)
                    except json.JSONDecodeError:
                        current_event['raw_data'] = joined
                events.append(current_event)
                current_event = {}

    # Handle last event if no trailing blank line
    if current_event:
        if 'data' in current_event:
            joined = '\n'.join(current_event['data'])
            try:
                current_event['parsed_data'] = json.loads(joined)
            except json.JSONDecodeError:
                current_event['raw_data'] = joined
        events.append(current_event)

    return events


# ==============================================================================
# Body Parsing
# ==============================================================================

def _parse_body(content: bytes | None, content_type: str) -> dict:
    """Parse body content, preserving all data."""
    if not content:
        return {"empty": True, "size": 0}

    size = len(content)
    ct = content_type.lower()

    # SSE streaming
    if "text/event-stream" in ct:
        return {
            "type": "sse",
            "events": _parse_sse_events(content),
            "size": size,
        }

    # JSON
    if "application/json" in ct:
        try:
            return {
                "type": "json",
                "data": json.loads(content),
                "size": size,
            }
        except json.JSONDecodeError as e:
            return {
                "type": "json",
                "parse_error": str(e),
                "raw": _safe_decode(content),
                "size": size,
            }

    # Form data
    if "application/x-www-form-urlencoded" in ct:
        from urllib.parse import parse_qs
        return {
            "type": "form",
            "data": parse_qs(_safe_decode(content)),
            "size": size,
        }

    # Binary
    if any(t in ct for t in ["image/", "application/octet-stream", "application/pdf", "audio/", "video/"]):
        return {
            "type": "binary",
            "content_type": content_type,
            "size": size,
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    # Text/HTML and unknown
    return {
        "type": "text",
        "content_type": content_type,
        "data": _safe_decode(content),
        "size": size,
    }


# ==============================================================================
# Connection Timing
# ==============================================================================

def _extract_connection_timing(conn) -> dict | None:
    """Extract granular timing from connection object."""
    timing = {}
    for attr in ['timestamp_start', 'timestamp_tcp_setup', 'timestamp_tls_setup', 'timestamp_end']:
        if hasattr(conn, attr) and getattr(conn, attr):
            timing[attr.replace('timestamp_', '')] = getattr(conn, attr)
    return timing if timing else None


# ==============================================================================
# Lifecycle Hooks
# ==============================================================================

def load(loader):
    """Called when mitmproxy starts."""
    # Create captures directory
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize log file
    with open(LOG_FILE, "w") as f:
        f.write(f"=== Capture started at {datetime.now().isoformat()} ===\n")
        f.write(f"Output directory: {CAPTURES_DIR}\n\n")


def done():
    """Called when mitmproxy shuts down."""
    _log(f"\n=== Capture ended at {datetime.now().isoformat()} ===\n")
    _log(f"Summary:\n")
    _log(f"  Requests: {COUNTER.get('n')}\n")
    _log(f"  Errors: {COUNTER.get('errors')}\n")
    _log(f"  WebSocket flows: {COUNTER.get('ws')}\n")


# ==============================================================================
# HTTP Capture
# ==============================================================================

def request(flow: http.HTTPFlow) -> None:
    """Capture complete request data."""
    n = COUNTER.increment("n")
    flow.metadata['capture_sequence'] = n

    headers = _headers_to_dict(flow.request.headers)
    content_type = headers.get("content-type", headers.get("Content-Type", ""))

    capture = {
        # Identification
        "flow_id": flow.id,
        "sequence": n,
        "direction": "request",
        "is_replay": flow.is_replay,
        # Timing
        "timestamp": flow.request.timestamp_start,
        "timestamp_iso": datetime.fromtimestamp(flow.request.timestamp_start).isoformat(),
        # URL
        "method": flow.request.method,
        "scheme": flow.request.scheme,
        "host": flow.request.host,
        "port": flow.request.port,
        "path": flow.request.path,
        "query": dict(flow.request.query),
        "url": flow.request.pretty_url,
        # Protocol
        "http_version": flow.request.http_version,
        # Headers & Cookies
        "headers": headers,
        "cookies": dict(flow.request.cookies),
        # Body
        "body": _parse_body(flow.request.content, content_type),
    }

    # Client connection
    if flow.client_conn:
        capture["client_conn"] = {
            "id": str(flow.client_conn.id),
            "address": list(flow.client_conn.peername) if flow.client_conn.peername else None,
            "tls_version": getattr(flow.client_conn, "tls_version", None),
            "timing": _extract_connection_timing(flow.client_conn),
        }

    # Save
    filename = CAPTURES_DIR / f"req_{n:03d}_{_safe_filename(flow.request.host, flow.request.path)}.json"
    _save_json_atomic(filename, capture)

    # Log
    _log(f"\n{'=' * 80}\n")
    _log(f"[{capture['timestamp_iso']}] REQUEST #{n}\n")
    _log(f"  {flow.request.method} {flow.request.pretty_url}\n")
    _log(f"  Size: {capture['body'].get('size', 0)} bytes\n")
    _log(f"  Saved: {filename.name}\n")


def response(flow: http.HTTPFlow) -> None:
    """Capture complete response data."""
    n = flow.metadata.get('capture_sequence', COUNTER.get("n"))

    if not flow.response:
        _log(f"  RESPONSE #{n}: <no response>\n")
        return

    headers = _headers_to_dict(flow.response.headers)
    content_type = headers.get("content-type", headers.get("Content-Type", ""))

    # Timing
    duration = None
    if flow.request.timestamp_start and flow.response.timestamp_end:
        duration = flow.response.timestamp_end - flow.request.timestamp_start

    capture = {
        # Identification
        "flow_id": flow.id,
        "sequence": n,
        "direction": "response",
        "is_replay": flow.is_replay,
        # Timing
        "timestamp": flow.response.timestamp_start,
        "timestamp_iso": datetime.fromtimestamp(flow.response.timestamp_start).isoformat() if flow.response.timestamp_start else None,
        "duration_seconds": duration,
        # Status
        "status_code": flow.response.status_code,
        "reason": flow.response.reason,
        # Protocol
        "http_version": flow.response.http_version,
        # Headers & Cookies
        "headers": headers,
        "cookies": dict(flow.response.cookies),
        # Body
        "body": _parse_body(flow.response.content, content_type),
    }

    # Server connection
    if flow.server_conn:
        capture["server_conn"] = {
            "id": str(flow.server_conn.id),
            "address": list(flow.server_conn.peername) if flow.server_conn.peername else None,
            "tls_established": flow.server_conn.tls_established,
            "tls_version": getattr(flow.server_conn, "tls_version", None),
            "alpn": getattr(flow.server_conn, "alpn_proto_negotiated", None),
            "sni": getattr(flow.server_conn, "sni", None),
            "timing": _extract_connection_timing(flow.server_conn),
        }

    # Save
    filename = CAPTURES_DIR / f"resp_{n:03d}_{_safe_filename(flow.request.host, flow.request.path)}.json"
    _save_json_atomic(filename, capture)

    # Log
    _log(f"  RESPONSE #{n}: {flow.response.status_code} {flow.response.reason}\n")
    _log(f"  Size: {capture['body'].get('size', 0)} bytes\n")
    if duration:
        _log(f"  Duration: {duration:.3f}s\n")
    _log(f"  Saved: {filename.name}\n")


def error(flow: http.HTTPFlow) -> None:
    """Capture connection errors, timeouts, and failures."""
    n = COUNTER.increment("errors")

    capture = {
        "sequence": n,
        "flow_id": flow.id,
        "timestamp_iso": datetime.now().isoformat(),
        "direction": "error",
    }

    if flow.error:
        capture["error"] = {
            "message": flow.error.msg,
            "timestamp": getattr(flow.error, "timestamp", None),
        }

    if flow.request:
        capture["request"] = {
            "method": flow.request.method,
            "url": flow.request.pretty_url,
            "host": flow.request.host,
        }

    if flow.response:
        capture["response"] = {
            "status_code": flow.response.status_code,
            "reason": flow.response.reason,
        }

    filename = CAPTURES_DIR / f"error_{n:03d}_{flow.id[:8]}.json"
    _save_json_atomic(filename, capture)

    _log(f"\n{'!' * 80}\n")
    _log(f"ERROR #{n}: {capture.get('error', {}).get('message', 'unknown')}\n")
    if flow.request:
        _log(f"  URL: {flow.request.pretty_url}\n")
    _log(f"  Saved: {filename.name}\n")


# ==============================================================================
# WebSocket Support
# ==============================================================================

try:
    from mitmproxy import websocket

    def websocket_start(flow: websocket.WebSocketFlow) -> None:
        """Capture WebSocket connection initiation."""
        n = COUNTER.increment("ws")

        capture = {
            "sequence": n,
            "flow_id": flow.id,
            "direction": "websocket_start",
            "timestamp_iso": datetime.fromtimestamp(flow.request.timestamp_start).isoformat(),
            "url": flow.request.pretty_url,
            "headers": _headers_to_dict(flow.request.headers),
        }

        filename = CAPTURES_DIR / f"ws_start_{n:03d}_{_safe_filename(flow.request.host, flow.request.path)}.json"
        _save_json_atomic(filename, capture)
        _log(f"\nWEBSOCKET START #{n}: {flow.request.pretty_url}\n")

    def websocket_message(flow: websocket.WebSocketFlow) -> None:
        """Capture WebSocket messages."""
        if not flow.messages:
            return

        message = flow.messages[-1]
        n = len(flow.messages)

        capture = {
            "flow_id": flow.id,
            "message_index": n,
            "direction": "websocket_message",
            "from_client": message.from_client,
            "timestamp": message.timestamp,
            "message_type": message.type.name if hasattr(message.type, 'name') else str(message.type),
            "content": message.text if hasattr(message, 'text') and message.is_text else None,
            "content_size": len(message.content) if message.content else 0,
            "content_hash": hashlib.sha256(message.content).hexdigest() if message.content and not message.is_text else None,
        }

        filename = CAPTURES_DIR / f"ws_msg_{flow.id[:8]}_{n:03d}.json"
        _save_json_atomic(filename, capture)

    def websocket_end(flow: websocket.WebSocketFlow) -> None:
        """Capture WebSocket connection closure."""
        capture = {
            "flow_id": flow.id,
            "direction": "websocket_end",
            "message_count": len(flow.messages),
            "timestamp_iso": datetime.now().isoformat(),
        }

        filename = CAPTURES_DIR / f"ws_end_{flow.id[:8]}.json"
        _save_json_atomic(filename, capture)
        _log(f"WEBSOCKET END: {len(flow.messages)} messages\n")

except ImportError:
    pass
```

---

## Traffic Capture Procedure

### Setup

```bash
# Ensure mitmproxy is installed
brew install mitmproxy

# From the project root
cd /Users/chris/claude-session-mcp
```

### Capture

```bash
# 1. Kill any existing proxy
pkill -f mitmdump 2>/dev/null

# 2. Clear old captures
rm -f captures/*.json captures/traffic.log

# 3. Start proxy (in foreground to see any errors)
mitmdump -p 8080 -s scripts/intercept_traffic.py --set stream=false

# 4. In another terminal, run Claude through proxy
HTTPS_PROXY=http://localhost:8080 NODE_TLS_REJECT_UNAUTHORIZED=0 claude

# 5. Interact with Claude (e.g., say "Hello!")

# 6. Exit Claude, then Ctrl+C the proxy
```

### Verify

```bash
# Check files were created
ls -la captures/*.json | head -20

# Check log
cat captures/traffic.log

# Check a request
cat captures/req_001_*.json | jq '{flow_id, method, url, body_type: .body.type}'

# Check a response
cat captures/resp_001_*.json | jq '{flow_id, status_code, body_type: .body.type}'
```

---

## Observations from Previous Capture (Session e2e049ea)

### Endpoints Discovered

| Endpoint | Count | Purpose |
|----------|-------|---------|
| `/v1/messages?beta=true` | 7 | Main LLM API |
| `/v1/messages/count_tokens?beta=true` | 1 | Token counting |
| `/api/hello` | 10 | Health check (~30s) |
| `/api/event_logging/batch` | 6 | Telemetry |
| `/api/oauth/claude_cli/client_data` | 1 | OAuth config |
| `/api/oauth/account/settings` | 1 | Account config |
| `/api/claude_code/organizations/metrics_enabled` | 1 | Org metrics |
| `/api/claude_code_grove` | 1 | Grove feature |
| `/api/eval/sdk-*` | 1 | A/B testing |
| `statsig.anthropic.com/v1/initialize` | 2 | Feature flags init |
| `statsig.anthropic.com/v1/rgstr` | 74 | Feature flag events |

### Key Headers Observed

**Request:**
```
anthropic-version: 2023-06-01
anthropic-beta: oauth-2025-04-20,interleaved-thinking-2025-05-14,context-management-2025-06-27
anthropic-dangerous-direct-browser-access: true
authorization: Bearer sk-ant-oat01-... (108 chars, opaque)
user-agent: claude-cli/2.0.76 (external, cli)
x-stainless-arch: arm64
x-stainless-lang: js
x-stainless-os: MacOS
x-stainless-package-version: 0.70.0
x-stainless-runtime: node
x-stainless-runtime-version: v24.3.0
```

**Response (rate limits):**
```
anthropic-ratelimit-unified-status: allowed
anthropic-ratelimit-unified-5h-status: allowed
anthropic-ratelimit-unified-5h-reset: 1767409200
anthropic-ratelimit-unified-5h-utilization: 0.43
anthropic-ratelimit-unified-7d-status: allowed
anthropic-ratelimit-unified-7d-reset: 1767888000
anthropic-ratelimit-unified-7d-utilization: 0.09
anthropic-ratelimit-unified-representative-claim: five_hour
anthropic-ratelimit-unified-fallback-percentage: 0.5
anthropic-ratelimit-unified-overage-disabled-reason: org_level_disabled
anthropic-ratelimit-unified-overage-status: rejected
anthropic-organization-id: 1e4ddbbc-a180-41a9-87d3-d7dcd4a204ab
request-id: req_011CWjbphJUo1ZvuHNdPNi2c
```

### SSE Event Types

```
message_start      - Initial message with metadata
content_block_start - Start of content block
content_block_delta - Streaming content chunks
content_block_stop  - End of content block
message_delta      - Final usage/stop_reason
message_stop       - End of message
ping               - Keep-alive
```

### Request Body Structure (/v1/messages)

```json
{
  "model": "claude-opus-4-5-20251101",
  "max_tokens": 16000,
  "stream": true,
  "system": [
    {"type": "text", "text": "You are Claude Code..."}
  ],
  "messages": [
    {"role": "user", "content": [...]}
  ],
  "tools": [
    {"name": "Task", "description": "...", "input_schema": {...}}
  ],
  "thinking": {
    "budget_tokens": 31999,
    "type": "enabled"
  },
  "context_management": {
    "edits": [
      {"type": "clear_thinking_20251015", "keep": "all"}
    ]
  },
  "metadata": {
    "user_id": "user_{hash}_account_{uuid}_session_{uuid}"
  }
}
```

### Response Body Structure

**Non-streaming:**
```json
{
  "model": "claude-haiku-4-5-20251001",
  "id": "msg_01...",
  "type": "message",
  "role": "assistant",
  "content": [{"type": "text", "text": "..."}],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 119,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation": {
      "ephemeral_5m_input_tokens": 0,
      "ephemeral_1h_input_tokens": 0
    },
    "output_tokens": 21,
    "service_tier": "standard"
  },
  "context_management": {
    "applied_edits": []
  }
}
```

**Streaming (SSE events):**
- `message_start` has initial usage
- `content_block_delta` has incremental text
- `message_delta` has final usage and stop_reason

---

## Schema Building Approach

### Location

Schemas go in: `src/schemas/cc_internal_api/`

### Files Already Created

- `base.py` - PermissiveModel (extra='ignore'), FromSession/FromSdk markers
- `common.py` - CacheControl, ApiUsage
- `request.py` - MessagesRequest, SystemBlock, ToolDefinition, etc.
- `response.py` - MessagesResponse, ResponseContent, StopReason
- `__init__.py` - Re-exports
- `README.md` - Documentation

### Key Patterns

1. **Use PermissiveModel** - Allows unknown fields during observation
2. **FromSession markers** - Document correspondence with session schemas
3. **FromSdk markers** - Document correspondence with Anthropic SDK types
4. **Validation status** - Document VALIDATED/INFERRED/REFERENCE per field

### Example Schema

```python
class MessagesResponse(PermissiveModel):
    """
    Response from /v1/messages.

    VALIDATION STATUS: VALIDATED (2026-01-02)
    """

    model: Annotated[
        ModelId,
        FromSession(session.models.Message, 'model', status='validated'),
        FromSdk(anthropic.types.Message, 'model'),
    ]

    id: str
    type: Literal['message']
    role: Literal['assistant']
    content: list[ResponseContent]
    stop_reason: StopReason | None
    stop_sequence: str | None
    usage: ApiUsage
    context_management: ResponseContextManagement | None = None
```

---

## Current File Structure

```
/Users/chris/claude-session-mcp/
├── scripts/
│   └── intercept_traffic.py     # Capture script (NEEDS FIXES)
├── captures/                     # Traffic captures (gitignored)
│   ├── req_*.json
│   ├── resp_*.json
│   └── traffic.log
├── src/schemas/
│   ├── cc_internal_api/         # API schemas
│   │   ├── __init__.py
│   │   ├── base.py              # PermissiveModel, markers
│   │   ├── common.py            # CacheControl, ApiUsage
│   │   ├── request.py           # Request schemas
│   │   ├── response.py          # Response schemas
│   │   └── README.md            # Documentation
│   ├── session/                  # Session JSONL schemas
│   └── types.py                  # ModelId, JsonDatetime
├── CLAUDE.md                     # Development guide
└── docs/
    └── handoff-api-schema-capture.md  # This file
```

---

## Immediate Next Steps

1. **Apply fixes to intercept_traffic.py**
   - Copy the complete fixed script from this document
   - Or apply the individual fixes (thread safety, atomic writes, lifecycle)

2. **Re-capture traffic**
   ```bash
   pkill -f mitmdump
   rm -f captures/*.json captures/traffic.log
   mitmdump -p 8080 -s scripts/intercept_traffic.py --set stream=false
   # In another terminal:
   HTTPS_PROXY=http://localhost:8080 NODE_TLS_REJECT_UNAUTHORIZED=0 claude
   # Say "Hello!" and exit
   ```

3. **Analyze captures**
   - Count files: `ls captures/*.json | wc -l`
   - Check endpoints: `cat captures/req_*.json | jq -r '.url' | sort | uniq -c | sort -rn`
   - Check response codes: `cat captures/resp_*.json | jq -r '.status_code' | sort | uniq -c`

4. **Update schemas from observed data**
   - Compare captures to existing schemas
   - Add any new fields/types
   - Update validation status comments

5. **Run validation and commit**
   ```bash
   uv run mypy src/ scripts/
   uv run ruff check src/ scripts/
   uv run pre-commit run --all-files
   git add -A
   git commit -m "Add observation-based API schemas for Claude Code internal API"
   ```

---

## Key Design Decisions

1. **CAPTURE vs ANALYZE separation** - Script only captures, analysis is separate
2. **No filtering** - Capture EVERYTHING, filter during analysis
3. **PermissiveModel** - Use `extra='ignore'` to allow unknown fields
4. **Atomic writes** - Prevent corrupted files on crash
5. **Thread safety** - Handle concurrent mitmproxy hooks
6. **Lifecycle hooks** - Proper init and cleanup

---

## User Preferences (from this session)

1. No exception swallowing - let errors surface
2. No premature optimization - capture everything
3. Observation-based - build from captured data, not docs
4. Minimal __init__.py - just re-exports
5. Absolute imports - use `session.models.TokenUsage` not aliases
6. Use Literals strictly - `ModelId` not `str`
7. Run pre-commit before committing

---

## Related Documentation

- `CLAUDE.md` - Full development guide with traffic capture section
- `src/schemas/cc_internal_api/README.md` - API schema documentation
- `src/schemas/session/README.md` - Session schema documentation
- Plan file: `/Users/chris/.claude/plans/imperative-beaming-lecun.md`

---

## Summary

This handoff document provides everything needed to continue the API schema development:

1. **The problem** - Need validated schemas for Claude Code's internal API
2. **The approach** - Capture traffic via mitmproxy, build schemas from observation
3. **Current state** - Script created but needs critical fixes
4. **The fixes** - Thread safety, atomic writes, lifecycle hooks
5. **The captures** - 151 files from previous session with rich observations
6. **Next steps** - Apply fixes, re-capture, update schemas, commit

The next AI should:
1. Apply the fixes to `scripts/intercept_traffic.py`
2. Re-capture traffic with the fixed script
3. Analyze the new captures
4. Update schemas based on observations
5. Run validation and commit