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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mitmproxy import addonmanager, connection, http

# ==============================================================================
# Configuration
# ==============================================================================

SCRIPT_DIR = Path(__file__).parent.parent
CAPTURES_DIR = SCRIPT_DIR / 'captures'
LOG_FILE = CAPTURES_DIR / 'traffic.log'


# ==============================================================================
# Thread-Safe Counter
# ==============================================================================


class ThreadSafeCounter:
    """Thread-safe counter for sequence numbers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {'n': 0, 'errors': 0, 'ws': 0}

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
    with open(LOG_FILE, 'a') as f:
        f.write(message)


def _safe_filename(host: str, path: str) -> str:
    """Convert host and path to safe filename component."""
    combined = f'{host}{path}'
    return combined.replace('/', '_').replace('?', '_').replace('=', '_').replace('.', '_')[:80]


def _headers_to_dict(headers: http.Headers) -> dict[str, str]:
    """Convert mitmproxy headers to dict, preserving all headers."""
    result: dict[str, str] = {}
    for key, value in headers.items():  # type: ignore[no-untyped-call]
        if key in result:
            result[key] = f'{result[key]}, {value}'  # HTTP standard for multiple
        else:
            result[key] = value
    return result


def _safe_decode(content: bytes) -> str:
    """Safely decode bytes with BOM handling."""
    # Strip UTF-8 BOM if present
    if content.startswith(b'\xef\xbb\xbf'):
        content = content[3:]

    for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode('utf-8', errors='replace')


def _save_json_atomic(filename: Path, data: dict[str, Any]) -> bool:
    """Write JSON atomically via temp file + rename."""
    try:
        fd, temp_path = tempfile.mkstemp(dir=filename.parent, prefix=f'.tmp_{filename.stem}_', suffix='.json')
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
        _log(f'ERROR: Failed to save {filename}: {e}\n')
        return False


# ==============================================================================
# SSE Parsing (WHATWG HTML § 9.2.5 compliant)
# ==============================================================================


def _parse_sse_events(content: bytes) -> list[dict[str, Any]]:
    """
    Parse Server-Sent Events into structured data.

    Follows WHATWG HTML § 9.2.5:
    - Handles CR, LF, CRLF line endings
    - Preserves newlines between consecutive data: lines
    - Captures SSE comments
    """
    events: list[dict[str, Any]] = []
    text = _safe_decode(content)

    # Normalize all line endings to LF
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    current_event: dict[str, Any] = {}

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


def _parse_body(content: bytes | None, content_type: str) -> dict[str, Any]:
    """Parse body content, preserving all data."""
    if not content:
        return {'empty': True, 'size': 0}

    size = len(content)
    ct = content_type.lower()

    # SSE streaming
    if 'text/event-stream' in ct:
        return {
            'type': 'sse',
            'events': _parse_sse_events(content),
            'size': size,
        }

    # JSON
    if 'application/json' in ct:
        try:
            return {
                'type': 'json',
                'data': json.loads(content),
                'size': size,
            }
        except json.JSONDecodeError as e:
            return {
                'type': 'json',
                'parse_error': str(e),
                'raw': _safe_decode(content),
                'size': size,
            }

    # Form data
    if 'application/x-www-form-urlencoded' in ct:
        from urllib.parse import parse_qs

        return {
            'type': 'form',
            'data': parse_qs(_safe_decode(content)),
            'size': size,
        }

    # Binary
    if any(t in ct for t in ['image/', 'application/octet-stream', 'application/pdf', 'audio/', 'video/']):
        return {
            'type': 'binary',
            'content_type': content_type,
            'size': size,
            'sha256': hashlib.sha256(content).hexdigest(),
        }

    # Text/HTML and unknown
    return {
        'type': 'text',
        'content_type': content_type,
        'data': _safe_decode(content),
        'size': size,
    }


# ==============================================================================
# Connection Timing
# ==============================================================================


def _extract_connection_timing(
    conn: connection.Client | connection.Server,
) -> dict[str, float] | None:
    """Extract granular timing from connection object."""
    timing: dict[str, float] = {}
    for attr in ['timestamp_start', 'timestamp_tcp_setup', 'timestamp_tls_setup', 'timestamp_end']:
        if hasattr(conn, attr) and getattr(conn, attr):
            timing[attr.replace('timestamp_', '')] = getattr(conn, attr)
    return timing if timing else None


# ==============================================================================
# Lifecycle Hooks
# ==============================================================================


def load(loader: addonmanager.Loader) -> None:
    """Called when mitmproxy starts."""
    # Create captures directory
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize log file
    with open(LOG_FILE, 'w') as f:
        f.write(f'=== Capture started at {datetime.now(tz=UTC).isoformat()} ===\n')
        f.write(f'Output directory: {CAPTURES_DIR}\n\n')


def done() -> None:
    """Called when mitmproxy shuts down."""
    _log(f'\n=== Capture ended at {datetime.now(tz=UTC).isoformat()} ===\n')
    _log('Summary:\n')
    _log(f'  Requests: {COUNTER.get("n")}\n')
    _log(f'  Errors: {COUNTER.get("errors")}\n')
    _log(f'  WebSocket flows: {COUNTER.get("ws")}\n')


# ==============================================================================
# HTTP Capture
# ==============================================================================


def request(flow: http.HTTPFlow) -> None:
    """Capture complete request data."""
    n = COUNTER.increment('n')
    flow.metadata['capture_sequence'] = n

    headers = _headers_to_dict(flow.request.headers)
    content_type = headers.get('content-type', headers.get('Content-Type', ''))
    body = _parse_body(flow.request.content, content_type)

    capture: dict[str, Any] = {
        # Identification
        'flow_id': flow.id,
        'sequence': n,
        'direction': 'request',
        'is_replay': flow.is_replay,
        # Timing
        'timestamp': flow.request.timestamp_start,
        'timestamp_iso': datetime.fromtimestamp(flow.request.timestamp_start, tz=UTC).isoformat(),
        # URL
        'method': flow.request.method,
        'scheme': flow.request.scheme,
        'host': flow.request.host,
        'port': flow.request.port,
        'path': flow.request.path,
        'query': dict(flow.request.query),
        'url': flow.request.pretty_url,
        # Protocol
        'http_version': flow.request.http_version,
        # Headers & Cookies
        'headers': headers,
        'cookies': dict(flow.request.cookies),
        # Body
        'body': body,
    }

    # Client connection
    if flow.client_conn:
        capture['client_conn'] = {
            'id': str(flow.client_conn.id),
            'address': list(flow.client_conn.peername) if flow.client_conn.peername else None,
            'tls_version': getattr(flow.client_conn, 'tls_version', None),
            'timing': _extract_connection_timing(flow.client_conn),
        }

    # Save
    filename = CAPTURES_DIR / f'req_{n:03d}_{_safe_filename(flow.request.host, flow.request.path)}.json'
    _save_json_atomic(filename, capture)

    # Log
    _log(f'\n{"=" * 80}\n')
    _log(f'[{capture["timestamp_iso"]}] REQUEST #{n}\n')
    _log(f'  {flow.request.method} {flow.request.pretty_url}\n')
    _log(f'  Size: {body.get("size", 0)} bytes\n')
    _log(f'  Saved: {filename.name}\n')


def response(flow: http.HTTPFlow) -> None:
    """Capture complete response data."""
    n = flow.metadata.get('capture_sequence', COUNTER.get('n'))

    if not flow.response:
        _log(f'  RESPONSE #{n}: <no response>\n')
        return

    headers = _headers_to_dict(flow.response.headers)
    content_type = headers.get('content-type', headers.get('Content-Type', ''))
    body = _parse_body(flow.response.content, content_type)

    # Timing
    duration = None
    if flow.request.timestamp_start and flow.response.timestamp_end:
        duration = flow.response.timestamp_end - flow.request.timestamp_start

    capture: dict[str, Any] = {
        # Identification
        'flow_id': flow.id,
        'sequence': n,
        'direction': 'response',
        'is_replay': flow.is_replay,
        # Timing
        'timestamp': flow.response.timestamp_start,
        'timestamp_iso': datetime.fromtimestamp(flow.response.timestamp_start, tz=UTC).isoformat()
        if flow.response.timestamp_start
        else None,
        'duration_seconds': duration,
        # Status
        'status_code': flow.response.status_code,
        'reason': flow.response.reason,
        # Protocol
        'http_version': flow.response.http_version,
        # Headers & Cookies
        'headers': headers,
        'cookies': dict(flow.response.cookies),
        # Body
        'body': body,
    }

    # Server connection
    if flow.server_conn:
        capture['server_conn'] = {
            'id': str(flow.server_conn.id),
            'address': list(flow.server_conn.peername) if flow.server_conn.peername else None,
            'tls_established': flow.server_conn.tls_established,
            'tls_version': getattr(flow.server_conn, 'tls_version', None),
            'alpn': getattr(flow.server_conn, 'alpn_proto_negotiated', None),
            'sni': getattr(flow.server_conn, 'sni', None),
            'timing': _extract_connection_timing(flow.server_conn),
        }

    # Save
    filename = CAPTURES_DIR / f'resp_{n:03d}_{_safe_filename(flow.request.host, flow.request.path)}.json'
    _save_json_atomic(filename, capture)

    # Log
    _log(f'  RESPONSE #{n}: {flow.response.status_code} {flow.response.reason}\n')
    _log(f'  Size: {body.get("size", 0)} bytes\n')
    if duration:
        _log(f'  Duration: {duration:.3f}s\n')
    _log(f'  Saved: {filename.name}\n')


def error(flow: http.HTTPFlow) -> None:
    """Capture connection errors, timeouts, and failures."""
    n = COUNTER.increment('errors')

    error_msg = flow.error.msg if flow.error else 'unknown'

    capture: dict[str, Any] = {
        'sequence': n,
        'flow_id': flow.id,
        'timestamp_iso': datetime.now(tz=UTC).isoformat(),
        'direction': 'error',
    }

    if flow.error:
        capture['error'] = {
            'message': flow.error.msg,
            'timestamp': getattr(flow.error, 'timestamp', None),
        }

    if flow.request:
        capture['request'] = {
            'method': flow.request.method,
            'url': flow.request.pretty_url,
            'host': flow.request.host,
        }

    if flow.response:
        capture['response'] = {
            'status_code': flow.response.status_code,
            'reason': flow.response.reason,
        }

    filename = CAPTURES_DIR / f'error_{n:03d}_{flow.id[:8]}.json'
    _save_json_atomic(filename, capture)

    _log(f'\n{"!" * 80}\n')
    _log(f'ERROR #{n}: {error_msg}\n')
    if flow.request:
        _log(f'  URL: {flow.request.pretty_url}\n')
    _log(f'  Saved: {filename.name}\n')


# ==============================================================================
# WebSocket Support
# Note: Since mitmproxy 6+, WebSocket hooks receive http.HTTPFlow
# ==============================================================================


def websocket_start(flow: http.HTTPFlow) -> None:
    """Capture WebSocket connection initiation."""
    n = COUNTER.increment('ws')

    capture = {
        'sequence': n,
        'flow_id': flow.id,
        'direction': 'websocket_start',
        'timestamp_iso': datetime.fromtimestamp(flow.request.timestamp_start, tz=UTC).isoformat(),
        'url': flow.request.pretty_url,
        'headers': _headers_to_dict(flow.request.headers),
    }

    filename = CAPTURES_DIR / f'ws_start_{n:03d}_{_safe_filename(flow.request.host, flow.request.path)}.json'
    _save_json_atomic(filename, capture)
    _log(f'\nWEBSOCKET START #{n}: {flow.request.pretty_url}\n')


def websocket_message(flow: http.HTTPFlow) -> None:
    """Capture WebSocket messages."""
    if not flow.websocket or not flow.websocket.messages:
        return

    message = flow.websocket.messages[-1]
    n = len(flow.websocket.messages)

    capture = {
        'flow_id': flow.id,
        'message_index': n,
        'direction': 'websocket_message',
        'from_client': message.from_client,
        'timestamp': message.timestamp,
        'message_type': message.type.name if hasattr(message.type, 'name') else str(message.type),
        'content': message.text if hasattr(message, 'text') and message.is_text else None,
        'content_size': len(message.content) if message.content else 0,
        'content_hash': hashlib.sha256(message.content).hexdigest()
        if message.content and not message.is_text
        else None,
    }

    filename = CAPTURES_DIR / f'ws_msg_{flow.id[:8]}_{n:03d}.json'
    _save_json_atomic(filename, capture)


def websocket_end(flow: http.HTTPFlow) -> None:
    """Capture WebSocket connection closure."""
    message_count = len(flow.websocket.messages) if flow.websocket else 0
    capture = {
        'flow_id': flow.id,
        'direction': 'websocket_end',
        'message_count': message_count,
        'timestamp_iso': datetime.now(tz=UTC).isoformat(),
    }

    filename = CAPTURES_DIR / f'ws_end_{flow.id[:8]}.json'
    _save_json_atomic(filename, capture)
    _log(f'WEBSOCKET END: {message_count} messages\n')
