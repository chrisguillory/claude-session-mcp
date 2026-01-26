#!/usr/bin/env python3
"""
Intercept Claude Code API traffic via mitmproxy.

PURE CAPTURE - saves all traffic data to JSON files for later analysis.
No filtering, no extraction, no analysis at capture time.

================================================================================
SESSION CORRELATION ARCHITECTURE
================================================================================

This script integrates with claude-workspace's hook system to definitively
associate HTTP traffic with Claude Code sessions.

HOW IT WORKS:

    ┌─────────────────────────────────────────────────────────────────────┐
    │ 1. User runs: HTTPS_PROXY=http://localhost:8080 claude              │
    │                                                                     │
    │ 2. Claude Code starts, generates session_id internally              │
    │                                                                     │
    │ 3. Claude Code calls SessionStart hook (configured in settings)     │
    │    → Passes via stdin: { session_id, transcript_path, ... }         │
    │                                                                     │
    │ 4. Hook (claude-workspace/hooks/session-start.py):                  │
    │    a) Receives session_id directly from Claude Code                 │
    │    b) Walks process tree to find claude_pid                         │
    │    c) Writes to ~/.claude-workspace/sessions.json:                  │
    │       { session_id, state:"active", claude_pid, ... }               │
    │    d) Returns (~18ms) → Claude continues startup                    │
    │                                                                     │
    │ 5. Claude makes HTTP request → mitmproxy intercepts                 │
    │                                                                     │
    │ 6. client_connected hook fires:                                     │
    │    a) Get source_port from connection.peername                      │
    │    b) psutil maps source_port → PID                                 │
    │    c) Read sessions.json, find active session where claude_pid=PID  │
    │    d) Store session_id in connection.metadata                       │
    │                                                                     │
    │ 7. Create captures/<session_id>/ directory                          │
    │    All traffic from this connection → that directory                │
    └─────────────────────────────────────────────────────────────────────┘

WHY THIS IS DEFINITIVE:

    - session_id comes from Claude Code itself (not derived or guessed)
    - PID linkage established by hook walking actual process tree
    - psutil provides kernel-level PID-to-port mapping
    - sessions.json uses FileLock + atomic writes (no race conditions)
    - Hook completes before first HTTP request (timing is safe)

DEPENDENCIES:

    - claude-workspace hooks must be configured in ~/.claude/settings.json
    - ~/.claude-workspace/sessions.json must exist (created by hooks)
    - psutil for PID-to-port mapping

================================================================================

Usage:
    # Kill any existing proxy
    pkill -f mitmdump 2>/dev/null

    # Start the proxy (from repo root)
    mitmdump -p 8080 -s scripts/intercept_traffic.py --set stream=false

    # In another terminal, run Claude through the proxy
    HTTPS_PROXY=http://localhost:8080 NODE_TLS_REJECT_UNAUTHORIZED=0 claude

Output structure:
    captures/<session_id>/
        NNN_req_HOST_PATH.json   - Complete request data
        NNN_resp_HOST_PATH.json  - Complete response data
        manifest.json            - Session metadata
        traffic.log              - Summary log
    captures/unknown/            - Traffic without session correlation

    Note: Sequence-first naming (NNN_req/resp) keeps request/response pairs
    adjacent when sorted alphabetically.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
from expiringdict import ExpiringDict
from mitmproxy import addonmanager, connection, http
from pydantic import TypeAdapter

from src.schemas import claude_workspace

# ==============================================================================
# Configuration
# ==============================================================================

CAPTURE_VERSION = 2  # Increment when capture format changes

SCRIPT_DIR = Path(__file__).parent.parent
CAPTURES_BASE = SCRIPT_DIR / 'captures'
SESSIONS_PATH = Path.home() / '.claude-workspace' / 'sessions.json'
LOG_FILE = CAPTURES_BASE / 'traffic.log'

# Track sessions that have had manifests written (in-memory, resets on proxy restart)
_SESSIONS_SEEN: set[str] = set()
_SESSIONS_SEEN_LOCK = threading.Lock()

# Store session info per connection, keyed by connection ID (mitmproxy's Client.id)
# mitmproxy's Client object does NOT have a metadata attribute, so we use our own storage
_CONNECTION_SESSIONS: dict[str, claude_workspace.Session] = {}
_CONNECTION_SESSIONS_LOCK = threading.Lock()

# TypeAdapter for parsing sessions.json with Pydantic validation
_SESSION_DB_ADAPTER: TypeAdapter[claude_workspace.SessionDatabase] = TypeAdapter(claude_workspace.SessionDatabase)

# Cache for codesign verification results: exe_path -> is_claude_code (expires after 1 hour)
# We only need to verify each executable once per proxy lifetime
_CODESIGN_CACHE: ExpiringDict[str, bool] = ExpiringDict(max_len=100, max_age_seconds=3600)

# Track (pid, create_time) pairs we've already attempted retry for (expires after 2 min)
# This prevents repeated retry attempts for the same process
_RETRY_ATTEMPTED: ExpiringDict[tuple[int, float], bool] = ExpiringDict(max_len=1000, max_age_seconds=120)

# Retry configuration for session lookup timing race
_RETRY_MAX_ATTEMPTS = 10  # 10 attempts
_RETRY_DELAY_SECONDS = 0.3  # 300ms between attempts = 3s max total

# Session cache: maps PID → Session for active sessions
# Invalidated when sessions.json mtime changes (handles /clear correctly)
_SESSION_CACHE: dict[int, claude_workspace.Session] = {}
_SESSION_CACHE_MTIME: float = 0.0
_SESSION_CACHE_LOCK = threading.Lock()


# ==============================================================================
# Claude Code Process Verification
# ==============================================================================


def _is_claude_code_process(pid: int) -> bool:
    """Verify if a process is Claude Code using macOS codesign.

    Uses `codesign -dv` to check if the executable is signed by Anthropic.
    Results are cached by executable path to avoid repeated verification.

    Returns True if:
    - The process exists and we can get its executable path
    - The executable is signed with a Developer ID containing "Anthropic"

    Returns False if:
    - Process doesn't exist or we can't access it
    - Executable is not signed or not by Anthropic
    - codesign command fails for any reason
    """
    try:
        proc = psutil.Process(pid)
        exe_path = proc.exe()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False

    if not exe_path:
        return False

    # Check cache first
    if exe_path in _CODESIGN_CACHE:
        cached: bool = _CODESIGN_CACHE[exe_path]
        return cached

    # Run codesign to verify
    try:
        result = subprocess.run(
            ['codesign', '-dv', exe_path],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # codesign outputs to stderr, look for anthropic in the output (case-insensitive)
        # e.g., "Identifier=com.anthropic.claude-code"
        output = result.stderr.lower()
        is_claude = 'anthropic' in output
        _CODESIGN_CACHE[exe_path] = is_claude
        return is_claude
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
        # If codesign fails, assume not Claude Code
        _CODESIGN_CACHE[exe_path] = False
        return False


def _should_retry_session_lookup(pid: int) -> tuple[bool, tuple[int, float] | None]:
    """Check if we should retry session lookup for this PID.

    Returns (should_retry, key) where:
    - should_retry: True if this is a Claude Code process worth retrying
    - key: (pid, create_time) tuple to mark in _RETRY_ATTEMPTED when done

    Does NOT mark the key - caller must mark after retry completes.
    This allows multiple concurrent connections to all attempt retry,
    rather than only the first one.

    This prevents:
    - Wasting time retrying for non-Claude processes (curl, wget, etc.)
    - Repeated retry attempts after retry has already completed
    """
    # First check if it's Claude Code
    if not _is_claude_code_process(pid):
        return False, None

    # Get create_time for deduplication
    try:
        proc = psutil.Process(pid)
        create_time = proc.create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False, None

    key = (pid, create_time)

    # Check if retry already completed for this process
    if key in _RETRY_ATTEMPTED:
        return False, None

    # Don't mark here - caller marks when done
    return True, key


def _get_session_with_retry(pid: int) -> claude_workspace.Session | None:
    """Get session for PID with retry logic for timing race.

    If session not found immediately and process is Claude Code,
    retries for up to 3 seconds (10 attempts * 300ms).

    This handles the timing race where HTTP requests arrive before
    the SessionStart hook has written to sessions.json.

    Multiple concurrent connections from the same PID will all retry
    in parallel, allowing all of them to find the session when it appears.
    The key is only marked in _RETRY_ATTEMPTED after retry completes.
    """
    # Quick check first
    session = _get_session_for_pid(pid)
    if session:
        return session

    # Check if we should retry for this process
    should_retry, key = _should_retry_session_lookup(pid)
    if not should_retry or key is None:
        return None

    # Retry with backoff
    for attempt in range(_RETRY_MAX_ATTEMPTS):
        time.sleep(_RETRY_DELAY_SECONDS)
        session = _get_session_for_pid(pid)
        if session:
            _log(f'Session found after {attempt + 1} retries (PID {pid})\n')
            _RETRY_ATTEMPTED[key] = True  # Mark completed
            return session

    _log(f'Session not found after {_RETRY_MAX_ATTEMPTS} retries (PID {pid})\n')
    _RETRY_ATTEMPTED[key] = True  # Mark completed (not found)
    return None


# ==============================================================================
# Session Correlation
# ==============================================================================


def _get_pid_for_port(source_port: int) -> int | None:
    """Find which process owns a TCP connection from the given source port.

    Iterates through processes and checks their connections. This approach
    does NOT require elevated privileges because we only access processes
    where the effective UID matches ours.

    AccessDenied handling:
    - If effective UID matches ours and we get AccessDenied → raise (unexpected)
    - If effective UID differs (system/elevated processes) → skip (expected)

    ZombieProcess and NoSuchProcess are caught and skipped - these occur when
    processes terminate mid-iteration (observed in practice with CommCenter, bash).

    Returns None if port not found (connection may have closed or belongs
    to a process with elevated privileges).
    """
    my_uid = os.getuid()

    for proc in psutil.process_iter(['pid', 'name', 'uids']):
        proc_pid = proc.info['pid']
        proc_name = proc.info['name']
        proc_uids = proc.info['uids']

        # We can access processes where effective UID matches ours.
        # Setuid binaries (like /usr/bin/login) have effective UID = 0
        # even if we launched them, so we correctly skip those.
        should_be_accessible = proc_uids and proc_uids.effective == my_uid

        try:
            for conn in proc.net_connections():
                if conn.laddr and conn.laddr.port == source_port:
                    pid: int = proc_pid
                    return pid
        except psutil.AccessDenied:
            if should_be_accessible:
                # This is truly unexpected - we should be able to access this
                raise RuntimeError(
                    f'Unexpected AccessDenied for process with matching effective UID: '
                    f'PID {proc_pid} ({proc_name}) uids={proc_uids}'
                )
            # Expected - process has elevated/different effective UID, skip
        except psutil.ZombieProcess:
            # Zombie processes can't own active connections, skip
            pass
        except psutil.NoSuchProcess:
            # Process terminated mid-iteration, skip
            pass

    return None


def _get_session_for_pid(pid: int) -> claude_workspace.Session | None:
    """Find active Claude Code session for a given PID.

    Uses mtime-based caching: only re-reads sessions.json when it changes.
    This handles /clear correctly (file updated → cache invalidated).

    Thread-safe via _SESSION_CACHE_LOCK.

    Raises:
        FileNotFoundError: If sessions.json doesn't exist (hooks not configured)
        json.JSONDecodeError: If sessions.json is invalid JSON
        pydantic.ValidationError: If sessions.json doesn't match schema
    """
    global _SESSION_CACHE_MTIME, _SESSION_CACHE

    with _SESSION_CACHE_LOCK:
        # stat() inside lock to avoid TOCTOU race between threads
        try:
            current_mtime = SESSIONS_PATH.stat().st_mtime
        except FileNotFoundError:
            raise FileNotFoundError(
                f'Sessions file not found: {SESSIONS_PATH}\n'
                'Ensure claude-workspace hooks are configured in ~/.claude/settings.json'
            ) from None
        if current_mtime != _SESSION_CACHE_MTIME:
            # File changed, rebuild cache
            with open(SESSIONS_PATH) as f:
                data = json.load(f)

            # Validate with Pydantic - will raise ValidationError if schema doesn't match
            session_db = _SESSION_DB_ADAPTER.validate_python(data)

            # Build PID → Session mapping for active sessions only
            _SESSION_CACHE = {s.metadata.claude_pid: s for s in session_db.sessions if s.state == 'active'}
            _SESSION_CACHE_MTIME = current_mtime

        return _SESSION_CACHE.get(pid)


def _get_session_dir(session_id: str | None) -> Path:
    """Get or create the capture directory for a session."""
    if session_id:
        session_dir = CAPTURES_BASE / session_id
    else:
        session_dir = CAPTURES_BASE / 'unknown'

    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _write_manifest(session_dir: Path, session: claude_workspace.Session | None) -> None:
    """Write session manifest to capture directory."""
    manifest: dict[str, Any] = {
        'capture_version': CAPTURE_VERSION,
        'captured_at': datetime.now(tz=UTC).isoformat(),
    }

    if session:
        manifest['claude_session'] = {
            'session_id': session.session_id,
            'project_dir': session.project_dir,
            'transcript_path': session.transcript_path,
            'source': session.source,
            'claude_pid': session.metadata.claude_pid,
            'process_created_at': session.metadata.process_created_at.isoformat()
            if session.metadata.process_created_at
            else None,
        }

    manifest_path = session_dir / 'manifest.json'
    _save_json_atomic(manifest_path, manifest)


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
        fd_owned_by_file = False
        try:
            with os.fdopen(fd, 'w') as f:
                fd_owned_by_file = True  # fdopen took ownership, will close on exit
                json.dump(data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            if not fd_owned_by_file:
                os.close(fd)  # Only close if fdopen didn't take ownership
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
        events = _parse_sse_events(content)
        result: dict[str, Any] = {
            'type': 'sse',
            'events': events,
            'size': size,
        }
        # Concatenate text_delta events for readability
        text_deltas = [
            e.get('parsed_data', {}).get('delta', {}).get('text', '')
            for e in events
            if e.get('parsed_data', {}).get('delta', {}).get('type') == 'text_delta'
        ]
        if text_deltas:
            result['concatenated_text'] = ''.join(text_deltas)
        return result

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
    # Create base captures directory
    CAPTURES_BASE.mkdir(parents=True, exist_ok=True)

    # Initialize log file
    with open(LOG_FILE, 'w') as f:
        f.write(f'=== Capture started at {datetime.now(tz=UTC).isoformat()} ===\n')
        f.write(f'Output directory: {CAPTURES_BASE}\n\n')


def done() -> None:
    """Called when mitmproxy shuts down."""
    _log(f'\n=== Capture ended at {datetime.now(tz=UTC).isoformat()} ===\n')
    _log('Summary:\n')
    _log(f'  Requests: {COUNTER.get("n")}\n')
    _log(f'  Errors: {COUNTER.get("errors")}\n')
    _log(f'  WebSocket flows: {COUNTER.get("ws")}\n')


# ==============================================================================
# Session Discovery
# ==============================================================================


def _get_session_for_connection(client_id: str) -> claude_workspace.Session | None:
    """Get session for a connection ID (thread-safe)."""
    with _CONNECTION_SESSIONS_LOCK:
        return _CONNECTION_SESSIONS.get(client_id)


def client_connected(client: connection.Client) -> None:
    """Discover Claude Code session when client connects.

    This hook fires when a new TCP connection is established.
    We use psutil to map the client's source port to a PID,
    then look up the session in ~/.claude-workspace/sessions.json.

    For Claude Code processes, uses smart retry to handle the timing race
    where HTTP requests arrive before the SessionStart hook writes to sessions.json.

    Session info is stored in _CONNECTION_SESSIONS for use by request/response hooks.

    Exceptions (FileNotFoundError, JSONDecodeError, ValidationError) propagate to
    mitmproxy which logs them and continues - captures will have session_id: null.
    """
    if not client.peername:
        return

    source_port = client.peername[1]

    # Get PID for this connection
    pid = _get_pid_for_port(source_port)
    if pid is None:
        _log(f'WARNING: No PID found for port {source_port}\n')
        return

    # Look up session with smart retry for Claude Code processes
    # FileNotFoundError/JSONDecodeError/ValidationError propagate → mitmproxy logs, continues
    session = _get_session_with_retry(pid)
    if session:
        # Store session keyed by client.id for later retrieval
        with _CONNECTION_SESSIONS_LOCK:
            _CONNECTION_SESSIONS[client.id] = session
        _log(f'Session {session.session_id[:8]}... (PID {pid})\n')
    else:
        _log(f'WARNING: PID {pid} not in active sessions\n')


def client_disconnected(client: connection.Client) -> None:
    """Clean up session storage when client disconnects."""
    with _CONNECTION_SESSIONS_LOCK:
        _CONNECTION_SESSIONS.pop(client.id, None)


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

    # Get session from our connection storage (set by client_connected hook)
    session: claude_workspace.Session | None = None
    if flow.client_conn:
        session = _get_session_for_connection(flow.client_conn.id)
    session_id = session.session_id if session else None

    capture: dict[str, Any] = {
        # Identification
        'flow_id': flow.id,
        'sequence': n,
        'direction': 'request',
        'is_replay': flow.is_replay,
        'session_id': session_id,
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

    # Get session directory and write manifest on first capture (thread-safe)
    session_dir = _get_session_dir(session_id)
    if session_id:
        with _SESSIONS_SEEN_LOCK:
            if session_id not in _SESSIONS_SEEN:
                _write_manifest(session_dir, session)
                _SESSIONS_SEEN.add(session_id)

    # Save
    filename = session_dir / f'{n:03d}_req_{_safe_filename(flow.request.host, flow.request.path)}.json'
    _save_json_atomic(filename, capture)

    # Log
    _log(f'\n{"=" * 80}\n')
    _log(f'[{capture["timestamp_iso"]}] REQUEST #{n}\n')
    _log(f'  {flow.request.method} {flow.request.pretty_url}\n')
    _log(f'  Size: {body.get("size", 0)} bytes\n')
    _log(f'  Session: {session_id or "unknown"}\n')
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

    # Get session from our connection storage (set by client_connected hook)
    session: claude_workspace.Session | None = None
    if flow.client_conn:
        session = _get_session_for_connection(flow.client_conn.id)
    session_id = session.session_id if session else None

    capture: dict[str, Any] = {
        # Identification
        'flow_id': flow.id,
        'sequence': n,
        'direction': 'response',
        'is_replay': flow.is_replay,
        'session_id': session_id,
        # Request context (for correlation/discrimination)
        'method': flow.request.method,
        'scheme': flow.request.scheme,
        'host': flow.request.host,
        'port': flow.request.port,
        'path': flow.request.path,
        'query': dict(flow.request.query),
        'url': flow.request.pretty_url,
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

    # Get session directory (manifest already written in request())
    session_dir = _get_session_dir(session_id)

    # Save
    filename = session_dir / f'{n:03d}_resp_{_safe_filename(flow.request.host, flow.request.path)}.json'
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

    # Get session from our connection storage (set by client_connected hook)
    session: claude_workspace.Session | None = None
    if flow.client_conn:
        session = _get_session_for_connection(flow.client_conn.id)
    session_id = session.session_id if session else None

    capture: dict[str, Any] = {
        'sequence': n,
        'flow_id': flow.id,
        'timestamp_iso': datetime.now(tz=UTC).isoformat(),
        'direction': 'error',
        'session_id': session_id,
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

    session_dir = _get_session_dir(session_id)
    filename = session_dir / f'error_{n:03d}_{flow.id[:8]}.json'
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

    # Get session from our connection storage (set by client_connected hook)
    session: claude_workspace.Session | None = None
    if flow.client_conn:
        session = _get_session_for_connection(flow.client_conn.id)
    session_id = session.session_id if session else None

    capture = {
        'sequence': n,
        'flow_id': flow.id,
        'direction': 'websocket_start',
        'session_id': session_id,
        'timestamp_iso': datetime.fromtimestamp(flow.request.timestamp_start, tz=UTC).isoformat(),
        'url': flow.request.pretty_url,
        'headers': _headers_to_dict(flow.request.headers),
    }

    session_dir = _get_session_dir(session_id)
    filename = session_dir / f'ws_start_{n:03d}_{_safe_filename(flow.request.host, flow.request.path)}.json'
    _save_json_atomic(filename, capture)
    _log(f'\nWEBSOCKET START #{n}: {flow.request.pretty_url}\n')


def websocket_message(flow: http.HTTPFlow) -> None:
    """Capture WebSocket messages."""
    if not flow.websocket or not flow.websocket.messages:
        return

    message = flow.websocket.messages[-1]
    n = len(flow.websocket.messages)

    # Get session from our connection storage (set by client_connected hook)
    session: claude_workspace.Session | None = None
    if flow.client_conn:
        session = _get_session_for_connection(flow.client_conn.id)
    session_id = session.session_id if session else None

    capture = {
        'flow_id': flow.id,
        'message_index': n,
        'direction': 'websocket_message',
        'session_id': session_id,
        'from_client': message.from_client,
        'timestamp': message.timestamp,
        'message_type': message.type.name if hasattr(message.type, 'name') else str(message.type),
        'content': message.text if hasattr(message, 'text') and message.is_text else None,
        'content_size': len(message.content) if message.content else 0,
        'content_hash': hashlib.sha256(message.content).hexdigest()
        if message.content and not message.is_text
        else None,
    }

    session_dir = _get_session_dir(session_id)
    filename = session_dir / f'ws_msg_{flow.id[:8]}_{n:03d}.json'
    _save_json_atomic(filename, capture)


def websocket_end(flow: http.HTTPFlow) -> None:
    """Capture WebSocket connection closure."""
    message_count = len(flow.websocket.messages) if flow.websocket else 0

    # Get session from our connection storage (set by client_connected hook)
    session: claude_workspace.Session | None = None
    if flow.client_conn:
        session = _get_session_for_connection(flow.client_conn.id)
    session_id = session.session_id if session else None

    capture = {
        'flow_id': flow.id,
        'direction': 'websocket_end',
        'session_id': session_id,
        'message_count': message_count,
        'timestamp_iso': datetime.now(tz=UTC).isoformat(),
    }

    session_dir = _get_session_dir(session_id)
    filename = session_dir / f'ws_end_{flow.id[:8]}.json'
    _save_json_atomic(filename, capture)
    _log(f'WEBSOCKET END: {message_count} messages\n')
