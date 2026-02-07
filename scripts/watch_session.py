#!/usr/bin/env -S uv run --no-project
# /// script
# dependencies = []
# ///

"""
Watch a Claude Code session JSONL with live-updating HTML viewer.

Does an initial full render via claude-code-log, then streams new records
to the browser in real-time via Server-Sent Events. No page reloads needed.

Usage:
    ./scripts/watch_session.py SESSION_ID [--interval 1.0] [--open-browser]

Examples:
    ./scripts/watch_session.py 408c123a --open-browser
    ./scripts/watch_session.py 408c123a-1b11-4ddd-9e49-06baf2ca1d56

Requires:
    - rg (ripgrep) for session discovery
    - uvx (from uv) for running claude-code-log
"""

from __future__ import annotations

import argparse
import contextlib
import http.server
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from html import escape as html_escape
from pathlib import Path
from typing import Any

# ==============================================================================
# Terminal Colors
# ==============================================================================


class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

    @classmethod
    def disable(cls) -> None:
        for attr in ('RED', 'GREEN', 'YELLOW', 'BLUE', 'CYAN', 'BOLD', 'DIM', 'RESET'):
            setattr(cls, attr, '')


if not sys.stdout.isatty():
    Colors.disable()


# ==============================================================================
# Session Discovery
# ==============================================================================


def find_session(session_id_or_prefix: str) -> Path:
    """Find a session JSONL by ID or prefix using rg."""
    claude_dir = Path.home() / '.claude' / 'projects'

    if not claude_dir.exists():
        print(f'{Colors.RED}Error:{Colors.RESET} {claude_dir} does not exist')
        sys.exit(1)

    if not shutil.which('rg'):
        print(f'{Colors.RED}Error:{Colors.RESET} rg (ripgrep) not found on PATH')
        sys.exit(1)

    result = subprocess.run(
        ['rg', '--files', '--glob', f'{session_id_or_prefix}*.jsonl', str(claude_dir)],
        capture_output=True,
        text=True,
    )

    if not result.stdout.strip():
        print(f'{Colors.RED}Error:{Colors.RESET} No session found matching "{session_id_or_prefix}"')
        sys.exit(1)

    matches = [Path(p) for p in result.stdout.strip().split('\n') if p]
    session_files = [m for m in matches if not m.name.startswith('agent-')]

    if not session_files:
        print(f'{Colors.RED}Error:{Colors.RESET} No session found matching "{session_id_or_prefix}"')
        sys.exit(1)

    if len(session_files) > 1:
        print(f'{Colors.YELLOW}Ambiguous prefix "{session_id_or_prefix}" matches:{Colors.RESET}')
        for f in session_files:
            print(f'  {f.stem}')
        sys.exit(1)

    return session_files[0]


# ==============================================================================
# JSONL Tail State
# ==============================================================================


class TailState:
    """Tracks byte offset in a JSONL file for incremental reading."""

    def __init__(self, jsonl_path: Path) -> None:
        self.jsonl_path = jsonl_path
        self.byte_offset: int = 0
        self.partial_line: bytes = b''

    def snapshot_end(self) -> None:
        """Set offset to current end of file (after initial render)."""
        self.byte_offset = self.jsonl_path.stat().st_size
        self.partial_line = b''

    def read_new_records(self) -> list[dict[str, Any]]:
        """Read new complete JSONL lines since last offset. Returns parsed dicts."""
        try:
            size = self.jsonl_path.stat().st_size
        except FileNotFoundError:
            return []

        if size <= self.byte_offset and not self.partial_line:
            return []

        with open(self.jsonl_path, 'rb') as f:
            f.seek(self.byte_offset)
            raw = f.read()

        if not raw:
            return []

        # Prepend any buffered partial line from previous read
        if self.partial_line:
            raw = self.partial_line + raw
            self.partial_line = b''

        # Split into lines; buffer incomplete trailing line
        parts = raw.split(b'\n')
        if raw.endswith(b'\n'):
            complete_lines = parts[:-1]  # last element is empty string after final \n
            self.byte_offset += len(raw)
        else:
            complete_lines = parts[:-1]
            self.partial_line = parts[-1]
            self.byte_offset += len(raw) - len(self.partial_line)

        records = []
        for line_bytes in complete_lines:
            line = line_bytes.strip()
            if not line:
                continue
            with contextlib.suppress(json.JSONDecodeError):
                records.append(json.loads(line))
        return records


# ==============================================================================
# HTML Fragment Renderer
# ==============================================================================

LONG_CONTENT_THRESHOLD = 2000


def _next_id(counter: list[int]) -> str:
    counter[0] += 1
    return f't-{counter[0]}'


def _format_timestamp(iso_ts: str) -> str:
    if not iso_ts:
        return ''
    try:
        return iso_ts.replace('T', ' ')[:19]
    except (IndexError, TypeError):
        return str(iso_ts)


def _render_usage(usage: dict[str, Any]) -> str:
    if not usage:
        return ''
    parts = [f'Input: {usage.get("input_tokens", 0)}', f'Output: {usage.get("output_tokens", 0)}']
    cc = usage.get('cache_creation_input_tokens', 0)
    cr = usage.get('cache_read_input_tokens', 0)
    if cc:
        parts.append(f'Cache Creation: {cc}')
    if cr:
        parts.append(f'Cache Read: {cr}')
    return f"<span class='token-usage'>{' | '.join(parts)}</span>"


def _wrap_message(
    msg_id: str,
    classes: str,
    icon: str,
    label: str,
    timestamp: str,
    display_time: str,
    content_html: str,
    usage_html: str = '',
) -> str:
    header_label = f'{icon} {label}' if icon else label
    return (
        f"<div class='message {classes}' data-message-id='{msg_id}' id='msg-{msg_id}'>"
        f"<div class='header'>"
        f'<span>{header_label}</span>'
        f"<div class='header-info'>"
        f"<div class='timestamp-row'>"
        f"<span class='timestamp' data-timestamp='{html_escape(timestamp)}'>"
        f'{html_escape(display_time)}</span>'
        f'</div>'
        f'{usage_html}'
        f'</div></div>'
        f"<div class='content'>{content_html}</div>"
        f'</div>'
    )


def _render_long_text(text: str) -> str:
    if len(text) > LONG_CONTENT_THRESHOLD:
        preview = html_escape(text[:300]) + '...'
        return (
            f"<details open class='collapsible-details'>"
            f"<summary><div class='preview-content'><pre>{preview}</pre></div></summary>"
            f"<div class='details-content'><pre>{html_escape(text)}</pre></div>"
            f'</details>'
        )
    return f'<pre>{html_escape(text)}</pre>'


def _render_tool_params(tool_input: dict[str, Any]) -> str:
    if not isinstance(tool_input, dict) or not tool_input:
        return ''
    rows = []
    for key, value in tool_input.items():
        val_str = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
        val_display = html_escape(val_str[:200]) + '...' if len(val_str) > 200 else html_escape(val_str)
        rows.append(
            f"<tr><td class='tool-param-key'>{html_escape(key)}</td>"
            f"<td class='tool-param-value'>{val_display}</td></tr>"
        )
    return f"<table class='tool-params-table'>{''.join(rows)}</table>"


def _render_user(record: dict[str, Any], counter: list[int], ts: str, dt: str) -> list[str]:
    message = record.get('message', {})
    content = message.get('content')
    if content is None:
        return []

    sidechain = ' sidechain' if record.get('isSidechain') else ''
    fragments = []

    if isinstance(content, str):
        mid = _next_id(counter)
        fragments.append(
            _wrap_message(mid, f'user{sidechain}', '🤷', 'User', ts, dt, f'<pre>{html_escape(content)}</pre>')
        )
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get('type', '')
            if btype == 'text':
                mid = _next_id(counter)
                fragments.append(
                    _wrap_message(
                        mid,
                        f'user{sidechain}',
                        '🤷',
                        'User',
                        ts,
                        dt,
                        f'<pre>{html_escape(block.get("text", ""))}</pre>',
                    )
                )
            elif btype == 'tool_result':
                mid = _next_id(counter)
                is_error = block.get('is_error', False)
                err_cls = ' error' if is_error else ''
                icon = '🚨' if is_error else ''
                label = 'Error' if is_error else ''
                result_content = block.get('content', '')
                if isinstance(result_content, str):
                    body = _render_long_text(result_content)
                elif isinstance(result_content, list):
                    parts = [
                        p.get('text', '') for p in result_content if isinstance(p, dict) and p.get('type') == 'text'
                    ]
                    body = _render_long_text('\n'.join(parts)) if parts else ''
                else:
                    body = ''
                fragments.append(_wrap_message(mid, f'tool_result{err_cls}{sidechain}', icon, label, ts, dt, body))
    return fragments


def _render_assistant(record: dict[str, Any], counter: list[int], ts: str, dt: str) -> list[str]:
    message = record.get('message', {})
    content = message.get('content')
    if not isinstance(content, list):
        return []

    usage = message.get('usage') or record.get('usage') or {}
    usage_html = _render_usage(usage)
    sidechain = ' sidechain' if record.get('isSidechain') else ''
    fragments = []
    usage_shown = False

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get('type', '')
        use_usage = usage_html if not usage_shown else ''

        if btype == 'thinking':
            mid = _next_id(counter)
            text = block.get('thinking', '')
            fragments.append(
                _wrap_message(
                    mid,
                    f'thinking{sidechain}',
                    '💭',
                    'Thinking',
                    ts,
                    dt,
                    f'<div class="thinking-content markdown"><pre>{html_escape(text)}</pre></div>',
                    use_usage,
                )
            )
            usage_shown = True

        elif btype == 'text':
            mid = _next_id(counter)
            text = block.get('text', '')
            fragments.append(
                _wrap_message(
                    mid,
                    f'assistant{sidechain}',
                    '🤖',
                    'Assistant',
                    ts,
                    dt,
                    f'<div class="assistant-text markdown"><pre>{html_escape(text)}</pre></div>',
                    use_usage,
                )
            )
            usage_shown = True

        elif btype == 'tool_use':
            mid = _next_id(counter)
            tool_name = block.get('name', 'Unknown')
            tool_input = block.get('input', {})
            content_html = _render_tool_params(tool_input)
            fragments.append(
                _wrap_message(
                    mid,
                    f'tool_use{sidechain}',
                    '🛠️',
                    tool_name,
                    ts,
                    dt,
                    content_html,
                )
            )

    return fragments


def _render_system(record: dict[str, Any], counter: list[int], ts: str, dt: str) -> list[str]:
    subtype = record.get('subtype', '')
    if subtype not in ('local_command', 'informational', 'api_error', 'compact_boundary'):
        return []

    mid = _next_id(counter)
    content_text = record.get('content', '')
    level = record.get('level', 'info')
    level_class = f'system-{level}' if level in ('warning', 'error', 'info') else 'system-info'
    icon_map = {'error': '🚨', 'warning': '⚠️', 'info': '⚙️'}
    icon = icon_map.get(level, '⚙️')
    label = f'System {level.capitalize()}'

    if subtype == 'api_error':
        error = record.get('error', {})
        error_msg = str(error) if not isinstance(error, dict) else error.get('message', str(error))
        body = f'<pre>{html_escape(str(error_msg))}</pre>'
        level_class = 'system-error'
        icon = '🚨'
        label = 'API Error'
    elif subtype == 'compact_boundary':
        meta = record.get('compactMetadata', {}) or {}
        trigger = meta.get('trigger', 'unknown')
        pre_tokens = meta.get('preTokens', '?')
        body = f'<pre>Conversation compacted ({trigger}, {pre_tokens} tokens before)</pre>'
        level_class = 'system-info'
    else:
        body = f'<pre>{html_escape(str(content_text))}</pre>' if content_text else ''

    return [_wrap_message(mid, f'system {level_class}', icon, label, ts, dt, body)]


def render_record(record: dict[str, Any], counter: list[int]) -> list[str]:
    """Render a JSONL record as HTML fragment(s) matching claude-code-log CSS."""
    rec_type = record.get('type', '')
    timestamp = record.get('timestamp', '')
    display_time = _format_timestamp(timestamp)

    if rec_type == 'user':
        return _render_user(record, counter, timestamp, display_time)
    elif rec_type == 'assistant':
        return _render_assistant(record, counter, timestamp, display_time)
    elif rec_type == 'system':
        return _render_system(record, counter, timestamp, display_time)
    return []


# ==============================================================================
# SSE Broadcasting
# ==============================================================================


def broadcast_sse(server: http.server.HTTPServer, html: str, event_id: str = '') -> None:
    """Push an HTML fragment to all connected SSE clients."""
    encoded = json.dumps(html)
    with server.sse_lock:  # type: ignore[attr-defined]
        dead: list[queue.Queue] = []  # type: ignore[type-arg]
        for q in server.sse_clients:  # type: ignore[attr-defined]
            try:
                q.put_nowait((encoded, event_id))
            except queue.Full:
                dead.append(q)
        for q in dead:
            server.sse_clients.remove(q)  # type: ignore[attr-defined]


def broadcast_event(server: http.server.HTTPServer, event: str, data: str = '') -> None:
    """Push a named SSE event to all connected clients."""
    with server.sse_lock:  # type: ignore[attr-defined]
        for q in server.sse_clients:  # type: ignore[attr-defined]
            with contextlib.suppress(queue.Full):
                q.put_nowait(('__event__', f'event: {event}\ndata: {data}\n\n'))


# ==============================================================================
# HTTP Server with SSE Support
# ==============================================================================


class _WatchHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with SSE endpoint for live session tailing."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def finish(self) -> None:
        """Suppress BrokenPipeError during cleanup (CPython bug #14574)."""
        with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError):
            super().finish()

    def do_GET(self) -> None:
        if self.path == '/events' or self.path.startswith('/events?'):
            self._handle_sse()
        elif self.path == '/refresh':
            self._handle_refresh()
        else:
            super().do_GET()

    def _handle_sse(self) -> None:
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('X-Accel-Buffering', 'no')
        self.end_headers()

        # Send retry directive
        self.wfile.write(b'retry: 3000\n\n')
        self.wfile.flush()

        # Check for Last-Event-ID reconnect
        last_id = self.headers.get('Last-Event-ID', '')
        if last_id:
            # Replay records from that byte offset
            self._replay_from_offset(last_id)

        client_queue: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=1000)
        with self.server.sse_lock:  # type: ignore[attr-defined]
            self.server.sse_clients.append(client_queue)  # type: ignore[attr-defined]

        try:
            while True:
                try:
                    encoded, event_id = client_queue.get(timeout=15)
                    if encoded == '__event__':
                        # Named event - event_id contains the full formatted event
                        self.wfile.write(event_id.encode())
                    else:
                        msg = f'data: {encoded}\n'
                        if event_id:
                            msg += f'id: {event_id}\n'
                        msg += '\n'
                        self.wfile.write(msg.encode())
                    self.wfile.flush()
                except queue.Empty:
                    # Heartbeat
                    self.wfile.write(b': heartbeat\n\n')
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with self.server.sse_lock:  # type: ignore[attr-defined]
                if client_queue in self.server.sse_clients:  # type: ignore[attr-defined]
                    self.server.sse_clients.remove(client_queue)  # type: ignore[attr-defined]

    def _replay_from_offset(self, last_id: str) -> None:
        """Replay records from a byte offset (Last-Event-ID reconnect)."""
        try:
            offset = int(last_id)
        except (ValueError, TypeError):
            return

        jsonl_path = self.server.jsonl_path  # type: ignore[attr-defined]
        try:
            size = jsonl_path.stat().st_size
        except FileNotFoundError:
            return

        if offset >= size or offset < 0:
            return

        with open(jsonl_path, 'rb') as f:
            f.seek(offset)
            raw = f.read()

        if not raw:
            return

        counter = self.server.msg_counter  # type: ignore[attr-defined]
        lines = raw.split(b'\n')
        current_offset = offset
        for line_bytes in lines:
            if not line_bytes.strip():
                current_offset += len(line_bytes) + 1
                continue
            try:
                record = json.loads(line_bytes)
                fragments = render_record(record, counter)
                for html_frag in fragments:
                    encoded = json.dumps(html_frag)
                    eid = str(current_offset + len(line_bytes) + 1)
                    self.wfile.write(f'data: {encoded}\nid: {eid}\n\n'.encode())
            except json.JSONDecodeError:
                pass
            current_offset += len(line_bytes) + 1

        self.wfile.flush()

    def _handle_refresh(self) -> None:
        self.server.refresh_flag.set()  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'OK')


def start_http_server(
    directory: Path,
    jsonl_path: Path,
    msg_counter: list[int],
    port: int = 0,
) -> tuple[http.server.HTTPServer, int]:
    """Start threaded HTTP server with SSE support. Returns (server, port)."""
    server = http.server.ThreadingHTTPServer(
        ('127.0.0.1', port),
        lambda *args, **kwargs: _WatchHandler(*args, directory=str(directory), **kwargs),
    )
    server.daemon_threads = True

    # Shared state on server object (avoids handler __init__ ordering issues)
    server.sse_clients = []  # type: ignore[attr-defined]
    server.sse_lock = threading.Lock()  # type: ignore[attr-defined]
    server.refresh_flag = threading.Event()  # type: ignore[attr-defined]
    server.jsonl_path = jsonl_path  # type: ignore[attr-defined]
    server.msg_counter = msg_counter  # type: ignore[attr-defined]

    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, actual_port


# ==============================================================================
# Initial HTML Generation
# ==============================================================================


INJECT_SCRIPT = """<div id="live-tail"></div>
<script>
// Expand everything after page loads
setTimeout(()=>{
  document.querySelectorAll("details").forEach(d=>d.setAttribute("open",""));
  document.querySelectorAll(".filter-toggle").forEach(t=>t.classList.add("active"));
  let e=document.getElementById("selectAll");if(e)e.click();
  for(let i=0;i<5;i++){let f=document.querySelectorAll(".fold-bar-section.folded");if(!f.length)break;f.forEach(s=>s.click());}
},300);

// SSE Live Tail
(()=>{
  const tail = document.getElementById("live-tail");
  const scrollThreshold = 200;
  let lastEvent = Date.now();

  function isNearBottom(){
    return (document.documentElement.scrollHeight - window.innerHeight - window.scrollY) < scrollThreshold;
  }

  // Full Refresh button
  const refreshBtn = document.createElement("button");
  refreshBtn.className = "floating-btn";
  refreshBtn.title = "Full Refresh (re-render entire session)";
  refreshBtn.textContent = "\\u{1F504}";
  refreshBtn.style.cssText = "position:fixed;bottom:260px;right:20px;z-index:1000;";
  refreshBtn.addEventListener("click", () => {
    refreshBtn.textContent = "\\u23F3";
    fetch("/refresh").then(() => {
      const check = setInterval(() => {
        fetch(location.pathname + "?t=" + Date.now())
          .then(() => { clearInterval(check); location.reload(); });
      }, 1000);
    });
  });
  document.body.appendChild(refreshBtn);

  // Scroll-to-bottom button
  const scrollBtn = document.createElement("button");
  scrollBtn.className = "floating-btn";
  scrollBtn.title = "Scroll to bottom";
  scrollBtn.textContent = "\\u{2B07}\\u{FE0F}";
  scrollBtn.style.cssText = "position:fixed;bottom:210px;right:20px;z-index:1000;";
  scrollBtn.addEventListener("click", () => {
    window.scrollTo({top: document.documentElement.scrollHeight, behavior: "smooth"});
  });
  document.body.appendChild(scrollBtn);

  // SSE connection with reconnect
  function connectSSE(){
    const lastId = localStorage.getItem("_sse_last_id") || "";
    const url = "/events" + (lastId ? "?last_id=" + lastId : "");
    const es = new EventSource(url);

    es.onmessage = function(e){
      lastEvent = Date.now();
      if(e.lastEventId) localStorage.setItem("_sse_last_id", e.lastEventId);

      let html;
      try { html = JSON.parse(e.data); } catch(err){ return; }

      const wasNear = isNearBottom();
      tail.insertAdjacentHTML("beforeend", html);

      // Expand any details in new content
      const last = tail.lastElementChild;
      if(last && last.querySelectorAll){
        last.querySelectorAll("details").forEach(d=>d.setAttribute("open",""));
      }

      if(wasNear){
        window.scrollTo({top: document.documentElement.scrollHeight, behavior: "smooth"});
      }
    };

    es.addEventListener("reload", function(){
      location.reload();
    });

    es.onerror = function(){
      es.close();
      setTimeout(connectSSE, 3000);
    };

    // Client-side heartbeat timeout (handles sleep/wake zombie connections)
    const hbCheck = setInterval(()=>{
      if(Date.now() - lastEvent > 30000){
        clearInterval(hbCheck);
        es.close();
        setTimeout(connectSSE, 1000);
      }
    }, 5000);
  }

  connectSSE();
})();
</script>"""


def inject_live_tail(html_path: Path) -> None:
    """Post-process HTML to add live-tail container and SSE client script."""
    content = html_path.read_text()
    content = content.replace('</body>', INJECT_SCRIPT + '\n</body>')
    html_path.write_text(content)


def regenerate(jsonl_path: Path, html_path: Path) -> float:
    """Full HTML generation via claude-code-log (atomic swap). Returns elapsed time."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_jsonl = Path(tmp_dir) / jsonl_path.name
        shutil.copy2(jsonl_path, tmp_jsonl)

        start = time.monotonic()
        result = subprocess.run(
            ['uvx', 'claude-code-log@latest', str(tmp_jsonl)],
            capture_output=True,
            text=True,
        )
        elapsed = time.monotonic() - start

        if result.returncode != 0:
            print(f'{Colors.RED}Error from claude-code-log:{Colors.RESET}')
            print(result.stderr or result.stdout)
            return elapsed

        tmp_html = tmp_jsonl.with_suffix('.html')
        if tmp_html.exists():
            inject_live_tail(tmp_html)
            os.replace(tmp_html, html_path)

    return elapsed


# ==============================================================================
# Watch Loop
# ==============================================================================


def watch(jsonl_path: Path, interval: float, open_browser: bool) -> None:
    """Watch JSONL and stream new records via SSE."""
    html_path = jsonl_path.with_suffix('.html')
    session_id = jsonl_path.stem
    msg_counter = [0]

    # Start HTTP server
    server, port = start_http_server(html_path.parent, jsonl_path, msg_counter)
    url = f'http://127.0.0.1:{port}/{html_path.name}'

    print(f'{Colors.BOLD}{"=" * 60}{Colors.RESET}')
    print(f'{Colors.BOLD}Claude Code Session Watcher (SSE){Colors.RESET}')
    print(f'{Colors.BOLD}{"=" * 60}{Colors.RESET}')
    print()
    print(f'  {Colors.CYAN}Session:{Colors.RESET}  {session_id}')
    print(f'  {Colors.CYAN}File:{Colors.RESET}     {jsonl_path}')
    print(f'  {Colors.CYAN}Serving:{Colors.RESET}  {url}')
    print(f'  {Colors.CYAN}Polling:{Colors.RESET}  every {interval}s')
    print()

    # Initial full generation
    ts = time.strftime('%H:%M:%S')
    print(f'{Colors.DIM}[{ts}]{Colors.RESET} Initial generation...', flush=True)
    elapsed = regenerate(jsonl_path, html_path)
    size_kb = html_path.stat().st_size / 1024 if html_path.exists() else 0
    print(
        f'{Colors.DIM}[{ts}]{Colors.RESET} {Colors.GREEN}Generated{Colors.RESET} in {elapsed:.1f}s ({size_kb:.0f} KB)'
    )

    if open_browser:
        webbrowser.open(url)
        print(f'{Colors.DIM}[{ts}]{Colors.RESET} Opened in browser')

    # Start tailing from current end of file
    tail = TailState(jsonl_path)
    tail.snapshot_end()

    print()
    print(f'{Colors.DIM}Streaming via SSE... (Ctrl+C to stop){Colors.RESET}')
    print()

    while True:
        time.sleep(interval)

        # Check for full refresh request
        if server.refresh_flag.is_set():  # type: ignore[attr-defined]
            server.refresh_flag.clear()  # type: ignore[attr-defined]
            ts = time.strftime('%H:%M:%S')
            print(f'{Colors.DIM}[{ts}]{Colors.RESET} Full refresh requested...', flush=True)
            elapsed = regenerate(jsonl_path, html_path)
            tail.snapshot_end()
            msg_counter[0] = 0
            broadcast_event(server, 'reload', 'refresh')
            print(f'{Colors.DIM}[{ts}]{Colors.RESET} {Colors.GREEN}Regenerated{Colors.RESET} in {elapsed:.1f}s')
            continue

        # Check for file removal
        try:
            file_size = jsonl_path.stat().st_size
        except FileNotFoundError:
            print(f'\n{Colors.YELLOW}Session file removed, stopping.{Colors.RESET}')
            break

        # Check for file truncation
        if file_size < tail.byte_offset:
            ts = time.strftime('%H:%M:%S')
            print(f'{Colors.DIM}[{ts}]{Colors.RESET} {Colors.YELLOW}File truncated — full refresh{Colors.RESET}')
            elapsed = regenerate(jsonl_path, html_path)
            tail.snapshot_end()
            msg_counter[0] = 0
            broadcast_event(server, 'reload', 'truncated')
            continue

        # Read and stream new records
        new_records = tail.read_new_records()
        if not new_records:
            continue

        ts = time.strftime('%H:%M:%S')
        rendered_count = 0

        for record in new_records:
            fragments = render_record(record, msg_counter)
            for html_frag in fragments:
                broadcast_sse(server, html_frag, str(tail.byte_offset))
                rendered_count += 1

        if rendered_count:
            print(
                f'{Colors.DIM}[{ts}]{Colors.RESET} Streamed {rendered_count} block(s) from {len(new_records)} record(s)'
            )


# ==============================================================================
# CLI
# ==============================================================================


def main() -> None:
    signal.signal(signal.SIGINT, lambda *_: (print(f'\n{Colors.DIM}Stopped.{Colors.RESET}'), sys.exit(0)))

    parser = argparse.ArgumentParser(
        description='Watch a Claude Code session with live SSE updates.',
    )
    parser.add_argument('session', help='Session ID or prefix')
    parser.add_argument('--interval', type=float, default=1.0, help='Polling interval in seconds (default: 1.0)')
    parser.add_argument('--open-browser', action='store_true', help='Open HTML in browser after first generation')

    args = parser.parse_args()

    if not shutil.which('uvx'):
        print(f'{Colors.RED}Error:{Colors.RESET} uvx not found on PATH (install uv)')
        sys.exit(1)

    jsonl_path = find_session(args.session)
    watch(jsonl_path, args.interval, args.open_browser)


if __name__ == '__main__':
    main()
