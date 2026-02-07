#!/usr/bin/env -S uv run --no-project
# /// script
# dependencies = []
# ///

"""
Watch a Claude Code session JSONL and regenerate HTML on changes.

Monitors a session file for new records and re-runs claude-code-log to
regenerate the HTML view. Serves the HTML over a local HTTP server with
auto-reload — the browser refreshes automatically when new content arrives.

Usage:
    ./scripts/watch_session.py SESSION_ID [--interval 2.0] [--open-browser]

Examples:
    ./scripts/watch_session.py 408c123a
    ./scripts/watch_session.py 408c123a-1b11-4ddd-9e49-06baf2ca1d56 --open-browser
    ./scripts/watch_session.py 408c123a --interval 1.0

Requires:
    - rg (ripgrep) for session discovery
    - uvx (from uv) for running claude-code-log
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path

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
# HTTP Server
# ==============================================================================


class _SilentHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def start_http_server(directory: Path, port: int = 0) -> tuple[http.server.HTTPServer, int]:
    """Start a silent HTTP server in a daemon thread. Returns (server, port)."""
    handler = functools.partial(_SilentHandler, directory=str(directory))
    server = http.server.HTTPServer(('127.0.0.1', port), handler)
    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, actual_port


# ==============================================================================
# Regeneration
# ==============================================================================


INJECT_SCRIPT = """<script>
// Expand everything after page loads
setTimeout(()=>{
document.querySelectorAll("details").forEach(d=>d.setAttribute("open",""));
document.querySelectorAll(".filter-toggle").forEach(t=>t.classList.add("active"));
let e=document.getElementById("selectAll");if(e)e.click();
for(let i=0;i<5;i++){let f=document.querySelectorAll(".fold-bar-section.folded");if(!f.length)break;f.forEach(s=>s.click());}
},300);
// Auto-reload when HTML file changes
(()=>{
let lastLen=0;
const s=sessionStorage.getItem("_scrollY");
if(s){sessionStorage.removeItem("_scrollY");setTimeout(()=>window.scrollTo(0,parseInt(s)),500);}
setInterval(()=>{
fetch(location.pathname+"?t="+Date.now()).then(r=>r.text()).then(t=>{
if(lastLen&&t.length!==lastLen){sessionStorage.setItem("_scrollY",""+window.scrollY);location.reload();}
lastLen=t.length;
});
},3000);
})();
</script>"""


def expand_all_details(html_path: Path) -> None:
    """Post-process HTML to expand everything and enable auto-reload."""
    content = html_path.read_text()
    content = content.replace('</body>', INJECT_SCRIPT + '\n</body>')
    html_path.write_text(content)


def regenerate(jsonl_path: Path, html_path: Path) -> float:
    """Regenerate HTML atomically (no flash of missing page). Returns generation time."""
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
            expand_all_details(tmp_html)
            os.replace(tmp_html, html_path)

    return elapsed


# ==============================================================================
# Watch Loop
# ==============================================================================


def watch(jsonl_path: Path, interval: float, open_browser: bool) -> None:
    """Poll for changes and regenerate HTML."""
    html_path = jsonl_path.with_suffix('.html')
    session_id = jsonl_path.stem

    # Start HTTP server in the HTML directory
    _, port = start_http_server(html_path.parent)
    url = f'http://127.0.0.1:{port}/{html_path.name}'

    print(f'{Colors.BOLD}{"=" * 60}{Colors.RESET}')
    print(f'{Colors.BOLD}Claude Code Session Watcher{Colors.RESET}')
    print(f'{Colors.BOLD}{"=" * 60}{Colors.RESET}')
    print()
    print(f'  {Colors.CYAN}Session:{Colors.RESET}  {session_id}')
    print(f'  {Colors.CYAN}File:{Colors.RESET}     {jsonl_path}')
    print(f'  {Colors.CYAN}Serving:{Colors.RESET}  {url}')
    print(f'  {Colors.CYAN}Polling:{Colors.RESET}  every {interval}s')
    print()

    # Initial generation
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

    print()
    print(f'{Colors.DIM}Watching for changes... (Ctrl+C to stop){Colors.RESET}')
    print()

    # Track file state
    stat = jsonl_path.stat()
    last_size = stat.st_size
    last_mtime = stat.st_mtime

    while True:
        time.sleep(interval)

        try:
            stat = jsonl_path.stat()
        except FileNotFoundError:
            print(f'\n{Colors.YELLOW}Session file removed, stopping.{Colors.RESET}')
            break

        if stat.st_size != last_size or stat.st_mtime != last_mtime:
            last_size = stat.st_size
            last_mtime = stat.st_mtime

            ts = time.strftime('%H:%M:%S')
            print(
                f'{Colors.DIM}[{ts}]{Colors.RESET} Change detected ({last_size:,} bytes) — regenerating...', flush=True
            )

            elapsed = regenerate(jsonl_path, html_path)
            size_kb = html_path.stat().st_size / 1024 if html_path.exists() else 0
            print(
                f'{Colors.DIM}[{ts}]{Colors.RESET} {Colors.GREEN}Regenerated{Colors.RESET} in {elapsed:.1f}s ({size_kb:.0f} KB)'
            )


# ==============================================================================
# CLI
# ==============================================================================


def main() -> None:
    signal.signal(signal.SIGINT, lambda *_: (print(f'\n{Colors.DIM}Stopped.{Colors.RESET}'), sys.exit(0)))

    parser = argparse.ArgumentParser(
        description='Watch a Claude Code session and regenerate HTML on changes.',
    )
    parser.add_argument('session', help='Session ID or prefix')
    parser.add_argument('--interval', type=float, default=2.0, help='Polling interval in seconds (default: 2.0)')
    parser.add_argument('--open-browser', action='store_true', help='Open HTML in browser after first generation')

    args = parser.parse_args()

    if not shutil.which('uvx'):
        print(f'{Colors.RED}Error:{Colors.RESET} uvx not found on PATH (install uv)')
        sys.exit(1)

    jsonl_path = find_session(args.session)
    watch(jsonl_path, args.interval, args.open_browser)


if __name__ == '__main__':
    main()
