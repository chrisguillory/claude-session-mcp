#!/usr/bin/env python3
"""Walk up the process tree and print each ancestor. Tests whether Claude Code is an ancestor."""

from __future__ import annotations

import os
import subprocess


def walk_process_tree() -> None:
    pid = os.getpid()
    print(f'Current PID: {pid}')
    print(f'Parent PID:  {os.getppid()}')
    print()
    print('Process tree (walking up):')
    print(f'{"PID":>8}  {"PPID":>8}  {"COMMAND"}')
    print(f'{"---":>8}  {"----":>8}  {"-------"}')

    current = pid
    depth = 0
    while current and current != 0 and depth < 20:
        try:
            result = subprocess.run(
                ['ps', '-p', str(current), '-o', 'pid=,ppid=,comm='],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                break
            line = result.stdout.strip()
            if not line:
                break
            parts = line.split(None, 2)
            if len(parts) < 3:
                break
            p, pp, comm = parts
            is_claude = 'claude' in comm.lower()
            marker = ' <-- CLAUDE FOUND!' if is_claude else ''
            print(f'{p:>8}  {pp:>8}  {comm}{marker}')
            current = int(pp)
            depth += 1
        except (ValueError, OSError):
            break

    print()
    if depth >= 20:
        print('Stopped after 20 levels.')


if __name__ == '__main__':
    walk_process_tree()
