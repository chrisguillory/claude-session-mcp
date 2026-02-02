---
name: version-test
description: Test Claude Code version behaviors. Install versions, run controlled prompts, inspect session artifacts.
argument-hint: <version> [target]
disable-model-invocation: true
allowed-tools: Bash, Read, Glob, Grep
---

# version-test

Explore Claude Code version behaviors empirically.

## Usage

```
/version-test <version> [target]
```

## Environment

```
Active:    !`readlink ~/.local/bin/claude | xargs basename 2>/dev/null`
Installed: !`ls ~/.local/share/claude/versions/ 2>/dev/null | grep -E '^[0-9]' | sort -V | tr '\n' ' '`
```

## Changelog

```bash
gh api repos/anthropics/claude-code/contents/CHANGELOG.md --jq '.content' | base64 -d
```

Use a subagent for extensive changelog analysis to avoid context bloat.

https://github.com/anthropics/claude-code/blob/main/CHANGELOG.md

## Observation targets

| Target | Prompt | Look for |
|--------|--------|----------|
| `agent` | "Launch agent to compute 1+1" | `subagents/` dir vs agent files in root |
| `schema` | Any prompt | Validate JSONL with Pydantic models |
| `feature` | n/a | `"$binary" --help` output |

## Workflow

### 1. Install

```bash
binary=$(.claude/skills/version-test/scripts/install-version.sh $0)
```

### 2. Session ID

`MMPP0000-0000-0000-0000-00000000000N` (MM=minor, PP=patch, N=counter)

### 3. Run

```bash
"$binary" --session-id "$SESSION_ID" -p "<prompt>"
```

### 4. Inspect

```bash
find ~/.claude/projects -name "*$SESSION_ID*"
ls -laR "$SESSION_DIR"
```

For schema validation:
```bash
uv run python3 << EOF
from pydantic import TypeAdapter
import json
from src.schemas.session import SessionRecord
adapter = TypeAdapter(SessionRecord)
for line in open("$SESSION_FILE"):
    if line.strip():
        adapter.validate_python(json.loads(line))
print("✓ valid")
EOF
```

### 5. Cleanup

```bash
uv run python -m src.cli.main delete "$SESSION_ID" --force --no-backup
```

## Known changes

| Version | Change |
|---------|--------|
| 2.1.2 | `subagents/` directory for agent files |
| 2.1.0 | Skills, forked contexts, hooks |
| 2.0.64 | Background agents, TaskOutputTool |