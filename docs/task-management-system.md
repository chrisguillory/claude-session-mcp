# Claude Code Task Management System (v2.1.16+)

Investigation by Claude Opus 4.5 under Chris's direction, January 2026.

## Overview

Claude Code 2.1.16 introduced a new task management system with dependency tracking. This document covers what we discovered through system prompt analysis and empirical testing.

## The Tools

Interactive mode provides four task tools:

| Tool         | Purpose                                             |
|--------------|-----------------------------------------------------|
| `TaskCreate` | Create a task with subject, description, activeForm |
| `TaskUpdate` | Update status, owner, dependencies, metadata        |
| `TaskList`   | View all tasks with blocked/unblocked state         |
| `TaskGet`    | Fetch full details of a specific task               |

### TaskCreate Parameters

```
subject: "Fix authentication bug"     # Brief title (imperative)
description: "Detailed requirements"  # Full context
activeForm: "Fixing auth bug"         # Shown in spinner (present continuous)
```

Tasks are created with `status: pending` and no owner.

### TaskUpdate Parameters

```
taskId: "1"                    # Required
status: "pending|in_progress|completed"
owner: "agent-name"            # For tracking who's working on what
addBlocks: ["2", "3"]          # Tasks that can't start until I finish
addBlockedBy: ["4", "5"]       # Tasks that must finish before I start
metadata: {key: "value"}       # Arbitrary data
```

### TaskList Output

```
#1 [pending] Build feature
#2 [in_progress] Write tests (TestRunner) [blocked by #1]
#3 [completed] Setup environment
```

Shows: ID, status, subject, owner (if set), blocked-by (if any).

## Critical Finding: Orchestrator-Only Architecture

**Subagents spawned via the `Task` tool do NOT have access to Task\* tools.**

We tested this by spawning agents and asking them to claim tasks. They reported:

> "I cannot complete this request because the required tools (TaskList, TaskUpdate) are not available in my current environment."

### Implications

The `owner` field is **not for workers to self-assign**. It's for the orchestrating agent to track assignments. The correct pattern:

```
1. Orchestrator: TaskCreate "Pick apples"
2. Orchestrator: TaskUpdate {taskId: "1", owner: "FruitPicker", status: "in_progress"}
3. Orchestrator: Task(spawn agent with prompt: "Pick apples in /orchard, return count")
4. Worker: (does actual work, returns: "Picked 47 apples")
5. Orchestrator: TaskUpdate {taskId: "1", status: "completed"}
```

Workers do **real work** (code, files, bash). Task management stays with the orchestrator.

## Critical Finding: No Dependency Enforcement

The dependency system is **purely informational**. There is zero validation or enforcement.

### Tests Performed

| Test                             | Expected | Actual                           |
|----------------------------------|----------|----------------------------------|
| Set owner on blocked task        | Reject   | Allowed                          |
| Mark blocked task in_progress    | Reject   | Allowed                          |
| Complete blocked task            | Reject   | Allowed (still shows "blocked"!) |
| Circular dependency A↔B          | Reject   | Allowed (creates deadlock)       |
| Self-dependency A→A              | Reject   | Allowed                          |
| Reference non-existent task #999 | Reject   | Silently ignored                 |

### Evidence

```
TaskCreate "Task A" → #1
TaskCreate "Task B" → #2
TaskUpdate {taskId: "2", addBlockedBy: ["1"]}

TaskList shows:
  #1 [pending] Task A
  #2 [pending] Task B [blocked by #1]

# Now violate the dependency:
TaskUpdate {taskId: "2", status: "completed"}

TaskList shows:
  #1 [pending] Task A
  #2 [completed] Task B [blocked by #1]  ← COMPLETED while "blocked"!
```

You can also create circular dependencies:

```
TaskUpdate {taskId: "1", addBlockedBy: ["2"]}
TaskUpdate {taskId: "2", addBlockedBy: ["1"]}

TaskList shows:
  #1 [pending] Task A [blocked by #2]
  #2 [pending] Task B [blocked by #1]  ← Deadlock, both permanently blocked
```

And self-dependencies:

```
TaskUpdate {taskId: "1", addBlockedBy: ["1"]}

TaskList shows:
  #1 [pending] Task A [blocked by #1]  ← Blocked by itself
```

## What The System Actually Is

The task management system is a **personal notebook** for the orchestrating agent:

- **Visibility**: Helps track what needs doing and what's blocked
- **No enforcement**: Completely honor-system based
- **No sharing**: Subagents can't see or modify tasks
- **No validation**: Accepts circular deps, self-deps, out-of-order completion

It's useful for:
- Keeping the orchestrator organized on complex multi-step work
- Showing the user progress via the task list display
- Planning work breakdown before execution

It's NOT useful for:
- Coordinating multiple agents (they can't see tasks)
- Enforcing execution order (no validation)
- Preventing mistakes (accepts everything)

## Best Practices

1. **Orchestrator checks dependencies manually** before starting work
2. **Pass context via prompt** when spawning workers, not via task system
3. **Don't rely on enforcement** - the system trusts you completely
4. **Use for visibility** - it helps humans see progress
5. **Avoid complex dependency graphs** - easy to create deadlocks

## Comparison: Interactive vs Non-Interactive Mode

| Feature      | Interactive                  | Non-Interactive (`-p`) |
|--------------|------------------------------|------------------------|
| Tool         | TaskCreate/Update/List/Get   | TodoWrite              |
| Granularity  | Individual task updates      | Replace entire list    |
| Dependencies | Yes (addBlocks/addBlockedBy) | No                     |
| Owner field  | Yes                          | No                     |
| Metadata     | Yes                          | No                     |

## DAG Visualization

See [task-dag.svg](../task-dag.svg) for a visual representation of a multi-team dependency graph we tested.

## Related Files

- `~/.claude/todos/` - Persisted todo state files
- Session `.jsonl` - May contain TodoWrite tool calls in non-interactive mode