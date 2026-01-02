# Domain Models

Application-level models for Claude Code session management.

## Current State

All models are in `models.py` (~185 lines). Small enough to stay together for now.

## Contents

- **DomainModel** - Base class (mutable, unlike frozen StrictModel)
- **Session, AgentSession** - Session file representations
- **CompleteSessionArchive** - Multi-file container for teleportation
- **SessionAnalysis, SessionMetadata, TokenCosts** - Analytics

Note: `schemas/session/` has minimal SessionMetadata/SessionAnalysis for JSONL parsing. The domain versions here are richer, with additional fields for full analysis workflows.

## Planned Split

When analysis capabilities expand:

```
domain/
├── __init__.py         # Re-exports
├── session.py          # Session, AgentSession, CompleteSessionArchive
├── analysis.py         # SessionAnalysis, SessionMetadata, TokenCosts
└── timeline.py         # TimelineEvent, ReplayableSession (future)
```

## Future Types (APM-style analysis)

```python
# domain/timeline.py (FUTURE)
class TimelineEvent(DomainModel):
    timestamp: datetime
    event_type: Literal['user_message', 'assistant_response', 'tool_call', ...]
    record: SessionRecord
    cumulative_tokens: int
    cumulative_cost: Decimal

class ReplayableSession(DomainModel):
    events: list[TimelineEvent]
    def step_forward(self) -> TimelineEvent: ...
    def seek(self, index: int) -> TimelineEvent: ...

class SessionMetrics(DomainModel):
    duration: timedelta
    token_breakdown: TokenBreakdown
    tool_usage: dict[str, ToolStats]
    cache_efficiency: CacheEfficiency
```

## See Also

- `schemas/operations/discovery.py` - SessionInfo for session discovery
- `schemas/session/` - Frozen JSONL record schemas