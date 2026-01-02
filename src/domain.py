"""
Domain models for Claude Code session management.

This module contains enriched models with computed fields, metadata, and application-level
structures. These models are built on top of the raw data models from models.py.

Separation of concerns:
- models.py: Pure JSONL schema representations (parsing)
- domain.py: Application logic, analysis, and enrichment (this file)

Architecture (top-down):
1. CompleteSessionArchive - Complete session with all agent sessions
2. Session - Main session file
3. AgentSession - Individual agent session file
4. SessionAnalysis - Analysis with costs and insights
5. SessionMetadata - Computed statistics
6. SessionList - Discovery results
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.session import SessionRecord

# ==============================================================================
# Base Configuration for Domain Models
# ==============================================================================


class DomainModel(BaseModel):
    """Base model for domain/application models."""

    model_config = ConfigDict(
        extra='forbid',
        strict=True,
        frozen=False,
    )


# ==============================================================================
# Top Level: Complete Session Archive (for memorialization/teleportation)
# ==============================================================================


class CompleteSessionArchive(DomainModel):
    """
    Complete session archive including all agent sessions for memorialization.

    This is the top-level structure used for session teleportation - it contains
    everything needed to restore a session on a different machine.
    """

    main_session: Session
    agent_sessions: dict[str, AgentSession] = Field(default_factory=dict)  # Keyed by agent_id
    original_project_path: str  # Original project path before translation
    archived_at: str  # ISO timestamp when archive was created


# ==============================================================================
# Session Level: Main and Agent Sessions
# ==============================================================================


class Session(DomainModel):
    """
    A complete Claude Code session with all records from a single JSONL file.

    Represents the main session file: ~/.claude/projects/{project}/ {session_id}.jsonl
    """

    session_id: str  # The UUID of the session
    records: list[SessionRecord]  # All records from {session_id}.jsonl
    agent_ids: list[str] = Field(default_factory=list)  # Referenced agent IDs (computed from records)


class AgentSession(DomainModel):
    """
    An agent/subprocess session from a separate file.

    Represents an agent session file: ~/.claude/projects/{project}/agent-{agentId}.jsonl
    Linked to main session via agentId foreign key in UserRecord.
    """

    agent_id: str  # The agent ID (e.g., '207bf8be')
    parent_session_id: str  # Foreign key to parent Session
    records: list[SessionRecord]  # All records from agent-{agentId}.jsonl


# ==============================================================================
# Analysis Level: Enriched with Costs and Insights
# ==============================================================================


class SessionAnalysis(DomainModel):
    """
    Complete analysis of a session with costs and insights.

    Used by analyze_session tool to provide rich analytics.
    """

    metadata: SessionMetadata

    # Cost analysis
    cost_breakdown: list[TokenCosts] = Field(default_factory=list)
    total_cost_usd: float = 0.0

    # Timing
    duration_seconds: float | None = None
    first_message_time: str | None = None
    last_message_time: str | None = None

    # Summary
    summary_text: str | None = None  # From SummaryRecord if present

    # Insights
    average_response_time_ms: float | None = None
    tools_by_frequency: dict[str, int] = Field(default_factory=dict)
    errors_encountered: int = 0


# ==============================================================================
# Metadata Level: Computed Statistics
# ==============================================================================


class SessionMetadata(DomainModel):
    """
    Metadata computed from analyzing a session.

    Contains aggregated statistics about token usage, tools, files, etc.
    """

    session_id: str
    record_count: int
    first_timestamp: str
    last_timestamp: str

    # Path information
    unique_cwds: list[str] = Field(default_factory=list)
    unique_project_paths: list[str] = Field(default_factory=list)

    # Record type counts
    user_message_count: int = 0
    assistant_message_count: int = 0
    summary_count: int = 0
    system_message_count: int = 0
    file_history_snapshot_count: int = 0
    queue_operation_count: int = 0

    # Token usage (aggregated)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0

    # Usage patterns
    models_used: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    files_touched: list[str] = Field(default_factory=list)
    git_branches: list[str] = Field(default_factory=list)

    # Agent information
    agent_count: int = 0  # Number of agent sessions referenced
    agent_ids: list[str] = Field(default_factory=list)
    has_agents: bool = False


class TokenCosts(DomainModel):
    """Token cost breakdown by model."""

    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    cache_creation_cost_usd: float
    cache_read_cost_usd: float
    total_cost_usd: float


# ==============================================================================
# Discovery Level: Session Lists and Info
# ==============================================================================


class SessionList(DomainModel):
    """
    List of discovered sessions.

    Used by list_sessions tool to return discovery results.
    """

    sessions: list[SessionInfo] = Field(default_factory=list)
    total_count: int
    project_paths: list[str] = Field(default_factory=list)  # Unique project paths found


class SessionInfo(DomainModel):
    """
    Basic information about a discovered session.

    Lightweight summary for listing sessions without loading all records.
    """

    session_id: str
    project_path: str
    file_path: str  # Full path to {session_id}.jsonl
    record_count: int
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    has_summary: bool = False
    summary_text: str | None = None
    has_agents: bool = False
    agent_ids: list[str] = Field(default_factory=list)
