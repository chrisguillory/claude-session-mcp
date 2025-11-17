"""
MCP server configuration.

Extends base configuration with MCP-specific settings.
"""

from __future__ import annotations

from src.config.base import BaseSessionSettings, lazy_settings


class McpServerSettings(BaseSessionSettings):
    """MCP server-specific configuration."""

    pass  # Empty for now, room for MCP-specific settings


# Module-level singleton (lazy-loaded)
settings = lazy_settings(McpServerSettings)
