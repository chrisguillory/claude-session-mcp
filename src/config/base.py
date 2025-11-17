"""
Base configuration for Claude Session services.

Shared settings and helper functions for all service types (MCP, HTTP).
"""

from __future__ import annotations

import os
import pathlib
from typing import TypeVar

import pydantic
import pydantic_settings
import lazy_object_proxy


T = TypeVar('T', bound='BaseSessionSettings')


class BaseSessionSettings(pydantic_settings.BaseSettings):
    """Shared configuration across all session services (MCP, HTTP)."""

    model_config = pydantic_settings.SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True,  # Fail fast on misconfiguration
        extra='forbid',  # Reject unknown environment variables
    )

    # Application metadata
    APP_NAME: str = 'claude-session-mcp'
    VERSION: str = '0.1.0'

    # Compression settings (not overrideable at call-time)
    COMPRESSION_LEVEL: int = 3  # 0-9, zstd compression level (3 = balanced)

    @pydantic.field_validator('COMPRESSION_LEVEL')
    @classmethod
    def validate_compression_level(cls, v: int) -> int:
        """Validate compression level is within zstd bounds."""
        if not 0 <= v <= 9:
            raise ValueError('COMPRESSION_LEVEL must be between 0-9')
        return v


def get_settings(settings_class: type[T], env_file: str | None = None) -> T:
    """
    Factory for creating settings with dynamic .env file loading.

    LOAD_ENV_FILE environment variable specifies custom .env file path.
    When unset (production), loads from environment variables only.

    Args:
        settings_class: Settings class to instantiate
        env_file: Optional path to .env file (overrides LOAD_ENV_FILE)

    Returns:
        Settings instance

    Raises:
        FileNotFoundError: If specified .env file doesn't exist
    """
    env_file_path = env_file or os.getenv('LOAD_ENV_FILE')

    if not env_file_path:
        return settings_class()  # No .env file, load from environment only

    resolved_path = pathlib.Path(env_file_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f'Environment file not found: {resolved_path}')

    return settings_class(_env_file=resolved_path)


def lazy_settings(settings_class: type[T]) -> T:
    """
    Lazy settings - defers instantiation until first access.

    Args:
        settings_class: Settings class to instantiate

    Returns:
        Proxy that instantiates settings on first access
    """
    return lazy_object_proxy.Proxy(lambda: get_settings(settings_class))
