"""
Statsig feature flag endpoint capture classes.

This module contains capture wrappers for Statsig endpoints:
- /v1/rgstr - Event registration/logging
- /v1/initialize - Feature flag initialization
"""

from __future__ import annotations

from typing import Literal

from src.schemas.captures.base import StatsigRequestCapture, StatsigResponseCapture
from src.schemas.cc_internal_api import (
    StatsigInitializeRequest,
    StatsigInitializeResponse,
    StatsigRegisterRequest,
    StatsigRegisterResponse,
)

# ==============================================================================
# Register (/v1/rgstr)
# ==============================================================================


class StatsigRegisterRequestCapture(StatsigRequestCapture):
    """Captured POST /v1/rgstr request (event logging)."""

    method: Literal['POST']
    body: StatsigRegisterRequest


class StatsigRegisterResponseCapture(StatsigResponseCapture):
    """Captured POST /v1/rgstr response (202 Accepted)."""

    body: StatsigRegisterResponse


# ==============================================================================
# Initialize (/v1/initialize)
# ==============================================================================


class StatsigInitializeRequestCapture(StatsigRequestCapture):
    """Captured POST /v1/initialize request (feature flag init)."""

    method: Literal['POST']
    body: StatsigInitializeRequest


class StatsigInitializeResponseCapture(StatsigResponseCapture):
    """
    Captured POST /v1/initialize response.

    Three response types (discriminated union):
    - 204 No Content: Empty response when no updates (EmptyBody)
    - 200 OK: Full feature flags (StatsigInitializeFullResponse)
    - 200 OK: Delta updates (StatsigInitializeDeltaResponse)
    """

    body: StatsigInitializeResponse
