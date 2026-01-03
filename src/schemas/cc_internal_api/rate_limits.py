"""
Rate limit header schemas for Claude Code internal API.

Claude Code receives comprehensive rate limit information in response headers.
This module provides schemas for parsing and extracting these headers.

Headers follow the pattern: anthropic-ratelimit-unified-{component}
"""

from __future__ import annotations

from typing import Literal

from src.schemas.cc_internal_api.base import PermissiveModel

# ==============================================================================
# Rate Limit Status Types
# ==============================================================================

RateLimitStatus = Literal['allowed', 'throttled']
OverageStatus = Literal['allowed', 'rejected']  # Different from RateLimitStatus
FallbackStatus = Literal['available', 'unavailable']
RepresentativeClaim = Literal['five_hour', 'seven_day']


# ==============================================================================
# Rate Limit Window
# ==============================================================================


class RateLimitWindow(PermissiveModel):
    """
    Rate limit status for a specific time window (5h or 7d).

    VALIDATION STATUS: VALIDATED
    Observed in response headers.
    """

    status: RateLimitStatus
    reset: int  # Unix timestamp when window resets
    utilization: float  # 0.0 to 1.0+ (can exceed 1.0 if over limit)


# ==============================================================================
# Unified Rate Limit
# ==============================================================================


class UnifiedRateLimit(PermissiveModel):
    """
    Complete rate limit information extracted from response headers.

    VALIDATION STATUS: VALIDATED
    Observed in all /v1/messages response headers.

    Header prefix: anthropic-ratelimit-unified-

    The unified rate limit system tracks usage across two windows:
    - 5-hour window: Short-term burst limiting
    - 7-day window: Long-term quota management

    Features:
    - Fallback capacity for burst handling
    - Overage mechanism for organizations
    - Representative claim indicates which window is limiting
    """

    # Overall status
    status: RateLimitStatus
    reset: int  # Overall reset timestamp

    # 5-hour window
    h5: RateLimitWindow

    # 7-day window
    d7: RateLimitWindow

    # Metadata
    representative_claim: RepresentativeClaim  # Which window is limiting
    fallback: FallbackStatus  # Burst capacity availability
    fallback_percentage: float  # Additional capacity (e.g., 0.5 = 50%)

    # Overage (organizational)
    overage_status: OverageStatus
    overage_disabled_reason: str | None = None

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> UnifiedRateLimit:
        """
        Extract rate limit information from response headers.

        Args:
            headers: Response headers dict (case-insensitive keys)

        Returns:
            UnifiedRateLimit instance with parsed values

        Raises:
            KeyError: If required headers are missing
            ValueError: If header values are malformed
        """
        # Normalize header keys to lowercase
        h = {k.lower(): v for k, v in headers.items()}
        prefix = 'anthropic-ratelimit-unified-'

        def get(key: str) -> str:
            return h[f'{prefix}{key}']

        def get_optional(key: str) -> str | None:
            return h.get(f'{prefix}{key}')

        return cls(
            status=get('status'),  # type: ignore[arg-type]
            reset=int(get('reset')),
            h5=RateLimitWindow(
                status=get('5h-status'),  # type: ignore[arg-type]
                reset=int(get('5h-reset')),
                utilization=float(get('5h-utilization')),
            ),
            d7=RateLimitWindow(
                status=get('7d-status'),  # type: ignore[arg-type]
                reset=int(get('7d-reset')),
                utilization=float(get('7d-utilization')),
            ),
            representative_claim=get('representative-claim'),  # type: ignore[arg-type]
            fallback=get('fallback'),  # type: ignore[arg-type]
            fallback_percentage=float(get('fallback-percentage')),
            overage_status=get('overage-status'),  # type: ignore[arg-type]
            overage_disabled_reason=get_optional('overage-disabled-reason'),
        )

    @classmethod
    def from_headers_safe(cls, headers: dict[str, str]) -> UnifiedRateLimit | None:
        """
        Extract rate limit information, returning None if headers are missing.

        This is useful when processing non-Messages API responses that don't
        include rate limit headers.
        """
        try:
            return cls.from_headers(headers)
        except (KeyError, ValueError):
            return None
