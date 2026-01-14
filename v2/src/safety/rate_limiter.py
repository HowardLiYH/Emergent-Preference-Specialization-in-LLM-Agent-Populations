"""
Rate limiting for tool access.

Prevents abuse and ensures fair resource usage.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional
import logging
import threading

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""
    pass


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    max_requests: int = 60
    window_seconds: int = 60
    burst_limit: Optional[int] = None  # Max requests in short burst
    burst_window: int = 5  # Burst window in seconds


class RateLimiter:
    """
    Token bucket rate limiter.

    Features:
    - Per-resource rate limiting
    - Burst allowance
    - Thread-safe
    """

    def __init__(
        self,
        default_config: Optional[RateLimitConfig] = None
    ):
        """
        Initialize rate limiter.

        Args:
            default_config: Default rate limit configuration
        """
        self.default_config = default_config or RateLimitConfig()
        self.resource_configs: Dict[str, RateLimitConfig] = {}

        # Request timestamps per resource
        self._timestamps: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()

    def configure_resource(
        self,
        resource: str,
        config: RateLimitConfig
    ) -> None:
        """
        Set rate limit config for a specific resource.

        Args:
            resource: Resource identifier
            config: Rate limit configuration
        """
        self.resource_configs[resource] = config

    def _get_config(self, resource: str) -> RateLimitConfig:
        """Get config for a resource."""
        return self.resource_configs.get(resource, self.default_config)

    def _clean_old_timestamps(
        self,
        resource: str,
        window_seconds: int
    ) -> None:
        """Remove timestamps older than window."""
        now = time.time()
        cutoff = now - window_seconds
        self._timestamps[resource] = [
            ts for ts in self._timestamps[resource]
            if ts > cutoff
        ]

    def check(self, resource: str) -> bool:
        """
        Check if request is allowed without consuming quota.

        Args:
            resource: Resource identifier

        Returns:
            True if request would be allowed
        """
        config = self._get_config(resource)

        with self._lock:
            self._clean_old_timestamps(resource, config.window_seconds)
            current_count = len(self._timestamps[resource])

            return current_count < config.max_requests

    def acquire(self, resource: str) -> None:
        """
        Acquire a rate limit token.

        Args:
            resource: Resource identifier

        Raises:
            RateLimitError: If rate limit exceeded
        """
        config = self._get_config(resource)

        with self._lock:
            now = time.time()
            self._clean_old_timestamps(resource, config.window_seconds)

            current_count = len(self._timestamps[resource])

            if current_count >= config.max_requests:
                # Calculate wait time
                oldest = self._timestamps[resource][0]
                wait_time = config.window_seconds - (now - oldest)

                raise RateLimitError(
                    f"Rate limit exceeded for {resource}. "
                    f"Wait {wait_time:.1f}s before next request. "
                    f"Limit: {config.max_requests}/{config.window_seconds}s"
                )

            # Check burst limit if configured
            if config.burst_limit:
                burst_cutoff = now - config.burst_window
                burst_count = sum(
                    1 for ts in self._timestamps[resource]
                    if ts > burst_cutoff
                )
                if burst_count >= config.burst_limit:
                    raise RateLimitError(
                        f"Burst limit exceeded for {resource}. "
                        f"Limit: {config.burst_limit}/{config.burst_window}s"
                    )

            # Record request
            self._timestamps[resource].append(now)

            logger.debug(
                f"Rate limit acquired for {resource}: "
                f"{current_count + 1}/{config.max_requests}"
            )

    def get_remaining(self, resource: str) -> int:
        """
        Get remaining requests in current window.

        Args:
            resource: Resource identifier

        Returns:
            Number of remaining requests
        """
        config = self._get_config(resource)

        with self._lock:
            self._clean_old_timestamps(resource, config.window_seconds)
            return config.max_requests - len(self._timestamps[resource])

    def get_reset_time(self, resource: str) -> float:
        """
        Get seconds until rate limit resets.

        Args:
            resource: Resource identifier

        Returns:
            Seconds until oldest request expires
        """
        config = self._get_config(resource)

        with self._lock:
            if not self._timestamps[resource]:
                return 0.0

            oldest = self._timestamps[resource][0]
            now = time.time()
            return max(0, config.window_seconds - (now - oldest))

    def reset(self, resource: Optional[str] = None) -> None:
        """
        Reset rate limit counters.

        Args:
            resource: Specific resource to reset, or None for all
        """
        with self._lock:
            if resource:
                self._timestamps[resource] = []
            else:
                self._timestamps.clear()


# Pre-configured rate limiters for different tools
class ToolRateLimiters:
    """Pre-configured rate limiters for tools."""

    # L4 Web tool - strict limits
    WEB = RateLimiter(RateLimitConfig(
        max_requests=10,
        window_seconds=60,
        burst_limit=3,
        burst_window=5
    ))

    # LLM API calls - moderate limits
    LLM = RateLimiter(RateLimitConfig(
        max_requests=60,
        window_seconds=60,
        burst_limit=10,
        burst_window=10
    ))

    # Python execution - moderate limits
    PYTHON = RateLimiter(RateLimitConfig(
        max_requests=30,
        window_seconds=60
    ))
