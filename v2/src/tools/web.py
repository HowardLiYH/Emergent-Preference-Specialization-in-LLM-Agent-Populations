"""
L4: Web access tool with sandboxing and rate limiting.
"""

import time
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
import logging

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ToolSecurityError(Exception):
    """Raised when a security policy is violated."""
    pass


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""
    pass


class WebTool(BaseTool):
    """
    L4: Web access tool (base, unsandboxed version).

    WARNING: Use SafeWebTool for production use.
    """

    name: str = "web"
    level: int = 4

    def __init__(self):
        """Initialize web tool."""
        self._http_client = None

    @property
    def http_client(self):
        """Lazy-load HTTP client."""
        if self._http_client is None:
            try:
                import httpx
                self._http_client = httpx.Client(timeout=30.0)
            except ImportError:
                logger.error("httpx not installed")
        return self._http_client

    def fetch(self, url: str) -> str:
        """
        Fetch content from a URL.

        Args:
            url: URL to fetch

        Returns:
            Response content as string
        """
        if self.http_client is None:
            raise RuntimeError("HTTP client not available")

        response = self.http_client.get(url)
        response.raise_for_status()
        return response.text

    def execute(
        self,
        query: str,
        context: Optional[dict] = None
    ) -> ToolResult:
        """
        Execute web request.

        Args:
            query: URL to fetch or search query
            context: Optional context with 'url' key

        Returns:
            ToolResult with fetched content
        """
        url = query
        if context and 'url' in context:
            url = context['url']

        try:
            content = self.fetch(url)

            # Truncate if too long
            max_chars = context.get('max_chars', 10000) if context else 10000
            if len(content) > max_chars:
                content = content[:max_chars] + "\n... (truncated)"

            return ToolResult(
                success=True,
                output=content,
                tokens_used=len(content.split())  # Rough estimate
            )

        except Exception as e:
            logger.error(f"Web fetch failed: {e}")
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )


class SafeWebTool(WebTool):
    """
    L4: Sandboxed web access tool.

    Security features:
    - Domain whitelist
    - Rate limiting
    - Response size limits
    - No arbitrary code execution
    """

    # Allowed domains for data access
    ALLOWED_DOMAINS: List[str] = [
        'api.weather.gov',
        'api.exchangerate.host',
        'data.gov',
        'api.github.com',
        'api.stackexchange.com',
        'en.wikipedia.org',
        'raw.githubusercontent.com',
    ]

    # Rate limit configuration
    MAX_REQUESTS_PER_MINUTE: int = 10

    def __init__(
        self,
        allowed_domains: Optional[List[str]] = None,
        max_requests_per_minute: int = 10
    ):
        """
        Initialize safe web tool.

        Args:
            allowed_domains: Custom allowed domains (or use default)
            max_requests_per_minute: Rate limit
        """
        super().__init__()

        if allowed_domains is not None:
            self.ALLOWED_DOMAINS = allowed_domains

        self.MAX_REQUESTS_PER_MINUTE = max_requests_per_minute
        self._request_timestamps: List[float] = []

    def _check_domain(self, url: str) -> None:
        """
        Check if domain is allowed.

        Raises:
            ToolSecurityError if domain not allowed
        """
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Check against whitelist
        allowed = any(
            domain == allowed_domain or domain.endswith('.' + allowed_domain)
            for allowed_domain in self.ALLOWED_DOMAINS
        )

        if not allowed:
            raise ToolSecurityError(
                f"Domain not allowed: {domain}. "
                f"Allowed domains: {self.ALLOWED_DOMAINS}"
            )

    def _check_rate_limit(self) -> None:
        """
        Check if rate limit allows another request.

        Raises:
            RateLimitError if rate limit exceeded
        """
        now = time.time()

        # Remove timestamps older than 1 minute
        self._request_timestamps = [
            ts for ts in self._request_timestamps
            if now - ts < 60
        ]

        if len(self._request_timestamps) >= self.MAX_REQUESTS_PER_MINUTE:
            wait_time = 60 - (now - self._request_timestamps[0])
            raise RateLimitError(
                f"Rate limit exceeded. Wait {wait_time:.1f}s before next request."
            )

        # Record this request
        self._request_timestamps.append(now)

    def fetch(self, url: str) -> str:
        """
        Fetch content from a URL with security checks.

        Args:
            url: URL to fetch

        Returns:
            Response content as string

        Raises:
            ToolSecurityError: If domain not allowed
            RateLimitError: If rate limit exceeded
        """
        # Security checks
        self._check_domain(url)
        self._check_rate_limit()

        # Proceed with fetch
        return super().fetch(url)

    def execute(
        self,
        query: str,
        context: Optional[dict] = None
    ) -> ToolResult:
        """
        Execute sandboxed web request.

        Args:
            query: URL to fetch
            context: Optional context

        Returns:
            ToolResult with fetched content or error
        """
        try:
            return super().execute(query, context)
        except ToolSecurityError as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Security error: {e}"
            )
        except RateLimitError as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Rate limit: {e}"
            )

    def get_weather(self, latitude: float, longitude: float) -> ToolResult:
        """
        Get weather for a location (uses weather.gov API).

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate

        Returns:
            ToolResult with weather data
        """
        # Get gridpoint
        url = f"https://api.weather.gov/points/{latitude},{longitude}"

        try:
            content = self.fetch(url)
            # Parse and extract forecast URL, then fetch forecast
            import json
            data = json.loads(content)
            forecast_url = data['properties']['forecast']

            forecast_content = self.fetch(forecast_url)
            forecast_data = json.loads(forecast_content)

            # Format forecast
            periods = forecast_data['properties']['periods'][:4]
            forecast_text = "\n".join([
                f"{p['name']}: {p['detailedForecast']}"
                for p in periods
            ])

            return ToolResult(
                success=True,
                output=forecast_text,
                tokens_used=len(forecast_text.split())
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )

    def get_exchange_rate(
        self,
        base_currency: str = "USD",
        target_currency: str = "EUR"
    ) -> ToolResult:
        """
        Get current exchange rate.

        Args:
            base_currency: Base currency code
            target_currency: Target currency code

        Returns:
            ToolResult with exchange rate
        """
        url = f"https://api.exchangerate.host/latest?base={base_currency}&symbols={target_currency}"

        try:
            content = self.fetch(url)
            import json
            data = json.loads(content)

            rate = data['rates'].get(target_currency)
            if rate:
                return ToolResult(
                    success=True,
                    output=f"1 {base_currency} = {rate} {target_currency}",
                    tokens_used=10
                )
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Currency {target_currency} not found"
                )

        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )
