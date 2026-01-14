"""
Domain whitelist for web access.

Controls which domains agents can access via the L4 web tool.
"""

from typing import List, Set, Optional
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


class ToolSecurityError(Exception):
    """Raised when a security policy is violated."""
    pass


# Default allowed domains for safe data access
DEFAULT_ALLOWED_DOMAINS: List[str] = [
    # Government data APIs
    'api.weather.gov',
    'data.gov',
    'api.census.gov',

    # Financial data
    'api.exchangerate.host',
    'api.coindesk.com',

    # Reference/Knowledge
    'en.wikipedia.org',
    'api.wikimedia.org',

    # Developer resources
    'api.github.com',
    'raw.githubusercontent.com',
    'api.stackexchange.com',

    # Science/Academic
    'api.crossref.org',
    'api.semanticscholar.org',
]

# Explicitly blocked domains (even if pattern matches)
BLOCKED_DOMAINS: List[str] = [
    'localhost',
    '127.0.0.1',
    '0.0.0.0',
    '::1',
    'internal',
    'private',
    'admin',
]


class DomainWhitelist:
    """
    Domain whitelist for controlling web access.

    Features:
    - Explicit allow/block lists
    - Subdomain matching
    - Logging and monitoring
    """

    def __init__(
        self,
        allowed_domains: Optional[List[str]] = None,
        blocked_domains: Optional[List[str]] = None,
        allow_subdomains: bool = True
    ):
        """
        Initialize domain whitelist.

        Args:
            allowed_domains: List of allowed domains
            blocked_domains: List of blocked domains (overrides allowed)
            allow_subdomains: Whether to allow subdomains of allowed domains
        """
        self.allowed_domains: Set[str] = set(
            allowed_domains or DEFAULT_ALLOWED_DOMAINS
        )
        self.blocked_domains: Set[str] = set(
            blocked_domains or BLOCKED_DOMAINS
        )
        self.allow_subdomains = allow_subdomains

        # Monitoring
        self.access_log: List[dict] = []
        self.blocked_attempts: int = 0

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Remove port if present
        if ':' in domain:
            domain = domain.split(':')[0]

        return domain

    def _is_subdomain(self, domain: str, parent: str) -> bool:
        """Check if domain is a subdomain of parent."""
        return domain == parent or domain.endswith('.' + parent)

    def is_allowed(self, url: str) -> bool:
        """
        Check if URL is allowed.

        Args:
            url: URL to check

        Returns:
            True if allowed
        """
        domain = self._extract_domain(url)

        # Check blocked list first (takes priority)
        for blocked in self.blocked_domains:
            if self._is_subdomain(domain, blocked):
                return False

        # Check allowed list
        for allowed in self.allowed_domains:
            if self.allow_subdomains:
                if self._is_subdomain(domain, allowed):
                    return True
            else:
                if domain == allowed:
                    return True

        return False

    def check(self, url: str) -> None:
        """
        Check if URL is allowed, raise exception if not.

        Args:
            url: URL to check

        Raises:
            ToolSecurityError: If domain not allowed
        """
        domain = self._extract_domain(url)

        # Log access attempt
        self.access_log.append({
            'url': url,
            'domain': domain,
            'allowed': self.is_allowed(url)
        })

        # Keep log bounded
        if len(self.access_log) > 1000:
            self.access_log = self.access_log[-500:]

        if not self.is_allowed(url):
            self.blocked_attempts += 1
            logger.warning(f"Blocked access to domain: {domain}")
            raise ToolSecurityError(
                f"Domain not allowed: {domain}. "
                f"Allowed domains: {sorted(self.allowed_domains)}"
            )

    def add_domain(self, domain: str) -> None:
        """
        Add a domain to the allowed list.

        Args:
            domain: Domain to allow
        """
        self.allowed_domains.add(domain.lower())
        logger.info(f"Added domain to whitelist: {domain}")

    def remove_domain(self, domain: str) -> None:
        """
        Remove a domain from the allowed list.

        Args:
            domain: Domain to remove
        """
        self.allowed_domains.discard(domain.lower())
        logger.info(f"Removed domain from whitelist: {domain}")

    def block_domain(self, domain: str) -> None:
        """
        Add a domain to the blocked list.

        Args:
            domain: Domain to block
        """
        self.blocked_domains.add(domain.lower())
        logger.info(f"Blocked domain: {domain}")

    def get_stats(self) -> dict:
        """Get access statistics."""
        return {
            'allowed_domains': len(self.allowed_domains),
            'blocked_domains': len(self.blocked_domains),
            'total_accesses': len(self.access_log),
            'blocked_attempts': self.blocked_attempts,
            'recent_domains': list(set(
                log['domain'] for log in self.access_log[-50:]
            ))
        }


# Default whitelist instance
default_whitelist = DomainWhitelist()
