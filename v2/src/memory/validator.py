"""
Memory quality validation.

Filters suspicious or overly broad memory entries.
"""

from typing import Tuple, List, Optional
import re
import logging

logger = logging.getLogger(__name__)


# Suspicious patterns that indicate overly broad or adversarial content
SUSPICIOUS_PATTERNS: List[str] = [
    # Overly broad claims
    'all questions',
    'all tasks',
    'always works',
    'always correct',
    'never fails',
    'never wrong',
    'every time',
    'every question',
    'universal solution',
    'works for everything',
    'always use this',
    '100% of the time',

    # Potentially adversarial
    'ignore this',
    'disregard',
    'forget previous',
    'override',
    'secret instruction',

    # Too vague to be useful
    'just do it',
    'try harder',
    'be smarter',
    'think more',
]

# Minimum content requirements
MIN_CONTENT_LENGTH = 20
MAX_CONTENT_LENGTH = 2000


class MemoryValidationError(Exception):
    """Raised when memory validation fails."""
    pass


def validate_memory_entry(
    content: str,
    regime: Optional[str] = None,
    strict: bool = True
) -> Tuple[bool, str]:
    """
    Validate a memory entry for quality and safety.

    Args:
        content: Memory content to validate
        regime: Optional regime context
        strict: If True, reject on any issue; if False, warn only

    Returns:
        Tuple of (is_valid, reason)
    """
    # Check length
    if len(content.strip()) < MIN_CONTENT_LENGTH:
        return False, f"Content too short ({len(content)} chars, min {MIN_CONTENT_LENGTH})"

    if len(content) > MAX_CONTENT_LENGTH:
        return False, f"Content too long ({len(content)} chars, max {MAX_CONTENT_LENGTH})"

    # Check for suspicious patterns
    content_lower = content.lower()
    for pattern in SUSPICIOUS_PATTERNS:
        if pattern in content_lower:
            msg = f"Suspicious pattern detected: '{pattern}'"
            logger.warning(msg)
            if strict:
                return False, msg

    # Check for excessive repetition
    words = content_lower.split()
    if words:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return False, f"Content too repetitive (unique word ratio: {unique_ratio:.2f})"

    # Check for all caps (potential spam)
    if content.isupper() and len(content) > 50:
        return False, "Content is all uppercase (potential spam)"

    return True, "OK"


class MemoryValidator:
    """
    Configurable memory validator.
    """

    def __init__(
        self,
        custom_patterns: Optional[List[str]] = None,
        min_length: int = MIN_CONTENT_LENGTH,
        max_length: int = MAX_CONTENT_LENGTH,
        strict: bool = True
    ):
        """
        Initialize memory validator.

        Args:
            custom_patterns: Additional suspicious patterns
            min_length: Minimum content length
            max_length: Maximum content length
            strict: Whether to be strict about validation
        """
        self.patterns = SUSPICIOUS_PATTERNS.copy()
        if custom_patterns:
            self.patterns.extend(custom_patterns)

        self.min_length = min_length
        self.max_length = max_length
        self.strict = strict

        # Statistics
        self.total_validated = 0
        self.total_rejected = 0
        self.rejection_reasons: List[str] = []

    def validate(
        self,
        content: str,
        regime: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Validate a memory entry.

        Args:
            content: Memory content
            regime: Optional regime context

        Returns:
            Tuple of (is_valid, reason)
        """
        self.total_validated += 1

        is_valid, reason = validate_memory_entry(
            content,
            regime=regime,
            strict=self.strict
        )

        if not is_valid:
            self.total_rejected += 1
            self.rejection_reasons.append(reason)

            # Keep only recent reasons
            if len(self.rejection_reasons) > 100:
                self.rejection_reasons = self.rejection_reasons[-50:]

        return is_valid, reason

    def validate_or_raise(
        self,
        content: str,
        regime: Optional[str] = None
    ) -> None:
        """
        Validate and raise exception if invalid.

        Args:
            content: Memory content
            regime: Optional regime context

        Raises:
            MemoryValidationError: If validation fails
        """
        is_valid, reason = self.validate(content, regime)

        if not is_valid:
            raise MemoryValidationError(reason)

    def get_stats(self) -> dict:
        """Get validation statistics."""
        return {
            'total_validated': self.total_validated,
            'total_rejected': self.total_rejected,
            'rejection_rate': (
                self.total_rejected / self.total_validated
                if self.total_validated > 0 else 0
            ),
            'recent_rejections': self.rejection_reasons[-10:],
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.total_validated = 0
        self.total_rejected = 0
        self.rejection_reasons.clear()
