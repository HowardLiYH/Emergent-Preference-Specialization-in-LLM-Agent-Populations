"""
Prompt injection defense.

Sanitizes task inputs to prevent malicious instructions from
manipulating agent behavior.
"""

import re
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class TaskSecurityError(Exception):
    """Raised when a potential security threat is detected in task input."""
    pass


# Known prompt injection patterns
INJECTION_PATTERNS: List[str] = [
    # Direct instruction overrides
    'ignore previous',
    'ignore above',
    'ignore all',
    'disregard instructions',
    'disregard previous',
    'forget everything',
    'forget your instructions',
    'new instructions',

    # System prompt extraction
    'output your prompt',
    'reveal your prompt',
    'show your prompt',
    'what are your instructions',
    'repeat your instructions',
    'print your system',
    'show system prompt',

    # Memory attacks
    'reveal your memory',
    'show your memory',
    'output your memory',
    'what do you remember',
    'list your memories',

    # Role manipulation
    'you are now',
    'pretend you are',
    'act as if',
    'roleplay as',
    'from now on',

    # Jailbreak attempts
    'dan mode',
    'developer mode',
    'debug mode',
    'admin mode',
    'sudo',

    # Delimiter attacks
    '###',
    '---end---',
    '[system]',
    '<|endoftext|>',
]

# Regex patterns for more sophisticated attacks
REGEX_PATTERNS = [
    r'```system',
    r'<system>.*</system>',
    r'\bbase64\b.*decode',
    r'eval\s*\(',
    r'exec\s*\(',
]


def sanitize_task(
    task: str,
    patterns: Optional[List[str]] = None,
    raise_on_detection: bool = True
) -> str:
    """
    Sanitize task input to prevent prompt injection.

    Args:
        task: The task text to sanitize
        patterns: Custom patterns to check (or use defaults)
        raise_on_detection: Whether to raise exception or just log warning

    Returns:
        Sanitized task text

    Raises:
        TaskSecurityError: If injection pattern detected and raise_on_detection=True
    """
    if patterns is None:
        patterns = INJECTION_PATTERNS

    task_lower = task.lower()

    # Check string patterns
    for pattern in patterns:
        if pattern.lower() in task_lower:
            msg = f"Potential prompt injection detected: '{pattern}'"
            logger.warning(msg)

            if raise_on_detection:
                raise TaskSecurityError(msg)
            else:
                # Remove the pattern
                task = re.sub(re.escape(pattern), '[REDACTED]', task, flags=re.IGNORECASE)

    # Check regex patterns
    for regex_pattern in REGEX_PATTERNS:
        if re.search(regex_pattern, task, re.IGNORECASE):
            msg = f"Potential code injection detected: pattern '{regex_pattern}'"
            logger.warning(msg)

            if raise_on_detection:
                raise TaskSecurityError(msg)
            else:
                task = re.sub(regex_pattern, '[REDACTED]', task, flags=re.IGNORECASE)

    return task


def is_safe_task(task: str) -> bool:
    """
    Check if a task is safe without raising exceptions.

    Args:
        task: The task text to check

    Returns:
        True if safe, False if potentially malicious
    """
    try:
        sanitize_task(task, raise_on_detection=True)
        return True
    except TaskSecurityError:
        return False


def escape_for_prompt(text: str) -> str:
    """
    Escape text to be safely included in a prompt.

    Args:
        text: Text to escape

    Returns:
        Escaped text
    """
    # Escape common delimiter characters
    text = text.replace('```', '`​`​`')  # Zero-width space between backticks
    text = text.replace('###', '#​#​#')
    text = text.replace('---', '-​-​-')

    return text


class InputSanitizer:
    """
    Configurable input sanitizer with custom patterns.
    """

    def __init__(
        self,
        custom_patterns: Optional[List[str]] = None,
        block_mode: bool = True,
        log_detections: bool = True
    ):
        """
        Initialize sanitizer.

        Args:
            custom_patterns: Additional patterns to check
            block_mode: If True, raise exceptions; if False, redact
            log_detections: Whether to log detected issues
        """
        self.patterns = INJECTION_PATTERNS.copy()
        if custom_patterns:
            self.patterns.extend(custom_patterns)

        self.block_mode = block_mode
        self.log_detections = log_detections

        # Track detections for monitoring
        self.detection_count = 0
        self.detected_patterns: List[str] = []

    def sanitize(self, task: str) -> str:
        """
        Sanitize a task input.

        Args:
            task: Task text

        Returns:
            Sanitized text

        Raises:
            TaskSecurityError: If block_mode=True and injection detected
        """
        try:
            return sanitize_task(
                task,
                patterns=self.patterns,
                raise_on_detection=self.block_mode
            )
        except TaskSecurityError as e:
            self.detection_count += 1
            self.detected_patterns.append(str(e))
            raise

    def get_stats(self) -> dict:
        """Get detection statistics."""
        return {
            'detection_count': self.detection_count,
            'detected_patterns': self.detected_patterns[-10:],  # Last 10
        }
