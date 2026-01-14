"""
Context length management for LLM prompts.

Manages token budgets for memory, task, and system prompts.
"""

from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


# Context budget configuration
CONTEXT_BUDGET = {
    'max_memory_tokens': 2000,    # Budget for retrieved memories
    'max_task_tokens': 1000,      # Budget for task
    'max_system_tokens': 500,     # Budget for system prompt
    'max_total_tokens': 4000,     # Total context budget
}


def count_tokens(text: str, method: str = 'tiktoken') -> int:
    """
    Count tokens in text.

    Args:
        text: Text to count tokens for
        method: Counting method ('tiktoken', 'words', 'chars')

    Returns:
        Estimated token count
    """
    if method == 'tiktoken':
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            logger.warning("tiktoken not available, falling back to word count")
            method = 'words'

    if method == 'words':
        # Rough approximation: 1 token ≈ 0.75 words
        return int(len(text.split()) / 0.75)

    if method == 'chars':
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4

    return len(text.split())


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    method: str = 'tiktoken'
) -> str:
    """
    Truncate text to fit within token budget.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed
        method: Token counting method

    Returns:
        Truncated text
    """
    current_tokens = count_tokens(text, method)

    if current_tokens <= max_tokens:
        return text

    # Binary search for truncation point
    words = text.split()
    low, high = 0, len(words)

    while low < high:
        mid = (low + high + 1) // 2
        truncated = ' '.join(words[:mid])

        if count_tokens(truncated, method) <= max_tokens:
            low = mid
        else:
            high = mid - 1

    if low == 0:
        return ""

    return ' '.join(words[:low]) + "..."


class ContextManager:
    """
    Manages context length for LLM prompts.

    Ensures prompts fit within token limits by:
    - Allocating budgets to different components
    - Truncating when necessary
    - Prioritizing recent/relevant content
    """

    def __init__(
        self,
        budget: Optional[Dict[str, int]] = None,
        token_method: str = 'tiktoken'
    ):
        """
        Initialize context manager.

        Args:
            budget: Custom token budget configuration
            token_method: Token counting method
        """
        self.budget = budget or CONTEXT_BUDGET.copy()
        self.token_method = token_method

    def construct_prompt(
        self,
        task: str,
        memories: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Construct a prompt within token budget.

        Args:
            task: Task description
            memories: List of memory strings
            system_prompt: System prompt

        Returns:
            Constructed prompt fitting within budget
        """
        parts = []
        tokens_used = 0

        # 1. System prompt (if provided)
        if system_prompt:
            system_truncated = truncate_to_tokens(
                system_prompt,
                self.budget['max_system_tokens'],
                self.token_method
            )
            parts.append(system_truncated)
            tokens_used += count_tokens(system_truncated, self.token_method)

        # 2. Memories (prioritize by order - most relevant first)
        if memories:
            memory_budget = self.budget['max_memory_tokens']
            memory_parts = []
            memory_tokens = 0

            for memory in memories:
                mem_tokens = count_tokens(memory, self.token_method)

                if memory_tokens + mem_tokens > memory_budget:
                    # Try to fit partial
                    remaining = memory_budget - memory_tokens
                    if remaining > 50:  # Only include if meaningful
                        truncated = truncate_to_tokens(
                            memory, remaining, self.token_method
                        )
                        memory_parts.append(truncated)
                    break

                memory_parts.append(memory)
                memory_tokens += mem_tokens

            if memory_parts:
                memory_section = "Relevant context:\n" + "\n---\n".join(memory_parts)
                parts.append(memory_section)
                tokens_used += count_tokens(memory_section, self.token_method)

        # 3. Task
        remaining_budget = self.budget['max_total_tokens'] - tokens_used
        task_budget = min(remaining_budget, self.budget['max_task_tokens'])

        task_truncated = truncate_to_tokens(
            task, task_budget, self.token_method
        )

        parts.append(f"Task: {task_truncated}")
        tokens_used += count_tokens(task_truncated, self.token_method)

        prompt = "\n\n".join(parts)

        logger.debug(f"Constructed prompt with {tokens_used} tokens")
        return prompt

    def get_remaining_budget(self, used_tokens: int) -> int:
        """Get remaining token budget."""
        return max(0, self.budget['max_total_tokens'] - used_tokens)

    def validate_prompt(self, prompt: str) -> bool:
        """Check if prompt is within budget."""
        tokens = count_tokens(prompt, self.token_method)
        return tokens <= self.budget['max_total_tokens']

    def get_token_breakdown(
        self,
        task: str,
        memories: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Get token breakdown for prompt components.

        Returns dict with token counts for each component.
        """
        breakdown = {
            'system': 0,
            'memories': 0,
            'task': 0,
            'total': 0,
        }

        if system_prompt:
            breakdown['system'] = count_tokens(system_prompt, self.token_method)

        if memories:
            breakdown['memories'] = sum(
                count_tokens(m, self.token_method) for m in memories
            )

        breakdown['task'] = count_tokens(task, self.token_method)
        breakdown['total'] = sum(breakdown.values())

        return breakdown
