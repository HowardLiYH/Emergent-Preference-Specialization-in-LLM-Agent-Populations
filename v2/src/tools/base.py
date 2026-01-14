"""
Base tool interface and L0 (no tool) implementation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    tokens_used: int = 0


class BaseTool(ABC):
    """Abstract base class for all tools."""

    name: str = "base"
    level: int = -1

    @abstractmethod
    def execute(self, query: str, context: Optional[dict] = None) -> ToolResult:
        """Execute the tool with the given query."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(level={self.level})"


class L0Tool(BaseTool):
    """
    L0: Base LLM - No tools, pure text completion.

    This represents the baseline capability where the agent
    can only use the LLM's inherent knowledge without any
    external tool access.
    """

    name: str = "base_llm"
    level: int = 0

    def __init__(self, llm_client=None):
        """
        Initialize L0 tool.

        Args:
            llm_client: LLM client for text completion
        """
        self.llm_client = llm_client

    def execute(self, query: str, context: Optional[dict] = None) -> ToolResult:
        """
        Execute pure text completion without any tools.

        Args:
            query: The query to answer
            context: Optional context (memories, etc.)

        Returns:
            ToolResult with the LLM's response
        """
        if self.llm_client is None:
            return ToolResult(
                success=False,
                output=None,
                error="LLM client not configured"
            )

        try:
            # Build prompt with context if available
            prompt = query
            if context and context.get("memories"):
                memory_text = "\n".join(context["memories"])
                prompt = f"Relevant context:\n{memory_text}\n\nQuery: {query}"

            # Call LLM
            response = self.llm_client.generate(prompt)

            return ToolResult(
                success=True,
                output=response.text,
                tokens_used=response.tokens_used
            )
        except Exception as e:
            logger.error(f"L0 execution failed: {e}")
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )


# Tool level constants
TOOL_LEVELS = {
    0: "Base LLM (no tools)",
    1: "Python execution",
    2: "Vision/multi-modal",
    3: "RAG/retrieval",
    4: "Web access (sandboxed)",
}


def get_tool_description(level: int) -> str:
    """Get description for a tool level."""
    return TOOL_LEVELS.get(level, f"Unknown level {level}")
