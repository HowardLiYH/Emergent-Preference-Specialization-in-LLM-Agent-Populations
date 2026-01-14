"""
Core competition mechanics, batching, and context management.
"""

from .population import Population
from .curriculum import TaskCurriculum
from .batch_processor import BatchProcessor
from .llm_client import LLMClient
from .context_manager import ContextManager

__all__ = [
    "Population",
    "TaskCurriculum",
    "BatchProcessor",
    "LLMClient",
    "ContextManager",
]
