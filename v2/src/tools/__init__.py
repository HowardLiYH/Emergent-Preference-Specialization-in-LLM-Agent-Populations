"""
Tool-based capability levels (L0-L4).

L0: Base LLM (no tools)
L1: Python execution
L2: Vision/multi-modal
L3: RAG/retrieval
L4: Web access (sandboxed)
"""

from .base import BaseTool, L0Tool
from .python_tool import PythonTool
from .vision import VisionTool
from .rag import RAGTool
from .web import WebTool, SafeWebTool
from .selection import ToolSelectionPolicy
from .agent import Agent

__all__ = [
    "BaseTool",
    "L0Tool",
    "PythonTool",
    "VisionTool",
    "RAGTool",
    "WebTool",
    "SafeWebTool",
    "ToolSelectionPolicy",
    "Agent",
]
