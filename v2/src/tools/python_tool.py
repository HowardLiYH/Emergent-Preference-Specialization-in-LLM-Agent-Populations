"""
L1: Python execution tool with sandboxing.
"""

import ast
import sys
import traceback
from io import StringIO
from typing import Optional
import logging
import signal
from contextlib import contextmanager

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Raised when code execution times out."""
    pass


@contextmanager
def timeout(seconds: int):
    """Context manager for timing out code execution."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Execution timed out after {seconds} seconds")

    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class PythonTool(BaseTool):
    """
    L1: Python execution tool.

    Executes Python code in a sandboxed environment with:
    - Timeout protection
    - Limited builtins
    - No file system access
    - No network access
    """

    name: str = "python"
    level: int = 1

    # Safe builtins that can be used
    SAFE_BUILTINS = {
        'abs', 'all', 'any', 'bin', 'bool', 'chr', 'dict', 'divmod',
        'enumerate', 'filter', 'float', 'format', 'frozenset', 'hex',
        'int', 'isinstance', 'issubclass', 'iter', 'len', 'list',
        'map', 'max', 'min', 'next', 'oct', 'ord', 'pow', 'print',
        'range', 'repr', 'reversed', 'round', 'set', 'slice', 'sorted',
        'str', 'sum', 'tuple', 'type', 'zip',
    }

    # Forbidden imports
    FORBIDDEN_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'socket', 'requests',
        'urllib', 'http', 'ftplib', 'smtplib', 'telnetlib',
        'pickle', 'marshal', 'shelve', 'dbm', 'sqlite3',
        'multiprocessing', 'threading', 'ctypes', 'gc',
    }

    def __init__(self, timeout_seconds: int = 5, max_output_chars: int = 10000):
        """
        Initialize Python tool.

        Args:
            timeout_seconds: Maximum execution time
            max_output_chars: Maximum output length
        """
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars

    def _validate_code(self, code: str) -> Optional[str]:
        """
        Validate code for security issues.

        Returns error message if invalid, None if valid.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Syntax error: {e}"

        for node in ast.walk(tree):
            # Check for forbidden imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if module_name in self.FORBIDDEN_MODULES:
                        return f"Forbidden import: {module_name}"

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if module_name in self.FORBIDDEN_MODULES:
                        return f"Forbidden import: {module_name}"

            # Check for exec/eval
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ('exec', 'eval', 'compile', '__import__'):
                        return f"Forbidden function: {node.func.id}"

        return None

    def _create_safe_globals(self) -> dict:
        """Create a restricted globals dict for execution."""
        safe_builtins = {
            name: getattr(__builtins__ if isinstance(__builtins__, dict)
                         else __builtins__.__dict__, name, None)
            for name in self.SAFE_BUILTINS
        }
        safe_builtins['__builtins__'] = safe_builtins

        # Add safe math functions
        import math
        safe_builtins['math'] = math

        return safe_builtins

    def execute(self, code: str, context: Optional[dict] = None) -> ToolResult:
        """
        Execute Python code safely.

        Args:
            code: Python code to execute
            context: Optional context (unused for Python)

        Returns:
            ToolResult with stdout output or error
        """
        # Validate code first
        error = self._validate_code(code)
        if error:
            return ToolResult(
                success=False,
                output=None,
                error=error
            )

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # Create safe execution environment
            safe_globals = self._create_safe_globals()
            safe_locals = {}

            # Execute with timeout
            with timeout(self.timeout_seconds):
                exec(code, safe_globals, safe_locals)

            # Get output
            output = captured_output.getvalue()
            if len(output) > self.max_output_chars:
                output = output[:self.max_output_chars] + "\n... (truncated)"

            # Also capture return value if last statement is expression
            result = safe_locals.get('result', safe_locals.get('answer', None))
            if result is not None:
                output = f"{output}\nResult: {result}" if output else f"Result: {result}"

            return ToolResult(
                success=True,
                output=output.strip() if output else "Code executed successfully (no output)",
                tokens_used=len(code.split())  # Rough estimate
            )

        except TimeoutError as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )
        except Exception as e:
            tb = traceback.format_exc()
            return ToolResult(
                success=False,
                output=None,
                error=f"{type(e).__name__}: {e}\n{tb}"
            )
        finally:
            sys.stdout = old_stdout

    def solve_math(self, problem: str) -> ToolResult:
        """
        Solve a math problem by generating and executing code.

        Args:
            problem: Math problem description

        Returns:
            ToolResult with the solution
        """
        # This would typically call an LLM to generate code
        # For now, return a placeholder
        return ToolResult(
            success=False,
            output=None,
            error="solve_math requires LLM integration"
        )
