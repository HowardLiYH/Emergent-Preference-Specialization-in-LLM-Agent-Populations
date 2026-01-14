"""
Ablation control for validation.

Compares performance with and without specific components.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Result of an ablation experiment."""
    ablation_name: str
    full_accuracy: float
    ablated_accuracy: float
    difference: float
    component_necessary: bool
    n_samples: int


class AblationControl:
    """
    Runs ablation experiments to validate component necessity.

    Tests include:
    - Memory ablation: Performance with/without memory
    - Tool ablation: Performance with/without optimal tool
    - Competition ablation: Performance with/without fitness sharing
    """

    def __init__(
        self,
        necessity_threshold: float = 0.10
    ):
        """
        Initialize ablation control.

        Args:
            necessity_threshold: Min difference to consider component necessary
        """
        self.necessity_threshold = necessity_threshold
        self.results: Dict[str, AblationResult] = {}

    def run_memory_ablation(
        self,
        agent,
        tasks: List[Any],
        evaluator: callable
    ) -> AblationResult:
        """
        Test if memory is necessary.

        Args:
            agent: Agent to test
            tasks: Tasks to evaluate on
            evaluator: Evaluation function

        Returns:
            AblationResult
        """
        # Test WITH memory
        full_correct = 0
        for task in tasks:
            result = agent.execute_with_tool(task.question, task.regime)
            if result.success:
                is_correct, _ = evaluator(result.output, task)
                if is_correct:
                    full_correct += 1

        full_accuracy = full_correct / len(tasks) if tasks else 0

        # Test WITHOUT memory (temporarily disable)
        original_memory = agent.memory
        agent.memory = None

        ablated_correct = 0
        for task in tasks:
            result = agent.execute_with_tool(task.question, task.regime)
            if result.success:
                is_correct, _ = evaluator(result.output, task)
                if is_correct:
                    ablated_correct += 1

        ablated_accuracy = ablated_correct / len(tasks) if tasks else 0

        # Restore memory
        agent.memory = original_memory

        difference = full_accuracy - ablated_accuracy

        result = AblationResult(
            ablation_name='memory',
            full_accuracy=full_accuracy,
            ablated_accuracy=ablated_accuracy,
            difference=difference,
            component_necessary=difference >= self.necessity_threshold,
            n_samples=len(tasks)
        )

        self.results['memory'] = result
        return result

    def run_tool_ablation(
        self,
        agent,
        tasks: List[Any],
        evaluator: callable
    ) -> AblationResult:
        """
        Test if optimal tool selection is necessary.
        """
        # Test with optimal tool selection
        full_correct = 0
        for task in tasks:
            result = agent.execute_with_tool(task.question, task.regime)
            if result.success:
                is_correct, _ = evaluator(result.output, task)
                if is_correct:
                    full_correct += 1

        full_accuracy = full_correct / len(tasks) if tasks else 0

        # Test with forced L0 (no tool)
        ablated_correct = 0
        for task in tasks:
            result = agent.execute_with_tool(task.question, task.regime, tool_name='L0')
            if result.success:
                is_correct, _ = evaluator(result.output, task)
                if is_correct:
                    ablated_correct += 1

        ablated_accuracy = ablated_correct / len(tasks) if tasks else 0

        difference = full_accuracy - ablated_accuracy

        result = AblationResult(
            ablation_name='tool_selection',
            full_accuracy=full_accuracy,
            ablated_accuracy=ablated_accuracy,
            difference=difference,
            component_necessary=difference >= self.necessity_threshold,
            n_samples=len(tasks)
        )

        self.results['tool_selection'] = result
        return result

    def generate_report(self) -> str:
        """Generate ablation report."""
        lines = [
            "# Ablation Study Report",
            "",
            "| Component | Full | Ablated | Difference | Necessary? |",
            "|-----------|------|---------|------------|------------|",
        ]

        for name, result in self.results.items():
            status = "✅ Yes" if result.component_necessary else "❌ No"
            lines.append(
                f"| {name} | {result.full_accuracy:.1%} | "
                f"{result.ablated_accuracy:.1%} | {result.difference:+.1%} | {status} |"
            )

        return "\n".join(lines)
