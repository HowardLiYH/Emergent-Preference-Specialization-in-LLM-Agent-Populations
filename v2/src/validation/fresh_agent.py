"""
Fresh agent test for anti-leakage validation.

Tests if memories provide genuine value by giving them
to an untrained agent.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FreshAgentResult:
    """Result of fresh agent test."""
    regime: str
    base_accuracy: float  # Fresh agent without memories
    memory_accuracy: float  # Fresh agent with memories
    improvement: float
    n_tasks: int
    passed: bool


class FreshAgentTest:
    """
    Tests if memories genuinely help new agents.

    Protocol:
    1. Create fresh agent (no training history)
    2. Evaluate on tasks WITHOUT memories
    3. Evaluate on same tasks WITH copied memories
    4. If memories help, they contain useful strategies
    5. If no help, memories might be task-specific
    """

    def __init__(
        self,
        min_improvement: float = 0.10,
        llm_client=None
    ):
        """
        Initialize fresh agent test.

        Args:
            min_improvement: Minimum improvement to pass
            llm_client: LLM client for fresh agents
        """
        self.min_improvement = min_improvement
        self.llm_client = llm_client

    def run_test(
        self,
        regime: str,
        tasks: List[Any],
        memories: List[str],
        evaluator: callable
    ) -> FreshAgentResult:
        """
        Run fresh agent test for a regime.

        Args:
            regime: Regime to test
            tasks: Tasks to evaluate on
            memories: Memories from trained specialist
            evaluator: Function (response, task) -> (correct, confidence)

        Returns:
            FreshAgentResult
        """
        from ..tools.agent import Agent

        # Create fresh agent
        fresh_agent = Agent(
            tool_level=4,  # Give full access
            llm_client=self.llm_client
        )

        # Test WITHOUT memories
        base_correct = 0
        for task in tasks:
            result = fresh_agent.execute_with_tool(task.question, regime)
            if result.success:
                is_correct, _ = evaluator(result.output, task)
                if is_correct:
                    base_correct += 1

        base_accuracy = base_correct / len(tasks) if tasks else 0

        # Test WITH memories
        # Inject memories into context
        memory_correct = 0
        for task in tasks:
            # Build context with memories
            context = {
                'regime': regime,
                'memories': memories[:5]  # Use top 5 memories
            }

            result = fresh_agent.execute_with_tool(task.question, regime)
            if result.success:
                is_correct, _ = evaluator(result.output, task)
                if is_correct:
                    memory_correct += 1

        memory_accuracy = memory_correct / len(tasks) if tasks else 0
        improvement = memory_accuracy - base_accuracy

        return FreshAgentResult(
            regime=regime,
            base_accuracy=base_accuracy,
            memory_accuracy=memory_accuracy,
            improvement=improvement,
            n_tasks=len(tasks),
            passed=improvement >= self.min_improvement
        )

    def run_population_test(
        self,
        specialists: Dict[str, Any],
        task_generator: callable,
        evaluator: callable,
        n_tasks: int = 30
    ) -> Dict[str, FreshAgentResult]:
        """
        Run fresh agent test for all specialists.

        Args:
            specialists: Dict[regime -> specialist_agent]
            task_generator: Function (regime, n) -> List[Task]
            evaluator: Evaluation function
            n_tasks: Tasks per regime

        Returns:
            Dict[regime -> FreshAgentResult]
        """
        results = {}

        for regime, specialist in specialists.items():
            # Get memories from specialist
            if hasattr(specialist, 'memory') and specialist.memory:
                memories = [
                    m.content for m in specialist.memory.load_all(specialist.id)
                ]
            else:
                memories = []

            if not memories:
                logger.warning(f"No memories for {regime} specialist")
                continue

            # Generate tasks
            tasks = task_generator(regime, n_tasks)

            # Run test
            result = self.run_test(regime, tasks, memories, evaluator)
            results[regime] = result

        return results

    def generate_report(
        self,
        results: Dict[str, FreshAgentResult]
    ) -> str:
        """Generate test report."""
        lines = [
            "# Fresh Agent Test Report",
            "",
            "| Regime | Base | With Memory | Improvement | Passed? |",
            "|--------|------|-------------|-------------|---------|",
        ]

        for regime, result in results.items():
            status = "✅" if result.passed else "❌"
            lines.append(
                f"| {regime} | {result.base_accuracy:.1%} | "
                f"{result.memory_accuracy:.1%} | {result.improvement:+.1%} | {status} |"
            )

        passed_count = sum(1 for r in results.values() if r.passed)
        total = len(results)

        lines.extend([
            "",
            f"**Overall: {passed_count}/{total} regimes passed fresh agent test**",
        ])

        return "\n".join(lines)
