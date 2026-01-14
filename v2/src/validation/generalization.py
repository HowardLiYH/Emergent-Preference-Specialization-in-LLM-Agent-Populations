"""
Generalization test for anti-leakage validation.

Tests whether learned strategies generalize to unseen tasks
vs simply memorizing specific answers.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import random
import logging

logger = logging.getLogger(__name__)


@dataclass
class GeneralizationResult:
    """Result of generalization test."""
    agent_id: str
    training_accuracy: float
    holdout_accuracy: float
    generalization_gap: float
    passed: bool
    n_training: int
    n_holdout: int


class GeneralizationTest:
    """
    Tests if agents generalize to unseen tasks.

    Protocol:
    1. Train on subset of tasks
    2. Evaluate on held-out tasks (same regime, different instances)
    3. If holdout accuracy close to training, generalization is good
    4. If holdout much lower, might be memorizing answers
    """

    def __init__(
        self,
        holdout_ratio: float = 0.2,
        max_gap: float = 0.15,
        seed: Optional[int] = None
    ):
        """
        Initialize generalization test.

        Args:
            holdout_ratio: Fraction of tasks to hold out
            max_gap: Maximum acceptable gap between train/holdout
            seed: Random seed
        """
        self.holdout_ratio = holdout_ratio
        self.max_gap = max_gap
        self.rng = random.Random(seed)

    def split_tasks(
        self,
        tasks: List[Any]
    ) -> Tuple[List[Any], List[Any]]:
        """
        Split tasks into training and holdout sets.

        Args:
            tasks: Full list of tasks

        Returns:
            Tuple of (training_tasks, holdout_tasks)
        """
        n_holdout = max(1, int(len(tasks) * self.holdout_ratio))

        shuffled = tasks.copy()
        self.rng.shuffle(shuffled)

        holdout = shuffled[:n_holdout]
        training = shuffled[n_holdout:]

        return training, holdout

    def run_test(
        self,
        agent,
        tasks: List[Any],
        evaluator: callable
    ) -> GeneralizationResult:
        """
        Run generalization test for an agent.

        Args:
            agent: Agent to test
            tasks: All tasks for the agent's specialty regime
            evaluator: Function (response, task) -> (correct, confidence)

        Returns:
            GeneralizationResult
        """
        training_tasks, holdout_tasks = self.split_tasks(tasks)

        # Evaluate on training tasks
        training_correct = 0
        for task in training_tasks:
            result = agent.execute_with_tool(task.question, task.regime)
            if result.success:
                is_correct, _ = evaluator(result.output, task)
                if is_correct:
                    training_correct += 1

        training_accuracy = training_correct / len(training_tasks) if training_tasks else 0

        # Evaluate on holdout tasks
        holdout_correct = 0
        for task in holdout_tasks:
            result = agent.execute_with_tool(task.question, task.regime)
            if result.success:
                is_correct, _ = evaluator(result.output, task)
                if is_correct:
                    holdout_correct += 1

        holdout_accuracy = holdout_correct / len(holdout_tasks) if holdout_tasks else 0

        # Calculate generalization gap
        gap = training_accuracy - holdout_accuracy
        passed = gap <= self.max_gap

        return GeneralizationResult(
            agent_id=agent.id,
            training_accuracy=training_accuracy,
            holdout_accuracy=holdout_accuracy,
            generalization_gap=gap,
            passed=passed,
            n_training=len(training_tasks),
            n_holdout=len(holdout_tasks)
        )

    def run_population_test(
        self,
        agents: List,
        task_generator: callable,
        evaluator: callable,
        n_tasks_per_regime: int = 50
    ) -> Dict[str, GeneralizationResult]:
        """
        Run generalization test for entire population.

        Args:
            agents: List of agents
            task_generator: Function (regime, n) -> List[Task]
            evaluator: Function (response, task) -> (correct, confidence)
            n_tasks_per_regime: Tasks to generate per regime

        Returns:
            Dict mapping agent_id to GeneralizationResult
        """
        results = {}

        for agent in agents:
            specialty = agent.get_specialty()
            if not specialty:
                continue

            # Generate tasks for this regime
            tasks = task_generator(specialty, n_tasks_per_regime)

            # Run test
            result = self.run_test(agent, tasks, evaluator)
            results[agent.id] = result

        return results

    def generate_report(
        self,
        results: Dict[str, GeneralizationResult]
    ) -> str:
        """Generate test report."""
        lines = [
            "# Generalization Test Report",
            "",
            "| Agent | Training | Holdout | Gap | Passed? |",
            "|-------|----------|---------|-----|---------|",
        ]

        for agent_id, result in results.items():
            status = "✅" if result.passed else "❌"
            lines.append(
                f"| {agent_id} | {result.training_accuracy:.1%} | "
                f"{result.holdout_accuracy:.1%} | {result.generalization_gap:.1%} | {status} |"
            )

        passed_count = sum(1 for r in results.values() if r.passed)
        total = len(results)

        lines.extend([
            "",
            f"**Overall: {passed_count}/{total} agents passed generalization test**",
        ])

        return "\n".join(lines)
