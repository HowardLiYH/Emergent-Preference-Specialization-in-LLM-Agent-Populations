"""
Task curriculum for gradual difficulty increase.

Starts with easier tasks and progressively introduces harder ones.
"""

from typing import Dict, Optional
import random
import logging

logger = logging.getLogger(__name__)


class TaskCurriculum:
    """
    Curriculum learning for task difficulty.

    Gradually introduces higher-level tasks as training progresses.
    This helps agents learn basic tool use before tackling
    complex multi-tool scenarios.
    """

    # Default curriculum stages
    DEFAULT_STAGES = {
        # (start_gen, end_gen): {tool_level: probability}
        (1, 20): {'L0': 0.50, 'L1': 0.35, 'L2': 0.10, 'L3': 0.05, 'L4': 0.00},
        (21, 50): {'L0': 0.25, 'L1': 0.25, 'L2': 0.25, 'L3': 0.20, 'L4': 0.05},
        (51, float('inf')): {'L0': 0.20, 'L1': 0.20, 'L2': 0.20, 'L3': 0.20, 'L4': 0.20},
    }

    def __init__(
        self,
        stages: Optional[Dict] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize curriculum.

        Args:
            stages: Custom curriculum stages
            seed: Random seed for reproducibility
        """
        self.stages = stages or self.DEFAULT_STAGES
        self.rng = random.Random(seed)
        self.current_generation = 0

    def get_task_distribution(self, generation: int) -> Dict[str, float]:
        """
        Get task distribution for a given generation.

        Args:
            generation: Current generation number

        Returns:
            Dict mapping tool levels to probabilities
        """
        for (start, end), distribution in self.stages.items():
            if start <= generation <= end:
                return distribution

        # Default to last stage
        last_stage = list(self.stages.values())[-1]
        return last_stage

    def sample_task_level(self, generation: Optional[int] = None) -> str:
        """
        Sample a task level according to curriculum.

        Args:
            generation: Override current generation

        Returns:
            Sampled tool level (e.g., 'L0', 'L1', ...)
        """
        gen = generation if generation is not None else self.current_generation
        distribution = self.get_task_distribution(gen)

        levels = list(distribution.keys())
        probs = list(distribution.values())

        return self.rng.choices(levels, weights=probs, k=1)[0]

    def advance(self) -> int:
        """
        Advance to next generation.

        Returns:
            New generation number
        """
        self.current_generation += 1
        return self.current_generation

    def get_current_stage(self) -> str:
        """Get description of current curriculum stage."""
        gen = self.current_generation

        if gen <= 20:
            return "Early training: Focus on L0-L1 tasks"
        elif gen <= 50:
            return "Mid training: Balanced L0-L3 tasks"
        else:
            return "Late training: Uniform distribution"

    def get_progress(self) -> Dict[str, any]:
        """Get curriculum progress information."""
        return {
            'current_generation': self.current_generation,
            'stage': self.get_current_stage(),
            'distribution': self.get_task_distribution(self.current_generation),
        }

    def reset(self) -> None:
        """Reset curriculum to beginning."""
        self.current_generation = 0


class AdaptiveCurriculum(TaskCurriculum):
    """
    Adaptive curriculum that adjusts based on agent performance.

    If agents are doing well, introduce harder tasks sooner.
    If agents are struggling, keep them on easier tasks longer.
    """

    def __init__(
        self,
        target_win_rate: float = 0.5,
        adaptation_rate: float = 0.1,
        **kwargs
    ):
        """
        Initialize adaptive curriculum.

        Args:
            target_win_rate: Target win rate for adaptation
            adaptation_rate: How quickly to adapt (0-1)
            **kwargs: Arguments for base TaskCurriculum
        """
        super().__init__(**kwargs)
        self.target_win_rate = target_win_rate
        self.adaptation_rate = adaptation_rate

        # Performance tracking
        self.level_performance: Dict[str, Dict[str, float]] = {
            level: {'wins': 0, 'total': 0}
            for level in ['L0', 'L1', 'L2', 'L3', 'L4']
        }

        # Dynamic adjustment factors
        self.level_weights: Dict[str, float] = {
            level: 1.0 for level in ['L0', 'L1', 'L2', 'L3', 'L4']
        }

    def record_result(self, level: str, won: bool) -> None:
        """
        Record a task result for adaptation.

        Args:
            level: Task level
            won: Whether the task was successful
        """
        if level not in self.level_performance:
            return

        self.level_performance[level]['total'] += 1
        if won:
            self.level_performance[level]['wins'] += 1

        # Adapt weights
        self._adapt_weights()

    def _adapt_weights(self) -> None:
        """Adjust level weights based on performance."""
        for level, perf in self.level_performance.items():
            if perf['total'] < 10:  # Need minimum samples
                continue

            win_rate = perf['wins'] / perf['total']

            if win_rate > self.target_win_rate + 0.1:
                # Too easy, reduce weight
                self.level_weights[level] *= (1 - self.adaptation_rate)
            elif win_rate < self.target_win_rate - 0.1:
                # Too hard, increase weight (give more practice)
                self.level_weights[level] *= (1 + self.adaptation_rate)

            # Clamp weights
            self.level_weights[level] = max(0.1, min(3.0, self.level_weights[level]))

    def get_task_distribution(self, generation: int) -> Dict[str, float]:
        """Get adapted task distribution."""
        base_dist = super().get_task_distribution(generation)

        # Apply weights
        adapted = {
            level: prob * self.level_weights[level]
            for level, prob in base_dist.items()
        }

        # Normalize
        total = sum(adapted.values())
        if total > 0:
            adapted = {level: prob / total for level, prob in adapted.items()}

        return adapted
