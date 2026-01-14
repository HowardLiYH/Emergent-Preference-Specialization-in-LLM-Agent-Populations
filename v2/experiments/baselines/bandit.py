"""
Simple UCB1 bandit baseline.

No context awareness - treats each arm independently.
"""

import math
from typing import List, Dict, Optional, Any
import random
import logging

logger = logging.getLogger(__name__)


class UCB1Bandit:
    """
    UCB1 (Upper Confidence Bound) bandit.

    Baseline for comparing against population-based approach.
    Does NOT condition on task features.
    """

    def __init__(
        self,
        n_arms: int,
        exploration_weight: float = 2.0
    ):
        """
        Initialize UCB1 bandit.

        Args:
            n_arms: Number of arms (tools)
            exploration_weight: UCB exploration parameter
        """
        self.n_arms = n_arms
        self.exploration_weight = exploration_weight

        # Statistics per arm
        self.counts: List[int] = [0] * n_arms
        self.rewards: List[float] = [0.0] * n_arms
        self.total_pulls = 0

        # Token tracking
        self.total_tokens = 0

    def select_arm(self) -> int:
        """Select an arm using UCB1."""
        self.total_pulls += 1

        # Pull unexplored arms first
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        # Calculate UCB for each arm
        ucb_values = []
        for arm in range(self.n_arms):
            mean_reward = self.rewards[arm] / self.counts[arm]
            exploration_bonus = math.sqrt(
                self.exploration_weight * math.log(self.total_pulls) / self.counts[arm]
            )
            ucb_values.append(mean_reward + exploration_bonus)

        return max(range(self.n_arms), key=lambda a: ucb_values[a])

    def update(self, arm: int, reward: float, tokens: int = 0) -> None:
        """
        Update arm statistics.

        Args:
            arm: Arm that was pulled
            reward: Reward received (0 or 1)
            tokens: Tokens used
        """
        self.counts[arm] += 1
        self.rewards[arm] += reward
        self.total_tokens += tokens

    def get_accuracy(self) -> float:
        """Get overall accuracy."""
        total_rewards = sum(self.rewards)
        total_pulls = sum(self.counts)
        return total_rewards / total_pulls if total_pulls > 0 else 0

    def get_stats(self) -> Dict[str, Any]:
        """Get bandit statistics."""
        return {
            'total_pulls': sum(self.counts),
            'total_tokens': self.total_tokens,
            'accuracy': self.get_accuracy(),
            'arm_counts': self.counts.copy(),
            'arm_rewards': self.rewards.copy(),
        }

    def train_to_target(
        self,
        task_generator: callable,
        evaluator: callable,
        llm_client,
        target_accuracy: float = 0.80,
        max_iterations: int = 1000
    ) -> Dict[str, Any]:
        """
        Train until target accuracy is reached.

        Args:
            task_generator: Function (arm) -> task
            evaluator: Function (response, task) -> (correct, confidence)
            llm_client: LLM client
            target_accuracy: Target accuracy to reach
            max_iterations: Maximum iterations

        Returns:
            Training results
        """
        iterations = 0

        while iterations < max_iterations:
            # Select arm
            arm = self.select_arm()

            # Generate task and get response
            task = task_generator(arm)
            response = llm_client.generate(task.question)

            # Evaluate
            correct, _ = evaluator(response.text, task)
            reward = 1.0 if correct else 0.0

            # Update
            self.update(arm, reward, response.tokens_used)
            iterations += 1

            # Check if target reached
            if iterations >= 50 and self.get_accuracy() >= target_accuracy:
                break

        return {
            'iterations': iterations,
            'tokens': self.total_tokens,
            'accuracy': self.get_accuracy(),
            'reached_target': self.get_accuracy() >= target_accuracy,
        }
