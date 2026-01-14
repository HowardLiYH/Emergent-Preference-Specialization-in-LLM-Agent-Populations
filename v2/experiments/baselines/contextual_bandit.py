"""
Contextual bandit baseline using LinUCB.

Conditions on task features for smarter arm selection.
"""

import numpy as np
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class LinUCBBandit:
    """
    LinUCB contextual bandit.

    Uses linear regression to predict arm rewards based on context.
    Stronger baseline than simple UCB1.
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        alpha: float = 1.0
    ):
        """
        Initialize LinUCB bandit.

        Args:
            n_arms: Number of arms
            context_dim: Dimension of context features
            alpha: Exploration parameter
        """
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha

        # Initialize parameters for each arm
        self.A: Dict[int, np.ndarray] = {
            a: np.eye(context_dim) for a in range(n_arms)
        }
        self.b: Dict[int, np.ndarray] = {
            a: np.zeros(context_dim) for a in range(n_arms)
        }

        # Statistics
        self.total_pulls = 0
        self.total_tokens = 0
        self.rewards_history: List[float] = []

    def _extract_context(self, task) -> np.ndarray:
        """
        Extract context features from task.

        Args:
            task: Task object

        Returns:
            Context vector
        """
        # Simple feature extraction
        # In practice, would use better features
        context = np.zeros(self.context_dim)

        if hasattr(task, 'question'):
            # Length-based features
            context[0] = len(task.question) / 100  # Normalized length
            context[1] = task.question.count('?') / 3  # Question marks
            context[2] = len(task.question.split()) / 50  # Word count

            # Keyword features
            keywords = {
                'calculate': 3, 'compute': 3, 'math': 3,
                'image': 4, 'chart': 4, 'graph': 4,
                'document': 5, 'find': 5, 'search': 5,
                'current': 6, 'today': 6, 'now': 6,
            }
            for word, idx in keywords.items():
                if idx < self.context_dim:
                    context[idx] = 1.0 if word in task.question.lower() else 0.0

        return context

    def select_arm(self, context: np.ndarray) -> int:
        """
        Select arm using LinUCB.

        Args:
            context: Context feature vector

        Returns:
            Selected arm
        """
        self.total_pulls += 1

        ucb_values = []
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]

            # UCB value
            mean = context @ theta
            exploration = self.alpha * np.sqrt(context @ A_inv @ context)

            ucb_values.append(mean + exploration)

        return int(np.argmax(ucb_values))

    def update(
        self,
        arm: int,
        context: np.ndarray,
        reward: float,
        tokens: int = 0
    ) -> None:
        """
        Update arm parameters.

        Args:
            arm: Arm that was pulled
            context: Context vector
            reward: Reward received
            tokens: Tokens used
        """
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        self.total_tokens += tokens
        self.rewards_history.append(reward)

    def get_accuracy(self, window: Optional[int] = None) -> float:
        """Get accuracy over recent window."""
        if not self.rewards_history:
            return 0.0

        rewards = self.rewards_history
        if window:
            rewards = rewards[-window:]

        return sum(rewards) / len(rewards)

    def get_stats(self) -> Dict[str, Any]:
        """Get bandit statistics."""
        return {
            'total_pulls': self.total_pulls,
            'total_tokens': self.total_tokens,
            'accuracy': self.get_accuracy(),
            'recent_accuracy': self.get_accuracy(window=50),
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
        Train until target accuracy.

        Returns training results including tokens used.
        """
        iterations = 0

        while iterations < max_iterations:
            # Generate task
            task = task_generator(None)  # Random regime

            # Extract context
            context = self._extract_context(task)

            # Select arm
            arm = self.select_arm(context)

            # Get response
            response = llm_client.generate(task.question)

            # Evaluate
            correct, _ = evaluator(response.text, task)
            reward = 1.0 if correct else 0.0

            # Update
            self.update(arm, context, reward, response.tokens_used)
            iterations += 1

            # Check if target reached
            if iterations >= 50:
                recent_acc = self.get_accuracy(window=50)
                if recent_acc >= target_accuracy:
                    break

        return {
            'iterations': iterations,
            'tokens': self.total_tokens,
            'accuracy': self.get_accuracy(),
            'reached_target': self.get_accuracy() >= target_accuracy,
        }
