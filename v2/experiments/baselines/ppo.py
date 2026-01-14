"""
Simplified PPO baseline for prompt-based policy learning.

Uses LLM feedback for policy updates without gradient access.
"""

from typing import List, Dict, Optional, Any
import random
import logging

logger = logging.getLogger(__name__)


class SimplePPO:
    """
    Simplified PPO for prompt-based policy.

    Since we don't have gradient access to LLM weights,
    this uses a prompt-preference approach where we:
    1. Generate multiple responses
    2. Evaluate them
    3. Update preference over response styles
    """

    def __init__(
        self,
        n_actions: int = 5,  # Number of tool choices
        clip_epsilon: float = 0.2,
        lr: float = 0.1
    ):
        """
        Initialize simplified PPO.

        Args:
            n_actions: Number of possible actions (tools)
            clip_epsilon: PPO clipping parameter
            lr: Learning rate for preference updates
        """
        self.n_actions = n_actions
        self.clip_epsilon = clip_epsilon
        self.lr = lr

        # Policy as preferences (logits)
        self.preferences: List[float] = [0.0] * n_actions

        # Statistics
        self.total_episodes = 0
        self.total_tokens = 0
        self.rewards_history: List[float] = []

    def _softmax(self, logits: List[float]) -> List[float]:
        """Compute softmax probabilities."""
        import math
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exp_logits)
        return [e / sum_exp for e in exp_logits]

    def get_policy(self) -> List[float]:
        """Get current policy as probabilities."""
        return self._softmax(self.preferences)

    def select_action(self) -> int:
        """Sample action from current policy."""
        probs = self.get_policy()
        return random.choices(range(self.n_actions), weights=probs, k=1)[0]

    def update(
        self,
        action: int,
        reward: float,
        tokens: int = 0
    ) -> None:
        """
        Update policy based on outcome.

        Simplified PPO update without gradient access.
        """
        # Get old probability
        old_probs = self.get_policy()
        old_prob = old_probs[action]

        # Advantage estimate (simplified: just use reward - baseline)
        baseline = sum(self.rewards_history[-50:]) / max(1, len(self.rewards_history[-50:]))
        advantage = reward - baseline

        # Update preference for taken action
        if advantage > 0:
            # Increase preference for successful actions
            self.preferences[action] += self.lr * advantage
        else:
            # Decrease preference for unsuccessful actions
            self.preferences[action] += self.lr * advantage

        # Record
        self.total_episodes += 1
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
        """Get PPO statistics."""
        return {
            'total_episodes': self.total_episodes,
            'total_tokens': self.total_tokens,
            'accuracy': self.get_accuracy(),
            'policy': self.get_policy(),
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
        """
        iterations = 0

        while iterations < max_iterations:
            # Select action (tool)
            action = self.select_action()

            # Generate task
            task = task_generator(action)

            # Get response
            response = llm_client.generate(task.question)

            # Evaluate
            correct, _ = evaluator(response.text, task)
            reward = 1.0 if correct else 0.0

            # Update policy
            self.update(action, reward, response.tokens_used)
            iterations += 1

            # Check target
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
