"""
Tool selection via Thompson Sampling.

Agents use Thompson Sampling to discover which tools work best
for different regimes without prior knowledge of the optimal mapping.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import random
import logging

logger = logging.getLogger(__name__)


@dataclass
class BetaDistribution:
    """Beta distribution for Thompson Sampling."""
    alpha: float = 1.0  # Successes + 1
    beta: float = 1.0   # Failures + 1

    def sample(self) -> float:
        """Sample from the Beta distribution."""
        return random.betavariate(self.alpha, self.beta)

    def mean(self) -> float:
        """Expected value of the distribution."""
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        """Variance of the distribution."""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total ** 2 * (total + 1))

    def update(self, success: bool) -> None:
        """Update the distribution based on outcome."""
        if success:
            self.alpha += 1
        else:
            self.beta += 1

    def __repr__(self) -> str:
        return f"Beta(α={self.alpha:.1f}, β={self.beta:.1f}, μ={self.mean():.3f})"


class ToolSelectionPolicy:
    """
    Thompson Sampling policy for tool selection.

    Each agent maintains beliefs about the success probability
    of each tool. The policy samples from these beliefs and
    selects the tool with the highest sampled value.

    Key properties:
    - Natural exploration without ε-greedy
    - Converges to optimal tool as evidence accumulates
    - Can be regime-specific if provided with regime context
    """

    def __init__(
        self,
        available_tools: List[str],
        regime_specific: bool = True
    ):
        """
        Initialize tool selection policy.

        Args:
            available_tools: List of available tool names (e.g., ['L0', 'L1', ...])
            regime_specific: Whether to maintain separate beliefs per regime
        """
        self.available_tools = available_tools
        self.regime_specific = regime_specific

        # Global beliefs (used if regime_specific=False)
        self.beliefs: Dict[str, BetaDistribution] = {
            tool: BetaDistribution() for tool in available_tools
        }

        # Per-regime beliefs
        self.regime_beliefs: Dict[str, Dict[str, BetaDistribution]] = {}

    def _get_beliefs(self, regime: Optional[str] = None) -> Dict[str, BetaDistribution]:
        """Get beliefs for a regime (or global if not regime-specific)."""
        if not self.regime_specific or regime is None:
            return self.beliefs

        if regime not in self.regime_beliefs:
            # Initialize beliefs for new regime
            self.regime_beliefs[regime] = {
                tool: BetaDistribution() for tool in self.available_tools
            }

        return self.regime_beliefs[regime]

    def select(
        self,
        task_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Select a tool using Thompson Sampling.

        Args:
            task_context: Optional context including 'regime' key

        Returns:
            Selected tool name
        """
        regime = task_context.get('regime') if task_context else None
        beliefs = self._get_beliefs(regime)

        # Sample from each tool's belief distribution
        samples = {
            tool: belief.sample()
            for tool, belief in beliefs.items()
        }

        # Select tool with highest sample
        selected = max(samples, key=samples.get)

        logger.debug(
            f"Tool selection (regime={regime}): "
            f"samples={[(t, f'{s:.3f}') for t, s in samples.items()]}, "
            f"selected={selected}"
        )

        return selected

    def update(
        self,
        tool: str,
        success: bool,
        regime: Optional[str] = None
    ) -> None:
        """
        Update beliefs based on tool outcome.

        Args:
            tool: Tool that was used
            success: Whether the tool use was successful
            regime: Optional regime context
        """
        beliefs = self._get_beliefs(regime)

        if tool in beliefs:
            beliefs[tool].update(success)
            logger.debug(
                f"Updated {tool} for regime {regime}: "
                f"success={success}, new belief={beliefs[tool]}"
            )

    def get_tool_confidence(
        self,
        tool: str,
        regime: Optional[str] = None
    ) -> float:
        """
        Get current confidence in a tool.

        Args:
            tool: Tool name
            regime: Optional regime context

        Returns:
            Expected success rate (mean of beta distribution)
        """
        beliefs = self._get_beliefs(regime)
        if tool in beliefs:
            return beliefs[tool].mean()
        return 0.5  # Uninformative prior

    def get_stats(self, regime: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all tools.

        Args:
            regime: Optional regime context

        Returns:
            Dict mapping tool names to stats (mean, variance, trials)
        """
        beliefs = self._get_beliefs(regime)

        stats = {}
        for tool, belief in beliefs.items():
            trials = belief.alpha + belief.beta - 2  # Subtract initial priors
            stats[tool] = {
                'mean': belief.mean(),
                'variance': belief.variance(),
                'successes': belief.alpha - 1,
                'failures': belief.beta - 1,
                'trials': trials,
            }

        return stats

    def reset(self) -> None:
        """Reset all beliefs to uninformative priors."""
        self.beliefs = {
            tool: BetaDistribution() for tool in self.available_tools
        }
        self.regime_beliefs = {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize policy to dictionary."""
        return {
            'available_tools': self.available_tools,
            'regime_specific': self.regime_specific,
            'beliefs': {
                tool: {'alpha': b.alpha, 'beta': b.beta}
                for tool, b in self.beliefs.items()
            },
            'regime_beliefs': {
                regime: {
                    tool: {'alpha': b.alpha, 'beta': b.beta}
                    for tool, b in beliefs.items()
                }
                for regime, beliefs in self.regime_beliefs.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolSelectionPolicy':
        """Deserialize policy from dictionary."""
        policy = cls(
            available_tools=data['available_tools'],
            regime_specific=data.get('regime_specific', True)
        )

        # Restore beliefs
        for tool, params in data.get('beliefs', {}).items():
            if tool in policy.beliefs:
                policy.beliefs[tool] = BetaDistribution(**params)

        # Restore regime-specific beliefs
        for regime, beliefs in data.get('regime_beliefs', {}).items():
            policy.regime_beliefs[regime] = {
                tool: BetaDistribution(**params)
                for tool, params in beliefs.items()
            }

        return policy


class UCBToolSelection:
    """
    UCB1 (Upper Confidence Bound) alternative for tool selection.

    Provided as a baseline comparison to Thompson Sampling.
    """

    def __init__(
        self,
        available_tools: List[str],
        exploration_weight: float = 2.0
    ):
        """
        Initialize UCB1 policy.

        Args:
            available_tools: List of available tool names
            exploration_weight: Weight for exploration term (higher = more exploration)
        """
        self.available_tools = available_tools
        self.exploration_weight = exploration_weight

        self.counts: Dict[str, int] = {tool: 0 for tool in available_tools}
        self.successes: Dict[str, float] = {tool: 0.0 for tool in available_tools}
        self.total_pulls = 0

    def select(self, task_context: Optional[Dict[str, Any]] = None) -> str:
        """Select a tool using UCB1."""
        import math

        self.total_pulls += 1

        # Handle unexplored tools
        unexplored = [t for t in self.available_tools if self.counts[t] == 0]
        if unexplored:
            return random.choice(unexplored)

        # Compute UCB score for each tool
        scores = {}
        for tool in self.available_tools:
            mean_reward = self.successes[tool] / self.counts[tool]
            exploration_bonus = math.sqrt(
                self.exploration_weight * math.log(self.total_pulls) / self.counts[tool]
            )
            scores[tool] = mean_reward + exploration_bonus

        return max(scores, key=scores.get)

    def update(self, tool: str, success: bool, regime: Optional[str] = None) -> None:
        """Update statistics for a tool."""
        self.counts[tool] += 1
        if success:
            self.successes[tool] += 1
