"""
Thompson Sampling for Strategy Selection

Implements Beta distribution-based beliefs for strategy selection,
matching Paper 1's NichePopulation algorithm.

Key insight: Rather than random strategy selection, agents use Thompson Sampling
to balance exploration/exploitation when choosing which strategies to add.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from enum import Enum

try:
    from .domains import DomainType
    from .strategy_library import Strategy, STRATEGY_LIBRARY
except ImportError:
    from domains import DomainType
    from strategy_library import Strategy, STRATEGY_LIBRARY


@dataclass
class StrategyBelief:
    """
    Beta distribution belief for a single strategy's success probability.

    Uses conjugate prior: Beta(alpha, beta) where:
    - alpha = successes + 1 (prior)
    - beta = failures + 1 (prior)

    This matches Paper 1's MethodBelief class.
    """
    strategy_id: str
    domain: DomainType
    alpha: float = 1.0  # successes + prior
    beta: float = 1.0   # failures + prior
    times_selected: int = 0
    times_won: int = 0

    def sample(self, rng: np.random.Generator = None) -> float:
        """
        Sample from the Beta distribution (Thompson Sampling).

        Returns a sample from Beta(alpha, beta) representing
        the sampled success probability.
        """
        if rng is None:
            rng = np.random.default_rng()
        return rng.beta(self.alpha, self.beta)

    def update(self, won: bool):
        """
        Update belief based on outcome.

        Args:
            won: Whether the agent won when using this strategy
        """
        self.times_selected += 1
        if won:
            self.alpha += 1
            self.times_won += 1
        else:
            self.beta += 1

    def update_discounted(self, won: bool, num_active_strategies: int):
        """
        Update with discounted credit assignment.

        When multiple strategies are active, credit is split.
        This addresses the credit assignment problem in cumulative prompts.

        Args:
            won: Whether the agent won
            num_active_strategies: Number of strategies currently in portfolio
        """
        self.times_selected += 1
        credit = 1.0 / max(1, num_active_strategies)

        if won:
            self.alpha += credit
            self.times_won += 1
        else:
            self.beta += credit

    def expected_value(self) -> float:
        """Expected success probability: E[Beta(α,β)] = α/(α+β)"""
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        """Variance of the Beta distribution."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def confidence_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval for success probability."""
        from scipy import stats
        dist = stats.beta(self.alpha, self.beta)
        alpha = 1 - level
        return (dist.ppf(alpha / 2), dist.ppf(1 - alpha / 2))

    def __repr__(self) -> str:
        return (
            f"StrategyBelief({self.strategy_id}, "
            f"α={self.alpha:.1f}, β={self.beta:.1f}, "
            f"E={self.expected_value():.3f})"
        )


class StrategyBeliefManager:
    """
    Manages beliefs for all strategies across domains.

    This class handles:
    1. Thompson Sampling for strategy selection
    2. Credit assignment for belief updates
    3. Tracking of strategy performance
    """

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        self._beliefs: Dict[str, StrategyBelief] = {}
        self._domain_beliefs: Dict[DomainType, List[StrategyBelief]] = {
            d: [] for d in DomainType
        }
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of beliefs for all strategies."""
        if self._initialized:
            return

        for strategy in STRATEGY_LIBRARY.get_all():
            belief = StrategyBelief(
                strategy_id=strategy.id,
                domain=strategy.domain
            )
            self._beliefs[strategy.id] = belief
            self._domain_beliefs[strategy.domain].append(belief)

        self._initialized = True

    def get_belief(self, strategy_id: str) -> Optional[StrategyBelief]:
        """Get belief for a specific strategy."""
        self._ensure_initialized()
        return self._beliefs.get(strategy_id)

    def get_beliefs_for_domain(self, domain: DomainType) -> List[StrategyBelief]:
        """Get all beliefs for a domain."""
        self._ensure_initialized()
        return self._domain_beliefs.get(domain, [])

    def select_strategy_thompson(
        self,
        domain: DomainType,
        exclude_ids: set = None
    ) -> Optional[Strategy]:
        """
        Select a strategy using Thompson Sampling.

        Args:
            domain: Domain to select from
            exclude_ids: Strategy IDs to exclude (already in portfolio)

        Returns:
            Selected strategy, or None if no strategies available
        """
        self._ensure_initialized()
        exclude_ids = exclude_ids or set()

        # Get available strategies
        available_beliefs = [
            b for b in self._domain_beliefs[domain]
            if b.strategy_id not in exclude_ids
        ]

        if not available_beliefs:
            return None

        # Thompson Sampling: sample from each Beta distribution
        samples = {
            b.strategy_id: b.sample(self.rng)
            for b in available_beliefs
        }

        # Select strategy with highest sample
        selected_id = max(samples, key=samples.get)
        return STRATEGY_LIBRARY.get(selected_id)

    def select_strategy_to_replace(
        self,
        portfolio_ids: List[str]
    ) -> Optional[str]:
        """
        Select which strategy to replace using Thompson Sampling.

        Returns the strategy with LOWEST sampled value (worst performing).

        Args:
            portfolio_ids: IDs of strategies currently in portfolio

        Returns:
            ID of strategy to replace, or None if portfolio empty
        """
        if not portfolio_ids:
            return None

        self._ensure_initialized()

        # Sample from each strategy in portfolio
        samples = {}
        for sid in portfolio_ids:
            belief = self._beliefs.get(sid)
            if belief:
                samples[sid] = belief.sample(self.rng)
            else:
                samples[sid] = 0.5  # Default for unknown

        # Return strategy with LOWEST sample (to be replaced)
        return min(samples, key=samples.get)

    def update_belief(
        self,
        strategy_id: str,
        won: bool,
        num_active: int = 1
    ):
        """
        Update belief for a strategy.

        Args:
            strategy_id: Strategy to update
            won: Whether agent won
            num_active: Number of active strategies (for discounting)
        """
        self._ensure_initialized()
        belief = self._beliefs.get(strategy_id)
        if belief:
            if num_active > 1:
                belief.update_discounted(won, num_active)
            else:
                belief.update(won)

    def update_all_active(
        self,
        active_ids: List[str],
        won: bool
    ):
        """
        Update beliefs for all active strategies in portfolio.

        Uses discounted credit assignment.
        """
        n = len(active_ids)
        for sid in active_ids:
            self.update_belief(sid, won, n)

    def get_top_strategies(
        self,
        domain: DomainType,
        k: int = 5
    ) -> List[Tuple[Strategy, float]]:
        """
        Get top k strategies by expected value.

        Returns list of (Strategy, expected_value) tuples.
        """
        self._ensure_initialized()
        beliefs = self._domain_beliefs[domain]

        sorted_beliefs = sorted(
            beliefs,
            key=lambda b: b.expected_value(),
            reverse=True
        )

        result = []
        for belief in sorted_beliefs[:k]:
            strategy = STRATEGY_LIBRARY.get(belief.strategy_id)
            if strategy:
                result.append((strategy, belief.expected_value()))

        return result

    def get_statistics(self) -> Dict:
        """Get summary statistics of beliefs."""
        self._ensure_initialized()

        stats = {
            "total_strategies": len(self._beliefs),
            "strategies_tried": sum(1 for b in self._beliefs.values() if b.times_selected > 0),
            "total_selections": sum(b.times_selected for b in self._beliefs.values()),
            "total_wins": sum(b.times_won for b in self._beliefs.values()),
            "by_domain": {}
        }

        for domain in DomainType:
            domain_beliefs = self._domain_beliefs[domain]
            stats["by_domain"][domain.value] = {
                "count": len(domain_beliefs),
                "avg_expected": np.mean([b.expected_value() for b in domain_beliefs]) if domain_beliefs else 0,
                "max_expected": max([b.expected_value() for b in domain_beliefs]) if domain_beliefs else 0,
            }

        return stats


# Global manager instance
BELIEF_MANAGER = StrategyBeliefManager()


def select_strategy(
    domain: DomainType,
    exclude_ids: set = None,
    seed: int = None
) -> Optional[Strategy]:
    """
    Convenience function to select a strategy using Thompson Sampling.

    Args:
        domain: Domain to select from
        exclude_ids: IDs to exclude
        seed: Random seed (creates new manager if provided)

    Returns:
        Selected Strategy or None
    """
    if seed is not None:
        manager = StrategyBeliefManager(seed)
        return manager.select_strategy_thompson(domain, exclude_ids)
    return BELIEF_MANAGER.select_strategy_thompson(domain, exclude_ids)


if __name__ == "__main__":
    # Demo Thompson Sampling
    print("=" * 60)
    print("THOMPSON SAMPLING DEMO")
    print("=" * 60)

    manager = StrategyBeliefManager(seed=42)

    # Simulate selection and updates
    domain = DomainType.ARITHMETIC

    print(f"\nSimulating 50 rounds of strategy selection in {domain.value}...")

    selected_counts = {}
    for i in range(50):
        strategy = manager.select_strategy_thompson(domain)
        if strategy:
            selected_counts[strategy.id] = selected_counts.get(strategy.id, 0) + 1
            # Simulate outcome (some strategies work better)
            won = np.random.random() < (0.3 if "basic" in strategy.name.lower() else 0.7)
            manager.update_belief(strategy.id, won)

    print("\nTop 5 strategies after learning:")
    for strategy, expected in manager.get_top_strategies(domain, k=5):
        print(f"  {strategy.name}: E={expected:.3f}")

    print("\nSelection frequency:")
    for sid, count in sorted(selected_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  {sid}: {count} times")

    print("\n" + "=" * 60)
    print(manager.get_statistics())
