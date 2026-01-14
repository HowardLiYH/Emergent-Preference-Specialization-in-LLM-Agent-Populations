"""
Regime sampler with non-uniform distribution.
"""

import random
from typing import Optional, List, Dict
from .config import REGIMES, RegimeConfig, get_regime_names


class RegimeSampler:
    """
    Samples regimes according to their configured frequencies.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize regime sampler.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
        self.regimes = list(REGIMES.values())
        self.regime_names = [r.name for r in self.regimes]
        self.frequencies = [r.frequency for r in self.regimes]

        # Track sampling history
        self.history: List[str] = []

    def sample(self) -> str:
        """
        Sample a regime according to frequencies.

        Returns:
            Regime name
        """
        regime = self.rng.choices(
            self.regime_names,
            weights=self.frequencies,
            k=1
        )[0]

        self.history.append(regime)
        return regime

    def sample_n(self, n: int) -> List[str]:
        """
        Sample n regimes.

        Args:
            n: Number of regimes to sample

        Returns:
            List of regime names
        """
        return [self.sample() for _ in range(n)]

    def sample_with_config(self) -> RegimeConfig:
        """
        Sample a regime and return its full configuration.

        Returns:
            RegimeConfig object
        """
        name = self.sample()
        return REGIMES[name]

    def get_empirical_distribution(self) -> Dict[str, float]:
        """
        Get empirical distribution from sampling history.

        Returns:
            Dict mapping regime names to observed frequencies
        """
        if not self.history:
            return {name: 0.0 for name in self.regime_names}

        counts = {name: 0 for name in self.regime_names}
        for regime in self.history:
            counts[regime] += 1

        total = len(self.history)
        return {name: count / total for name, count in counts.items()}

    def get_expected_distribution(self) -> Dict[str, float]:
        """
        Get expected (configured) distribution.

        Returns:
            Dict mapping regime names to configured frequencies
        """
        return {r.name: r.frequency for r in self.regimes}

    def compare_distributions(self) -> Dict[str, Dict[str, float]]:
        """
        Compare empirical vs expected distributions.

        Returns:
            Dict with 'expected', 'empirical', and 'error' for each regime
        """
        expected = self.get_expected_distribution()
        empirical = self.get_empirical_distribution()

        return {
            name: {
                'expected': expected[name],
                'empirical': empirical[name],
                'error': abs(expected[name] - empirical[name])
            }
            for name in self.regime_names
        }

    def reset_history(self) -> None:
        """Clear sampling history."""
        self.history.clear()

    def set_seed(self, seed: int) -> None:
        """Set random seed."""
        self.rng = random.Random(seed)


class StratifiedRegimeSampler(RegimeSampler):
    """
    Stratified sampler that ensures each regime is sampled
    proportionally over a fixed window.
    """

    def __init__(
        self,
        window_size: int = 100,
        seed: Optional[int] = None
    ):
        """
        Initialize stratified sampler.

        Args:
            window_size: Size of stratification window
            seed: Random seed
        """
        super().__init__(seed)
        self.window_size = window_size
        self._queue: List[str] = []
        self._fill_queue()

    def _fill_queue(self) -> None:
        """Fill the queue with stratified samples."""
        self._queue = []

        for regime in self.regimes:
            count = int(regime.frequency * self.window_size)
            self._queue.extend([regime.name] * count)

        # Handle rounding by adding remaining samples
        while len(self._queue) < self.window_size:
            regime = self.rng.choices(
                self.regime_names,
                weights=self.frequencies,
                k=1
            )[0]
            self._queue.append(regime)

        # Shuffle
        self.rng.shuffle(self._queue)

    def sample(self) -> str:
        """
        Sample a regime (stratified).

        Returns:
            Regime name
        """
        if not self._queue:
            self._fill_queue()

        regime = self._queue.pop()
        self.history.append(regime)
        return regime
