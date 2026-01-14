"""
Non-uniform regime configuration.

Defines regimes with varying frequencies, rewards, and difficulties.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import random


@dataclass
class RegimeConfig:
    """Configuration for a single regime."""
    name: str
    optimal_tool: str  # Hidden from agents
    frequency: float   # Probability of this regime being selected
    reward: float      # Reward multiplier for wins
    difficulty: float  # Task difficulty (0-1)
    description: str

    def expected_value(self, n_specialists: int = 1, gamma: float = 0.5) -> float:
        """
        Calculate expected value for this regime.

        Uses the fitness sharing formula from Theorem 4.

        Args:
            n_specialists: Number of agents specializing in this regime
            gamma: Fitness sharing exponent

        Returns:
            Expected value per competition
        """
        if n_specialists == 0:
            n_specialists = 1  # Avoid division by zero

        return (self.frequency * self.reward * self.difficulty) / (n_specialists ** (1 + gamma))


# Regime definitions - optimal tool is HIDDEN from agents
REGIMES: Dict[str, RegimeConfig] = {
    'pure_qa': RegimeConfig(
        name='pure_qa',
        optimal_tool='L0',  # Base LLM
        frequency=0.30,
        reward=1.0,
        difficulty=0.2,
        description='Pure Q&A tasks requiring only world knowledge'
    ),

    'code_math': RegimeConfig(
        name='code_math',
        optimal_tool='L1',  # Python execution
        frequency=0.25,
        reward=2.0,
        difficulty=0.5,
        description='Mathematical and coding tasks requiring computation'
    ),

    'chart_analysis': RegimeConfig(
        name='chart_analysis',
        optimal_tool='L2',  # Vision
        frequency=0.15,
        reward=3.0,
        difficulty=0.7,
        description='Chart and image analysis tasks'
    ),

    'document_qa': RegimeConfig(
        name='document_qa',
        optimal_tool='L3',  # RAG
        frequency=0.20,
        reward=2.5,
        difficulty=0.6,
        description='Questions requiring document retrieval'
    ),

    'realtime_data': RegimeConfig(
        name='realtime_data',
        optimal_tool='L4',  # Web
        frequency=0.10,
        reward=4.0,
        difficulty=0.8,
        description='Tasks requiring real-time data from the web'
    ),
}


def get_regime_names() -> List[str]:
    """Get list of all regime names."""
    return list(REGIMES.keys())


def get_regime(name: str) -> Optional[RegimeConfig]:
    """Get a regime by name."""
    return REGIMES.get(name)


def get_all_regimes() -> List[RegimeConfig]:
    """Get all regime configurations."""
    return list(REGIMES.values())


def validate_frequencies() -> bool:
    """Check that frequencies sum to 1.0."""
    total = sum(r.frequency for r in REGIMES.values())
    return abs(total - 1.0) < 0.001


def compute_equilibrium_distribution(
    n_agents: int = 12,
    gamma: float = 0.5
) -> Dict[str, float]:
    """
    Compute theoretical equilibrium distribution of specialists.

    According to Theorem 4:
        n_r ∝ (f_r × R_r × D_r)^(2/3)

    Args:
        n_agents: Total number of agents
        gamma: Fitness sharing exponent

    Returns:
        Dict mapping regime names to expected specialist counts
    """
    exponent = 2 / 3  # When gamma = 0.5: 1/(1+gamma) = 2/3

    # Calculate raw proportions
    raw_proportions = {}
    for name, regime in REGIMES.items():
        value = (regime.frequency * regime.reward * regime.difficulty) ** exponent
        raw_proportions[name] = value

    # Normalize to sum to n_agents
    total = sum(raw_proportions.values())

    distribution = {
        name: (prop / total) * n_agents
        for name, prop in raw_proportions.items()
    }

    return distribution


# Verify frequencies on import
assert validate_frequencies(), "Regime frequencies must sum to 1.0"
