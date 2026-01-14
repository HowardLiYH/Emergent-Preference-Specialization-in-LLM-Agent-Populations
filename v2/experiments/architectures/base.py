"""
Base classes for architecture comparison experiments.

All architectures use the SAME inner algorithm (Thompson Sampling).
Only the population structure differs.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


@dataclass
class BetaDistribution:
    """Thompson Sampling belief distribution."""
    alpha: float = 1.0
    beta: float = 1.0
    
    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(self.alpha, self.beta)
    
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)
    
    def update(self, success: bool):
        if success:
            self.alpha += 1
        else:
            self.beta += 1


@dataclass
class BaseAgent:
    """
    Base agent class with Thompson Sampling.
    All architectures use this same inner algorithm.
    """
    id: str
    tool_level: int = 4
    beliefs: Dict[str, Dict[str, BetaDistribution]] = field(default_factory=dict)
    wins: int = 0
    regime_wins: Dict[str, int] = field(default_factory=dict)
    total_attempts: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        self.available_tools = [f'L{i}' for i in range(self.tool_level + 1)]
    
    def initialize_beliefs(self, regimes: List[str]):
        """Initialize Thompson Sampling beliefs for all regimes."""
        for regime in regimes:
            if regime not in self.beliefs:
                self.beliefs[regime] = {t: BetaDistribution() for t in self.available_tools}
                self.total_attempts[regime] = 0
    
    def select_tool(self, regime: str, rng: random.Random) -> str:
        """Thompson Sampling for tool selection."""
        if regime not in self.beliefs:
            return random.choice(self.available_tools)
        samples = {t: self.beliefs[regime][t].sample(rng) for t in self.available_tools}
        return max(samples, key=samples.get)
    
    def update(self, regime: str, tool: str, success: bool):
        """Update beliefs based on outcome."""
        if regime in self.beliefs and tool in self.beliefs[regime]:
            self.beliefs[regime][tool].update(success)
        self.total_attempts[regime] = self.total_attempts.get(regime, 0) + 1
        if success:
            self.wins += 1
            self.regime_wins[regime] = self.regime_wins.get(regime, 0) + 1
    
    def get_specialty(self) -> Optional[str]:
        """Get regime where agent has most wins."""
        if not self.regime_wins:
            return None
        return max(self.regime_wins, key=self.regime_wins.get)
    
    def get_confidence(self, regime: str) -> float:
        """Get confidence in a regime (mean of best tool belief)."""
        if regime not in self.beliefs:
            return 0.5
        means = {t: self.beliefs[regime][t].mean() for t in self.available_tools}
        return max(means.values())


@dataclass
class Regime:
    """Regime definition with non-uniform properties."""
    name: str
    optimal_tool: str
    frequency: float
    reward: float
    difficulty: float
    tool_bonuses: Dict[str, float] = field(default_factory=dict)


class BaseArchitecture(ABC):
    """
    Abstract base class for population architectures.
    All concrete architectures implement train() differently.
    """
    
    def __init__(self, n_agents: int, regimes: Dict[str, Regime]):
        self.n_agents = n_agents
        self.regimes = regimes
        self.agents = [BaseAgent(id=f"agent_{i}") for i in range(n_agents)]
        
        # Initialize all agents' beliefs
        for agent in self.agents:
            agent.initialize_beliefs(list(regimes.keys()))
        
        # Metrics tracking
        self.total_tokens = 0
        self.generations = 0
    
    @abstractmethod
    def train_step(self, task: Tuple[str, str], regime: str, rng: random.Random, 
                   evaluate_fn) -> Dict:
        """
        Execute one training step.
        
        Args:
            task: (question, answer) tuple
            regime: The regime this task belongs to
            rng: Random number generator
            evaluate_fn: Function to evaluate agent response
            
        Returns:
            Dict with step results
        """
        pass
    
    def compute_coverage(self) -> float:
        """Compute regime coverage."""
        covered = set()
        for agent in self.agents:
            spec = agent.get_specialty()
            if spec:
                covered.add(spec)
        return len(covered) / len(self.regimes) if self.regimes else 0
    
    def compute_sci(self) -> float:
        """Compute Specialization Concentration Index."""
        specialty_counts = defaultdict(int)
        for agent in self.agents:
            spec = agent.get_specialty()
            if spec:
                specialty_counts[spec] += 1
        
        if not specialty_counts:
            return 0.0
        
        total = sum(specialty_counts.values())
        n_regimes = len(self.regimes)
        
        hhi = sum((c / total) ** 2 for c in specialty_counts.values())
        return (hhi - 1/n_regimes) / (1 - 1/n_regimes) if n_regimes > 1 else hhi
    
    def get_results(self) -> Dict:
        """Get final results."""
        return {
            'coverage': self.compute_coverage(),
            'sci': self.compute_sci(),
            'total_tokens': self.total_tokens,
            'generations': self.generations,
            'agent_wins': {a.id: a.wins for a in self.agents},
            'agent_specialties': {a.id: a.get_specialty() for a in self.agents},
        }
