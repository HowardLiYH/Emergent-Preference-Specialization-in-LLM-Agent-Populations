"""
MARL Shared Critic Architecture.

Baseline: Centralized value function shared across all agents.
All agents receive the same learning signal.

Expected Result: Homogenization (all become generalists).
"""

import random
from typing import Dict, Tuple
from collections import defaultdict

from .base import BaseArchitecture, BaseAgent, Regime, BetaDistribution


class MARLSharedCritic(BaseArchitecture):
    """
    MARL with Shared Critic: All agents share learning signal.

    - NO competition
    - NO fitness sharing
    - YES sample sharing (via shared critic)

    Expected: All agents converge to same behavior (homogenization).
    """

    def __init__(self, n_agents: int, regimes: Dict[str, Regime]):
        super().__init__(n_agents, regimes)
        self.name = "MARL_Shared"

        # Shared critic: global beliefs about tool effectiveness
        self.shared_beliefs: Dict[str, Dict[str, BetaDistribution]] = {}
        for regime in regimes:
            self.shared_beliefs[regime] = {
                f'L{i}': BetaDistribution() for i in range(5)
            }

    def train_step(self, task: Tuple[str, str], regime: str, rng: random.Random,
                   evaluate_fn) -> Dict:
        """
        All agents share the same value estimate from centralized critic.
        One agent acts, but ALL agents learn from the outcome.
        """
        # Select one agent to act
        actor = rng.choice(self.agents)

        # Tool selection based on SHARED beliefs
        samples = {t: self.shared_beliefs[regime][t].sample(rng)
                   for t in actor.available_tools}
        tool = max(samples, key=samples.get)

        # Evaluate
        question, answer = task
        success, tokens = evaluate_fn(actor, tool, question, answer)

        # Update SHARED critic (all agents learn)
        self.shared_beliefs[regime][tool].update(success)

        # Also update all agents' individual beliefs (shared learning)
        for agent in self.agents:
            agent.update(regime, tool, success)

        self.total_tokens += tokens
        self.generations += 1

        return {
            'regime': regime,
            'actor': actor.id,
            'tool': tool,
            'success': success,
            'tokens': tokens,
            'all_agents_updated': True,
        }
