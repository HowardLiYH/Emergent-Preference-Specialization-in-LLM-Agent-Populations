"""
Independent Training Architecture.

Baseline: Each agent learns independently with NO population interaction.
Same inner algorithm (Thompson Sampling), but no competition or sharing.

Expected Result: No specialization, O(N) cost.
"""

import random
from typing import Dict, Tuple

from .base import BaseArchitecture, BaseAgent, Regime


class IndependentTraining(BaseArchitecture):
    """
    Independent Training: Each agent learns separately.
    
    - NO competition
    - NO fitness sharing
    - NO sample sharing
    
    Cost scales as O(N × samples_per_agent).
    """
    
    def __init__(self, n_agents: int, regimes: Dict[str, Regime]):
        super().__init__(n_agents, regimes)
        self.name = "Independent"
    
    def train_step(self, task: Tuple[str, str], regime: str, rng: random.Random,
                   evaluate_fn) -> Dict:
        """
        One agent attempts the task (round-robin or random).
        No competition - just independent learning.
        """
        # Select one agent randomly (simulates independent training)
        agent = rng.choice(self.agents)
        
        # Agent selects tool via Thompson Sampling
        tool = agent.select_tool(regime, rng)
        
        # Evaluate
        question, answer = task
        success, tokens = evaluate_fn(agent, tool, question, answer)
        
        # Update only this agent
        agent.update(regime, tool, success)
        
        self.total_tokens += tokens
        self.generations += 1
        
        return {
            'regime': regime,
            'agent': agent.id,
            'tool': tool,
            'success': success,
            'tokens': tokens,
        }
