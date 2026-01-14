"""
Tournament Selection Architecture.

Evolutionary baseline: Best agents in tournament survive.
Competition exists but NO fitness sharing.

Expected Result: Winner-take-all (one agent dominates).
"""

import random
from typing import Dict, Tuple, List

from .base import BaseArchitecture, BaseAgent, Regime


class TournamentSelection(BaseArchitecture):
    """
    Tournament Selection: Evolutionary competition without fitness sharing.
    
    - YES competition (tournament)
    - NO fitness sharing
    - Partial sample sharing
    
    Expected: One or few agents dominate (winner-take-all).
    """
    
    def __init__(self, n_agents: int, regimes: Dict[str, Regime], 
                 tournament_size: int = 4):
        super().__init__(n_agents, regimes)
        self.name = "Tournament"
        self.tournament_size = min(tournament_size, n_agents)
    
    def train_step(self, task: Tuple[str, str], regime: str, rng: random.Random,
                   evaluate_fn) -> Dict:
        """
        Tournament selection: Sample subset, evaluate all, winner learns.
        NO fitness sharing - pure winner-take-all.
        """
        # Select tournament contestants
        contestants = rng.sample(self.agents, self.tournament_size)
        
        # Each contestant attempts the task
        results = []
        total_step_tokens = 0
        
        question, answer = task
        
        for agent in contestants:
            tool = agent.select_tool(regime, rng)
            success, tokens = evaluate_fn(agent, tool, question, answer)
            total_step_tokens += tokens
            
            # Score: success gives 1, tie-break by random
            score = (1 if success else 0) + rng.random() * 0.01
            results.append((agent, tool, success, score))
        
        # Winner is highest score
        winner, winner_tool, winner_success, _ = max(results, key=lambda x: x[3])
        
        # ONLY winner updates (no fitness sharing)
        if winner_success:
            winner.update(regime, winner_tool, True)
        
        # Losers don't update at all (unlike CSE where they update with failure)
        
        self.total_tokens += total_step_tokens
        self.generations += 1
        
        return {
            'regime': regime,
            'tournament_size': self.tournament_size,
            'winner': winner.id if winner_success else None,
            'winner_tool': winner_tool if winner_success else None,
            'tokens': total_step_tokens,
        }
