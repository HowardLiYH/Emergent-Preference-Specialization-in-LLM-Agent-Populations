"""
Competitive Specialist Ecosystem (CSE) Architecture.

Our method: Competition + Fitness Sharing + Winner-Only Learning.

Expected Result: Full specialization with sublinear cost scaling.
"""

import random
from typing import Dict, Tuple
from collections import defaultdict
import math

from .base import BaseArchitecture, BaseAgent, Regime


class CompetitiveSpecialistEcosystem(BaseArchitecture):
    """
    Our CSE Architecture:
    
    - YES competition (all agents compete)
    - YES fitness sharing (1/√n penalty)
    - YES sample sharing (through competition)
    
    Expected: Full coverage with sublinear cost O(N^0.6).
    """
    
    def __init__(self, n_agents: int, regimes: Dict[str, Regime],
                 fitness_sharing_gamma: float = 0.5):
        super().__init__(n_agents, regimes)
        self.name = "CSE"
        self.gamma = fitness_sharing_gamma
    
    def train_step(self, task: Tuple[str, str], regime: str, rng: random.Random,
                   evaluate_fn) -> Dict:
        """
        Full competition with fitness sharing:
        1. All agents compete on task
        2. Fitness sharing penalizes crowded niches
        3. Only winner updates (winner-only learning)
        """
        question, answer = task
        regime_info = self.regimes[regime]
        
        # Count current specialists for fitness sharing
        specialty_counts = defaultdict(int)
        for agent in self.agents:
            spec = agent.get_specialty()
            if spec:
                specialty_counts[spec] += 1
        
        # All agents attempt task
        results = []
        total_step_tokens = 0
        
        for agent in self.agents:
            tool = agent.select_tool(regime, rng)
            success, tokens = evaluate_fn(agent, tool, question, answer)
            total_step_tokens += tokens
            
            if success:
                # Base score
                tool_bonus = regime_info.tool_bonuses.get(tool, 0.5)
                base_score = tool_bonus + rng.random() * 0.1
                
                # Apply fitness sharing penalty
                n_specialists = specialty_counts.get(regime, 0) + 1
                penalty = 1.0 / (n_specialists ** self.gamma)
                
                final_score = base_score * penalty
            else:
                final_score = 0
            
            results.append((agent, tool, success, final_score))
        
        # Find winner (highest score among successful)
        successful = [(a, t, s, sc) for a, t, s, sc in results if s]
        
        if successful:
            winner, winner_tool, _, _ = max(successful, key=lambda x: x[3])
            winner.update(regime, winner_tool, True)
            winner_id = winner.id
        else:
            winner_id = None
            winner_tool = None
        
        # Losers update with failure (they learn what doesn't work)
        for agent, tool, success, _ in results:
            if not success:
                agent.update(regime, tool, False)
        
        self.total_tokens += total_step_tokens
        self.generations += 1
        
        return {
            'regime': regime,
            'winner': winner_id,
            'winner_tool': winner_tool,
            'n_successful': len(successful),
            'tokens': total_step_tokens,
        }
