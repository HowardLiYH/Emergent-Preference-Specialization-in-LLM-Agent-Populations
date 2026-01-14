"""
Hierarchical Competition Architecture.

Two-level competition: 
1. First level: Agents compete within specialty groups
2. Second level: Group winners compete for regime tasks

Expected Result: Faster convergence with group structure.
"""

import random
from typing import Dict, Tuple, List
from collections import defaultdict

from .base import BaseArchitecture, BaseAgent, Regime


class HierarchicalCompetition(BaseArchitecture):
    """
    Hierarchical Competition: Two-level tournament structure.
    
    - Level 1: Intra-group competition (agents within same specialty)
    - Level 2: Inter-group competition (group representatives)
    
    Expected: Faster specialization through structured competition.
    """
    
    def __init__(self, n_agents: int, regimes: Dict[str, Regime], 
                 n_groups: int = None, gamma: float = 0.5):
        super().__init__(n_agents, regimes)
        self.name = "Hierarchical"
        self.gamma = gamma
        
        # Create groups (one per regime if not specified)
        self.n_groups = n_groups or len(regimes)
        self.groups: Dict[str, List[BaseAgent]] = defaultdict(list)
        
        # Initially assign agents to groups round-robin
        regime_names = list(regimes.keys())
        for i, agent in enumerate(self.agents):
            group_name = regime_names[i % len(regime_names)]
            self.groups[group_name].append(agent)
    
    def train_step(self, task: Tuple[str, str], regime: str, rng: random.Random,
                   evaluate_fn) -> Dict:
        """
        Two-level competition:
        1. Select representative from each group
        2. Representatives compete
        3. Winner updates
        """
        q, a = task
        regime_info = self.regimes[regime]
        
        # Level 1: Select representative from each group
        representatives = []
        level1_tokens = 0
        
        for group_name, group_agents in self.groups.items():
            if not group_agents:
                continue
            
            # Within-group competition (simplified: random selection)
            rep = rng.choice(group_agents)
            tool = rep.select_tool(regime, rng)
            success, tokens = evaluate_fn(rep, tool, q, a)
            level1_tokens += tokens
            
            if success:
                representatives.append((rep, tool, group_name))
        
        # Level 2: Inter-group competition
        if representatives:
            # Apply fitness sharing at group level
            group_counts = defaultdict(int)
            for agent in self.agents:
                spec = agent.get_specialty()
                if spec:
                    group_counts[spec] += 1
            
            scored = []
            for rep, tool, group in representatives:
                tool_bonus = regime_info.tool_bonuses.get(tool, 0.5)
                n_in_group = len(self.groups[group])
                penalty = 1.0 / (n_in_group ** self.gamma)
                score = tool_bonus * penalty + rng.random() * 0.1
                scored.append((rep, tool, score))
            
            winner, winner_tool, _ = max(scored, key=lambda x: x[2])
            winner.update(regime, winner_tool, True)
            
            # Reassign winner to regime's group if different
            current_group = None
            for g, agents in self.groups.items():
                if winner in agents:
                    current_group = g
                    break
            
            if current_group and current_group != regime:
                # Migration: move winner to winning regime's group
                self.groups[current_group].remove(winner)
                self.groups[regime].append(winner)
        
        self.total_tokens += level1_tokens
        self.generations += 1
        
        return {
            'regime': regime,
            'n_representatives': len(representatives),
            'tokens': level1_tokens,
        }
