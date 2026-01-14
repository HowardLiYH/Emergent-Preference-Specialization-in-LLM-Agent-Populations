"""
Subset Competition with Epsilon-Exploration.

Implements Prof. Levine's suggestion for efficient training with exploration.
Reduces training cost by only having top-K agents compete per task.
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class SubsetAgent:
    """Agent that participates in subset competition."""
    id: int
    regimes: List[str]
    
    # Tool beliefs (Thompson Sampling)
    tool_alpha: Dict[str, Dict[str, float]] = field(default_factory=dict)
    tool_beta: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Regime confidence
    regime_wins: Dict[str, int] = field(default_factory=dict)
    regime_attempts: Dict[str, int] = field(default_factory=dict)
    
    # Specialty tracking
    specialty: Optional[str] = None
    
    def __post_init__(self):
        tools = ['L0', 'L1', 'L2', 'L3', 'L4']
        for regime in self.regimes:
            self.tool_alpha[regime] = {t: 1.0 for t in tools}
            self.tool_beta[regime] = {t: 1.0 for t in tools}
            self.regime_wins[regime] = 0
            self.regime_attempts[regime] = 0
    
    def get_confidence(self, regime: str) -> float:
        """Get confidence for a regime based on win rate."""
        attempts = self.regime_attempts.get(regime, 0)
        if attempts == 0:
            return 0.5  # Prior
        wins = self.regime_wins.get(regime, 0)
        return wins / attempts
    
    def select_tool(self, regime: str, rng: random.Random) -> str:
        """Thompson Sampling for tool selection."""
        samples = {}
        for tool in self.tool_alpha[regime]:
            alpha = self.tool_alpha[regime][tool]
            beta = self.tool_beta[regime][tool]
            samples[tool] = rng.betavariate(alpha, beta)
        return max(samples, key=samples.get)
    
    def update_tool_belief(self, regime: str, tool: str, success: bool):
        """Update Thompson Sampling beliefs."""
        if success:
            self.tool_alpha[regime][tool] += 1
        else:
            self.tool_beta[regime][tool] += 1
    
    def update_on_win(self, regime: str):
        """Update after winning a competition."""
        self.regime_wins[regime] = self.regime_wins.get(regime, 0) + 1
        self.regime_attempts[regime] = self.regime_attempts.get(regime, 0) + 1
        
        # Update specialty if consistently winning
        if self.regime_wins[regime] >= 3:
            win_rate = self.regime_wins[regime] / self.regime_attempts[regime]
            if win_rate > 0.6:
                self.specialty = regime
    
    def update_on_loss(self, regime: str):
        """Update after losing a competition."""
        self.regime_attempts[regime] = self.regime_attempts.get(regime, 0) + 1


class SubsetCSE:
    """
    Competitive Specialist Ecosystem with Subset Competition.
    
    Only top-K agents compete per task, reducing training cost by N/K.
    Includes epsilon-exploration to prevent missing specialists.
    """
    
    def __init__(
        self,
        n_agents: int,
        regimes: List[str],
        K: int = 3,
        gamma: float = 0.5,
        epsilon: float = 0.1,
        timeout: float = 30.0  # Prof. Abbeel suggestion
    ):
        self.n_agents = n_agents
        self.regimes = regimes
        self.K = K
        self.gamma = gamma
        self.epsilon = epsilon
        self.timeout = timeout
        
        # Create agents
        self.agents = [
            SubsetAgent(id=i, regimes=regimes)
            for i in range(n_agents)
        ]
        
        # Tracking
        self.total_tokens = 0
        self.total_llm_calls = 0
        self.generation = 0
        self.history = []
    
    def select_competitors(self, regime: str, rng: random.Random) -> List[SubsetAgent]:
        """
        Select top-K agents with epsilon-exploration.
        
        With probability epsilon, replace the least confident
        agent in top-K with a random non-top-K agent.
        """
        # Sort by confidence
        confidences = [(a, a.get_confidence(regime)) for a in self.agents]
        sorted_agents = sorted(confidences, key=lambda x: -x[1])
        
        # Select top-K
        top_k = [a for a, _ in sorted_agents[:self.K]]
        
        # Epsilon-exploration
        if rng.random() < self.epsilon:
            others = [a for a, _ in sorted_agents[self.K:]]
            if others:
                # Replace least confident in top-K with random other
                explorer = rng.choice(others)
                top_k[-1] = explorer
        
        return top_k
    
    def compute_fitness_shared_score(
        self,
        agent: SubsetAgent,
        base_score: float,
        regime: str
    ) -> float:
        """Apply fitness sharing penalty based on niche crowding."""
        # Count agents in same niche
        niche_count = sum(
            1 for a in self.agents 
            if a.specialty == regime
        )
        niche_count = max(1, niche_count)
        
        # Fitness sharing: penalize crowded niches
        penalty = 1.0 / (niche_count ** self.gamma)
        return base_score * penalty
    
    def train_step(
        self,
        task: Tuple[str, str],  # (question, answer)
        regime: str,
        rng: random.Random,
        evaluate_fn  # Function to evaluate agent response
    ) -> Dict:
        """
        Run one training step with subset competition.
        
        Returns dict with winner info, tokens used, etc.
        """
        competitors = self.select_competitors(regime, rng)
        
        results = []
        for agent in competitors:
            tool = agent.select_tool(regime, rng)
            
            # Evaluate agent
            success, tokens, response = evaluate_fn(
                agent, tool, task[0], task[1]
            )
            
            self.total_tokens += tokens
            self.total_llm_calls += 1
            
            # Compute score with fitness sharing
            base_score = 1.0 if success else 0.0
            score = self.compute_fitness_shared_score(agent, base_score, regime)
            
            results.append({
                'agent': agent,
                'tool': tool,
                'success': success,
                'score': score,
                'tokens': tokens,
            })
            
            # Update tool belief
            agent.update_tool_belief(regime, tool, success)
        
        # Determine winner
        if results:
            winner_result = max(results, key=lambda r: r['score'])
            winner = winner_result['agent']
            
            # Update all agents
            for r in results:
                if r['agent'] == winner:
                    r['agent'].update_on_win(regime)
                else:
                    r['agent'].update_on_loss(regime)
        else:
            winner = None
            winner_result = None
        
        self.generation += 1
        
        step_result = {
            'generation': self.generation,
            'regime': regime,
            'n_competitors': len(competitors),
            'winner_id': winner.id if winner else None,
            'winner_tool': winner_result['tool'] if winner_result else None,
            'tokens_used': sum(r['tokens'] for r in results),
            'any_success': any(r['success'] for r in results),
        }
        self.history.append(step_result)
        
        return step_result
    
    def get_coverage(self) -> float:
        """Compute regime coverage by specialists."""
        covered = set()
        for agent in self.agents:
            if agent.specialty:
                covered.add(agent.specialty)
        return len(covered) / len(self.regimes)
    
    def get_specialists(self) -> Dict[str, List[SubsetAgent]]:
        """Get specialists grouped by regime."""
        specialists = {r: [] for r in self.regimes}
        for agent in self.agents:
            if agent.specialty:
                specialists[agent.specialty].append(agent)
        return specialists
    
    def get_metrics(self) -> Dict:
        """Get current training metrics."""
        specialists = self.get_specialists()
        
        return {
            'total_tokens': self.total_tokens,
            'total_llm_calls': self.total_llm_calls,
            'generations': self.generation,
            'coverage': self.get_coverage(),
            'n_specialists': sum(len(s) for s in specialists.values()),
            'specialists_per_regime': {
                r: len(s) for r, s in specialists.items()
            },
            'cost_per_generation': self.total_tokens / max(1, self.generation),
            'tokens_per_call': self.total_tokens / max(1, self.total_llm_calls),
        }
    
    def get_failure_rate(self) -> float:
        """Estimate failure rate based on uncovered regimes."""
        coverage = self.get_coverage()
        # Assume uncovered regimes have 50% failure, covered have 10%
        uncovered_rate = 1 - coverage
        return uncovered_rate * 0.5 + coverage * 0.1


class IndependentTraining:
    """
    Baseline: Independent training without competition.
    Each agent learns independently with Thompson Sampling.
    """
    
    def __init__(self, n_agents: int, regimes: List[str]):
        self.n_agents = n_agents
        self.regimes = regimes
        
        self.agents = [
            SubsetAgent(id=i, regimes=regimes)
            for i in range(n_agents)
        ]
        
        self.total_tokens = 0
        self.total_llm_calls = 0
        self.generation = 0
        self.history = []
    
    def train_step(
        self,
        task: Tuple[str, str],
        regime: str,
        rng: random.Random,
        evaluate_fn
    ) -> Dict:
        """Each agent independently attempts the task."""
        # Randomly select one agent (fair comparison with subset K=1)
        agent = rng.choice(self.agents)
        tool = agent.select_tool(regime, rng)
        
        success, tokens, response = evaluate_fn(
            agent, tool, task[0], task[1]
        )
        
        self.total_tokens += tokens
        self.total_llm_calls += 1
        
        agent.update_tool_belief(regime, tool, success)
        
        if success:
            agent.update_on_win(regime)
        
        self.generation += 1
        
        step_result = {
            'generation': self.generation,
            'regime': regime,
            'agent_id': agent.id,
            'tool': tool,
            'success': success,
            'tokens_used': tokens,
        }
        self.history.append(step_result)
        
        return step_result
    
    def get_coverage(self) -> float:
        """Compute regime coverage."""
        covered = set()
        for agent in self.agents:
            if agent.specialty:
                covered.add(agent.specialty)
        return len(covered) / len(self.regimes)
    
    def get_metrics(self) -> Dict:
        """Get training metrics."""
        return {
            'total_tokens': self.total_tokens,
            'total_llm_calls': self.total_llm_calls,
            'generations': self.generation,
            'coverage': self.get_coverage(),
        }
    
    def get_failure_rate(self) -> float:
        """Estimate failure rate."""
        coverage = self.get_coverage()
        uncovered_rate = 1 - coverage
        return uncovered_rate * 0.5 + coverage * 0.1
