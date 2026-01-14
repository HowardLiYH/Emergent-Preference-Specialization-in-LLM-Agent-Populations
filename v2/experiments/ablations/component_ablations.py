"""
Component Ablation Studies.

Tests the necessity of each CSE component:
1. Fitness Sharing (1/√n penalty)
2. Competition (winner selection)
3. Memory System

Each ablation removes ONE component to measure impact.
"""

import random
from typing import Dict, List, Tuple
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class BetaDistribution:
    alpha: float = 1.0
    beta: float = 1.0
    def sample(self, rng): return rng.betavariate(self.alpha, self.beta)
    def update(self, success):
        if success: self.alpha += 1
        else: self.beta += 1


@dataclass
class AblationAgent:
    id: str
    beliefs: Dict = field(default_factory=dict)
    wins: int = 0
    regime_wins: Dict = field(default_factory=dict)
    available_tools: List = field(default_factory=lambda: ['L0', 'L1', 'L2', 'L3', 'L4'])
    memories: List = field(default_factory=list)
    
    def initialize_beliefs(self, regimes):
        for regime in regimes:
            if regime not in self.beliefs:
                self.beliefs[regime] = {t: BetaDistribution() for t in self.available_tools}
    
    def select_tool(self, regime, rng):
        if regime not in self.beliefs: return rng.choice(self.available_tools)
        samples = {t: self.beliefs[regime][t].sample(rng) for t in self.available_tools}
        return max(samples, key=samples.get)
    
    def update(self, regime, tool, success, add_memory=False):
        if regime in self.beliefs and tool in self.beliefs[regime]:
            self.beliefs[regime][tool].update(success)
        if success:
            self.wins += 1
            self.regime_wins[regime] = self.regime_wins.get(regime, 0) + 1
            if add_memory:
                self.memories.append({'regime': regime, 'tool': tool})
    
    def get_specialty(self):
        if not self.regime_wins: return None
        return max(self.regime_wins, key=self.regime_wins.get)


class AblationExperiment:
    """Base class for ablation experiments."""
    
    def __init__(self, n_agents: int, regimes: Dict, config: Dict):
        self.n_agents = n_agents
        self.regimes = regimes
        self.config = config
        self.agents = [AblationAgent(id=f'agent_{i}') for i in range(n_agents)]
        for agent in self.agents:
            agent.initialize_beliefs(list(regimes.keys()))
        self.total_tokens = 0
    
    def compute_coverage(self):
        covered = {a.get_specialty() for a in self.agents if a.get_specialty()}
        return len(covered) / len(self.regimes) if self.regimes else 0
    
    def compute_sci(self):
        counts = defaultdict(int)
        for a in self.agents:
            spec = a.get_specialty()
            if spec: counts[spec] += 1
        if not counts: return 0.0
        total = sum(counts.values())
        n = len(self.regimes)
        hhi = sum((c / total) ** 2 for c in counts.values())
        return (hhi - 1/n) / (1 - 1/n) if n > 1 else hhi


class FullCSE(AblationExperiment):
    """Full CSE with all components."""
    name = "Full CSE"
    
    def train_step(self, task, regime, rng, evaluate_fn):
        q, a = task
        regime_info = self.regimes[regime]
        
        # Count specialists for fitness sharing
        spec_counts = defaultdict(int)
        for agent in self.agents:
            spec = agent.get_specialty()
            if spec: spec_counts[spec] += 1
        
        # All agents compete
        results = []
        for agent in self.agents:
            tool = agent.select_tool(regime, rng)
            success, tokens = evaluate_fn(agent, tool, q, a)
            self.total_tokens += tokens
            
            if success:
                tool_bonus = regime_info.tool_bonuses.get(tool, 0.5)
                n_spec = spec_counts.get(regime, 0) + 1
                penalty = 1.0 / (n_spec ** 0.5)  # Fitness sharing
                score = tool_bonus * penalty + rng.random() * 0.1
            else:
                score = 0
            results.append((agent, tool, success, score))
        
        # Winner selection (competition)
        successful = [r for r in results if r[2]]
        if successful:
            winner, winner_tool, _, _ = max(successful, key=lambda x: x[3])
            winner.update(regime, winner_tool, True, add_memory=True)
        
        for agent, tool, success, _ in results:
            if not success:
                agent.update(regime, tool, False)


class NoFitnessSharing(AblationExperiment):
    """Ablation: Remove fitness sharing (1/√n penalty)."""
    name = "No Fitness Sharing"
    
    def train_step(self, task, regime, rng, evaluate_fn):
        q, a = task
        regime_info = self.regimes[regime]
        
        # All agents compete (NO fitness sharing)
        results = []
        for agent in self.agents:
            tool = agent.select_tool(regime, rng)
            success, tokens = evaluate_fn(agent, tool, q, a)
            self.total_tokens += tokens
            
            if success:
                tool_bonus = regime_info.tool_bonuses.get(tool, 0.5)
                # NO penalty - winner takes all
                score = tool_bonus + rng.random() * 0.1
            else:
                score = 0
            results.append((agent, tool, success, score))
        
        # Winner selection still applies
        successful = [r for r in results if r[2]]
        if successful:
            winner, winner_tool, _, _ = max(successful, key=lambda x: x[3])
            winner.update(regime, winner_tool, True, add_memory=True)
        
        for agent, tool, success, _ in results:
            if not success:
                agent.update(regime, tool, False)


class NoCompetition(AblationExperiment):
    """Ablation: Remove competition (random winner selection)."""
    name = "No Competition"
    
    def train_step(self, task, regime, rng, evaluate_fn):
        q, a = task
        
        # Random agent selected (no competition)
        agent = rng.choice(self.agents)
        tool = agent.select_tool(regime, rng)
        success, tokens = evaluate_fn(agent, tool, q, a)
        self.total_tokens += tokens
        
        # Random selection always updates the selected agent
        agent.update(regime, tool, success, add_memory=success)


class NoMemory(AblationExperiment):
    """Ablation: Remove memory system."""
    name = "No Memory"
    
    def train_step(self, task, regime, rng, evaluate_fn):
        q, a = task
        regime_info = self.regimes[regime]
        
        spec_counts = defaultdict(int)
        for agent in self.agents:
            spec = agent.get_specialty()
            if spec: spec_counts[spec] += 1
        
        results = []
        for agent in self.agents:
            tool = agent.select_tool(regime, rng)
            success, tokens = evaluate_fn(agent, tool, q, a)
            self.total_tokens += tokens
            
            if success:
                tool_bonus = regime_info.tool_bonuses.get(tool, 0.5)
                n_spec = spec_counts.get(regime, 0) + 1
                penalty = 1.0 / (n_spec ** 0.5)
                score = tool_bonus * penalty + rng.random() * 0.1
            else:
                score = 0
            results.append((agent, tool, success, score))
        
        successful = [r for r in results if r[2]]
        if successful:
            winner, winner_tool, _, _ = max(successful, key=lambda x: x[3])
            # NO memory write
            winner.update(regime, winner_tool, True, add_memory=False)
        
        for agent, tool, success, _ in results:
            if not success:
                agent.update(regime, tool, False)


def run_ablation_study(regimes: Dict, n_agents: int = 12, 
                       n_generations: int = 60, n_seeds: int = 3) -> Dict:
    """
    Run complete ablation study comparing all configurations.
    
    Returns:
        Dict with results for each configuration
    """
    from dataclasses import dataclass
    
    @dataclass
    class Regime:
        name: str
        optimal_tool: str
        frequency: float
        tool_bonuses: Dict
    
    configs = [
        ('full_cse', FullCSE),
        ('no_fitness_sharing', NoFitnessSharing),
        ('no_competition', NoCompetition),
        ('no_memory', NoMemory),
    ]
    
    results = {}
    
    for config_name, config_class in configs:
        coverages = []
        scis = []
        tokens_list = []
        
        for seed in range(n_seeds):
            rng = random.Random(seed)
            experiment = config_class(n_agents, regimes, {})
            
            for gen in range(n_generations):
                regime_name = rng.choices(
                    list(regimes.keys()),
                    weights=[regimes[r].frequency for r in regimes],
                    k=1
                )[0]
                task = (f"Task for {regime_name}", "answer")
                
                def evaluate(agent, tool, q, a):
                    regime_info = regimes[regime_name]
                    tool_bonus = regime_info.tool_bonuses.get(tool, 0.5)
                    success = rng.random() < tool_bonus
                    return success, 20
                
                experiment.train_step(task, regime_name, rng, evaluate)
            
            coverages.append(experiment.compute_coverage())
            scis.append(experiment.compute_sci())
            tokens_list.append(experiment.total_tokens)
        
        results[config_name] = {
            'mean_coverage': sum(coverages) / len(coverages),
            'mean_sci': sum(scis) / len(scis),
            'mean_tokens': sum(tokens_list) / len(tokens_list),
        }
    
    return results


ABLATION_CONFIGS = {
    'full_cse': {'competition': True, 'fitness_sharing': True, 'memory': True},
    'no_fitness_sharing': {'competition': True, 'fitness_sharing': False, 'memory': True},
    'no_competition': {'competition': False, 'fitness_sharing': True, 'memory': True},
    'no_memory': {'competition': True, 'fitness_sharing': True, 'memory': False},
}
