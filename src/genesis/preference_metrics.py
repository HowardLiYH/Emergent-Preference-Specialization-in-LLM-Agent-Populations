"""
Preference Metrics for Synthetic Rule Domains

Measures how agents develop preferences for specific rules through competition.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from .synthetic_rules import RuleType


@dataclass
class AgentPreference:
    """Tracks an agent's preference development over time."""
    agent_id: str
    
    # Win/attempt tracking by rule
    wins_by_rule: Dict[str, int] = field(default_factory=dict)
    attempts_by_rule: Dict[str, int] = field(default_factory=dict)
    
    # Historical preference snapshots (for stability analysis)
    preference_history: List[Dict[str, float]] = field(default_factory=list)
    
    def record_attempt(self, rule_type: RuleType, won: bool):
        """Record an attempt on a rule."""
        rule_key = rule_type.value
        self.attempts_by_rule[rule_key] = self.attempts_by_rule.get(rule_key, 0) + 1
        if won:
            self.wins_by_rule[rule_key] = self.wins_by_rule.get(rule_key, 0) + 1
    
    def get_rpi(self, rule_type: RuleType) -> float:
        """Get Rule Preference Index for a specific rule."""
        rule_key = rule_type.value
        attempts = self.attempts_by_rule.get(rule_key, 0)
        wins = self.wins_by_rule.get(rule_key, 0)
        return wins / attempts if attempts > 0 else 0.0
    
    def get_all_rpi(self) -> Dict[str, float]:
        """Get RPI for all rules."""
        rpis = {}
        for rule_type in RuleType:
            rpis[rule_type.value] = self.get_rpi(rule_type)
        return rpis
    
    def get_primary_preference(self) -> Optional[RuleType]:
        """Get the rule this agent prefers most (highest RPI)."""
        rpis = self.get_all_rpi()
        if not rpis or max(rpis.values()) == 0:
            return None
        best_rule = max(rpis.keys(), key=lambda k: rpis[k])
        return RuleType(best_rule)
    
    def get_preference_strength(self) -> float:
        """How specialized is this agent? (max RPI - mean RPI)"""
        rpis = list(self.get_all_rpi().values())
        if not rpis or max(rpis) == 0:
            return 0.0
        return max(rpis) - np.mean(rpis)
    
    def snapshot_preferences(self):
        """Save current preferences to history."""
        self.preference_history.append(self.get_all_rpi().copy())
    
    def get_preference_stability(self, window: int = 10) -> float:
        """
        Calculate Preference Stability Index (PSI).
        
        Measures correlation between preferences at different time points.
        High PSI = stable preferences (not random drift).
        """
        if len(self.preference_history) < window:
            return 0.0
        
        # Compare current vs past preferences
        current = list(self.preference_history[-1].values())
        past = list(self.preference_history[-window].values())
        
        if sum(current) == 0 or sum(past) == 0:
            return 0.0
        
        # Pearson correlation
        correlation = np.corrcoef(current, past)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0


def compute_rpi_variance(agents: List[AgentPreference]) -> float:
    """
    Compute variance in Rule Preference Index across agents.
    
    High variance = agents have distinct preferences.
    Low variance = agents are similar (no differentiation).
    """
    if not agents:
        return 0.0
    
    # Get preference strength for each agent
    strengths = [agent.get_preference_strength() for agent in agents]
    
    return float(np.var(strengths))


def compute_population_psi(agents: List[AgentPreference], window: int = 10) -> float:
    """
    Compute average Preference Stability Index across population.
    
    High PSI = preferences are stable across time.
    """
    if not agents:
        return 0.0
    
    stabilities = [agent.get_preference_stability(window) for agent in agents]
    return float(np.mean(stabilities))


def compute_preference_diversity(agents: List[AgentPreference]) -> float:
    """
    Compute Preference Diversity (PD).
    
    PD = (number of distinct primary preferences) / (number of agents)
    
    High PD = population has diverse specialists.
    Low PD = all agents prefer same rule.
    """
    if not agents:
        return 0.0
    
    primary_prefs = [agent.get_primary_preference() for agent in agents]
    primary_prefs = [p for p in primary_prefs if p is not None]
    
    if not primary_prefs:
        return 0.0
    
    unique_prefs = len(set(primary_prefs))
    return unique_prefs / len(agents)


def compute_preference_performance_correlation(
    agents: List[AgentPreference],
    performance_scores: Dict[str, float]
) -> float:
    """
    Compute Preference-Performance Correlation (PPC).
    
    Tests: Do stronger preferences lead to better performance?
    
    Args:
        agents: List of agents with preference data
        performance_scores: Dict mapping agent_id to overall performance score
    
    Returns:
        Correlation between preference strength and performance
    """
    if not agents or not performance_scores:
        return 0.0
    
    strengths = []
    performances = []
    
    for agent in agents:
        if agent.agent_id in performance_scores:
            strengths.append(agent.get_preference_strength())
            performances.append(performance_scores[agent.agent_id])
    
    if len(strengths) < 3:
        return 0.0
    
    correlation = np.corrcoef(strengths, performances)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0


@dataclass
class PreferenceMetricsSummary:
    """Summary of all preference metrics for a population."""
    generation: int
    rpi_variance: float
    population_psi: float
    preference_diversity: float
    ppc: float
    
    # Breakdown by agent
    agent_primary_preferences: Dict[str, str]
    agent_preference_strengths: Dict[str, float]
    
    @property
    def passes_emergence_criteria(self) -> bool:
        """Check if preferences are emerging."""
        return self.rpi_variance > 0.15
    
    @property
    def passes_stability_criteria(self) -> bool:
        """Check if preferences are stable."""
        return self.population_psi > 0.7
    
    @property
    def passes_diversity_criteria(self) -> bool:
        """Check if population is diverse."""
        return self.preference_diversity > 0.5
    
    @property
    def overall_success(self) -> bool:
        """Check if all criteria pass."""
        return (
            self.passes_emergence_criteria and
            self.passes_stability_criteria and
            self.passes_diversity_criteria
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "generation": self.generation,
            "rpi_variance": self.rpi_variance,
            "population_psi": self.population_psi,
            "preference_diversity": self.preference_diversity,
            "ppc": self.ppc,
            "passes_emergence": self.passes_emergence_criteria,
            "passes_stability": self.passes_stability_criteria,
            "passes_diversity": self.passes_diversity_criteria,
            "overall_success": self.overall_success,
            "agent_preferences": self.agent_primary_preferences,
            "agent_strengths": self.agent_preference_strengths,
        }


def compute_all_metrics(
    agents: List[AgentPreference],
    generation: int,
    performance_scores: Dict[str, float] = None
) -> PreferenceMetricsSummary:
    """Compute all preference metrics for the current population state."""
    
    # Primary preferences
    agent_prefs = {}
    agent_strengths = {}
    for agent in agents:
        pref = agent.get_primary_preference()
        agent_prefs[agent.agent_id] = pref.value if pref else "none"
        agent_strengths[agent.agent_id] = agent.get_preference_strength()
    
    return PreferenceMetricsSummary(
        generation=generation,
        rpi_variance=compute_rpi_variance(agents),
        population_psi=compute_population_psi(agents),
        preference_diversity=compute_preference_diversity(agents),
        ppc=compute_preference_performance_correlation(agents, performance_scores or {}),
        agent_primary_preferences=agent_prefs,
        agent_preference_strengths=agent_strengths,
    )


def print_metrics_summary(summary: PreferenceMetricsSummary):
    """Pretty print metrics summary."""
    print(f"\n{'='*60}")
    print(f"PREFERENCE METRICS (Generation {summary.generation})")
    print(f"{'='*60}")
    
    status = lambda x: "✅" if x else "❌"
    
    print(f"RPI Variance:         {summary.rpi_variance:.4f} {status(summary.passes_emergence_criteria)} (threshold: 0.15)")
    print(f"Population PSI:       {summary.population_psi:.4f} {status(summary.passes_stability_criteria)} (threshold: 0.70)")
    print(f"Preference Diversity: {summary.preference_diversity:.4f} {status(summary.passes_diversity_criteria)} (threshold: 0.50)")
    print(f"Pref-Perf Correlation:{summary.ppc:.4f}")
    
    print(f"\nAgent Preferences:")
    for agent_id, pref in summary.agent_primary_preferences.items():
        strength = summary.agent_preference_strengths[agent_id]
        print(f"  {agent_id}: {pref} (strength: {strength:.3f})")
    
    print(f"\nOverall: {'✅ SUCCESS' if summary.overall_success else '❌ CRITERIA NOT MET'}")

