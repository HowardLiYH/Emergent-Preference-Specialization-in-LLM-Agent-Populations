"""
Preference Agent for Synthetic Rule Experiments

Agent with:
- Strategy level accumulation (0-3 per rule)
- Preference tracking (wins/attempts per rule)
- Preference history for stability analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from .synthetic_rules import RuleType
from .rule_strategies import STRATEGY_LEVELS, build_prompt_from_levels


@dataclass
class PreferenceAgent:
    """
    An agent that develops preferences through competition on synthetic rules.

    Key features:
    - Accumulates strategy levels (0-3) for each rule through wins
    - Tracks win/attempt ratios (RPI) for preference measurement
    - Snapshots preferences for stability analysis (PSI)
    """
    agent_id: str

    # Strategy levels: 0 = no knowledge, 1-3 = increasing expertise
    strategy_levels: Dict[RuleType, int] = field(default_factory=dict)

    # Win/attempt tracking
    wins_by_rule: Dict[RuleType, int] = field(default_factory=dict)
    attempts_by_rule: Dict[RuleType, int] = field(default_factory=dict)

    # Historical snapshots for stability analysis
    preference_history: List[Dict[str, float]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize all rules to level 0."""
        if not self.strategy_levels:
            self.strategy_levels = {r: 0 for r in RuleType}
        if not self.wins_by_rule:
            self.wins_by_rule = {r: 0 for r in RuleType}
        if not self.attempts_by_rule:
            self.attempts_by_rule = {r: 0 for r in RuleType}

    # ==================== Strategy Accumulation ====================

    def has_level_3_specialist(self) -> bool:
        """Check if this agent has reached Level 3 in any rule."""
        return any(level >= 3 for level in self.strategy_levels.values())

    def get_specialized_rule(self) -> Optional[RuleType]:
        """Get the rule where agent has Level 3 (if any)."""
        for rule, level in self.strategy_levels.items():
            if level >= 3:
                return rule
        return None

    def accumulate_strategy(self, rule_type: RuleType, exclusive: bool = True) -> bool:
        """
        Win on a rule → increase strategy level (max 3).

        Args:
            rule_type: The rule to accumulate strategy for
            exclusive: If True, once Level 3 in any rule, can only accumulate in that rule

        Returns:
            True if level increased, False if blocked or already at max
        """
        # EXCLUSIVITY: Once Level 3, can only accumulate in that rule
        if exclusive and self.has_level_3_specialist():
            specialized_rule = self.get_specialized_rule()
            if rule_type != specialized_rule:
                return False  # Blocked - already specialized in different rule

        current = self.strategy_levels.get(rule_type, 0)
        if current < 3:
            self.strategy_levels[rule_type] = current + 1
            return True
        return False

    def get_strategy_level(self, rule_type: RuleType) -> int:
        """Get current strategy level for a rule."""
        return self.strategy_levels.get(rule_type, 0)

    def get_total_strategy_levels(self) -> int:
        """Sum of all strategy levels (measure of total learning)."""
        return sum(self.strategy_levels.values())

    # ==================== Prompt Building ====================

    def get_prompt(self) -> str:
        """Build system prompt from accumulated strategies."""
        return build_prompt_from_levels(self.strategy_levels)

    # ==================== Preference Tracking ====================

    def record_attempt(self, rule_type: RuleType, won: bool):
        """Record an attempt on a rule."""
        self.attempts_by_rule[rule_type] = self.attempts_by_rule.get(rule_type, 0) + 1
        if won:
            self.wins_by_rule[rule_type] = self.wins_by_rule.get(rule_type, 0) + 1

    def get_rpi(self, rule_type: RuleType) -> float:
        """
        Get Rule Preference Index for a specific rule.
        RPI = wins / attempts
        """
        attempts = self.attempts_by_rule.get(rule_type, 0)
        wins = self.wins_by_rule.get(rule_type, 0)
        return wins / attempts if attempts > 0 else 0.0

    def get_all_rpi(self) -> Dict[RuleType, float]:
        """Get RPI for all rules."""
        return {r: self.get_rpi(r) for r in RuleType}

    def get_primary_preference(self) -> Optional[RuleType]:
        """Get the rule this agent prefers most (highest RPI)."""
        rpis = self.get_all_rpi()
        if not rpis or max(rpis.values()) == 0:
            return None
        return max(rpis.keys(), key=lambda r: rpis[r])

    def get_preference_strength(self) -> float:
        """
        How specialized is this agent?
        Strength = max(RPI) - mean(RPI)
        Higher = more specialized toward one rule
        """
        rpis = list(self.get_all_rpi().values())
        if not rpis or max(rpis) == 0:
            return 0.0
        return max(rpis) - np.mean(rpis)

    # ==================== Stability Analysis ====================

    def snapshot_preferences(self):
        """Save current preferences to history for stability analysis."""
        rpis = {r.value: self.get_rpi(r) for r in RuleType}
        self.preference_history.append(rpis)

    def get_preference_stability(self, window: int = 10) -> float:
        """
        Calculate Preference Stability Index (PSI).

        Measures correlation between preferences at different time points.
        High PSI = stable preferences (not random drift).
        """
        if len(self.preference_history) < window:
            return 0.0

        current = list(self.preference_history[-1].values())
        past = list(self.preference_history[-window].values())

        if sum(current) == 0 or sum(past) == 0:
            return 0.0

        # Pearson correlation
        correlation = np.corrcoef(current, past)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0

    # ==================== Utility ====================

    def reset(self):
        """Reset agent to initial state."""
        self.strategy_levels = {r: 0 for r in RuleType}
        self.wins_by_rule = {r: 0 for r in RuleType}
        self.attempts_by_rule = {r: 0 for r in RuleType}
        self.preference_history = []

    def copy(self) -> 'PreferenceAgent':
        """Create a copy of this agent."""
        new_agent = PreferenceAgent(agent_id=f"{self.agent_id}_copy")
        new_agent.strategy_levels = self.strategy_levels.copy()
        new_agent.wins_by_rule = self.wins_by_rule.copy()
        new_agent.attempts_by_rule = self.attempts_by_rule.copy()
        new_agent.preference_history = [h.copy() for h in self.preference_history]
        return new_agent

    def __repr__(self) -> str:
        primary = self.get_primary_preference()
        strength = self.get_preference_strength()
        total_levels = self.get_total_strategy_levels()
        return f"PreferenceAgent({self.agent_id}, primary={primary.value if primary else 'none'}, strength={strength:.3f}, levels={total_levels})"


def create_preference_agent(agent_id: str) -> PreferenceAgent:
    """Factory function to create a new preference agent."""
    return PreferenceAgent(agent_id=agent_id)


def create_handcrafted_specialist(rule_type: RuleType) -> PreferenceAgent:
    """
    Create a perfect specialist with Level 3 strategy for one rule.
    Used for ceiling baseline.
    """
    agent = PreferenceAgent(agent_id=f"handcrafted_{rule_type.value}")
    agent.strategy_levels[rule_type] = 3  # Maximum level
    return agent


def create_population(num_agents: int, prefix: str = "agent") -> List[PreferenceAgent]:
    """Create a population of identical starting agents."""
    return [PreferenceAgent(agent_id=f"{prefix}_{i}") for i in range(num_agents)]
