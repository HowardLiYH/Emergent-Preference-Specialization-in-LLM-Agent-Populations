"""
Fitness Sharing for Preference Diversity

Penalizes agents that specialize in crowded rules to promote diversity.
Implements the 1/sqrt(n) penalty from evolutionary computation literature.
"""

import math
import random
from typing import List, Dict, Optional

from .synthetic_rules import RuleType
from .preference_agent import PreferenceAgent


def compute_fitness_sharing_penalty(
    agents: List[PreferenceAgent],
    agent: PreferenceAgent
) -> float:
    """
    Reduce reward for winning in a crowded niche.

    If 5 agents already specialize in POSITION, winning POSITION
    gives less reward than winning RHYME (where only 1 agent specializes).

    Args:
        agents: All agents in the population
        agent: The agent to compute penalty for

    Returns:
        Sharing factor between 0 and 1 (1 = no penalty, lower = more penalty)
    """
    primary_pref = agent.get_primary_preference()

    if primary_pref is None:
        return 1.0  # No penalty for generalists

    # Count how many agents share this preference
    same_pref_count = sum(
        1 for a in agents
        if a.get_primary_preference() == primary_pref
    )

    # Penalty: 1/sqrt(n) where n = number of agents with same preference
    # n=1 → 1.0 (no penalty)
    # n=4 → 0.5 (50% penalty)
    # n=9 → 0.33 (67% penalty)
    sharing_factor = 1.0 / math.sqrt(same_pref_count)

    return sharing_factor


def compute_niche_counts(agents: List[PreferenceAgent]) -> Dict[RuleType, int]:
    """
    Count how many agents specialize in each rule.

    Returns:
        Dict mapping RuleType to count of specialists
    """
    counts = {r: 0 for r in RuleType}

    for agent in agents:
        pref = agent.get_primary_preference()
        if pref is not None:
            counts[pref] += 1

    return counts


def compute_niche_penalty_for_rule(
    agents: List[PreferenceAgent],
    rule_type: RuleType
) -> float:
    """
    Compute penalty for winning in a specific rule's niche.

    Args:
        agents: All agents
        rule_type: The rule being competed on

    Returns:
        Penalty factor (0-1)
    """
    # Count agents whose primary preference is this rule
    specialist_count = sum(
        1 for a in agents
        if a.get_primary_preference() == rule_type
    )

    if specialist_count == 0:
        return 1.0  # No specialists = no penalty (encourage exploration)

    return 1.0 / math.sqrt(specialist_count)


def apply_fitness_sharing(
    agents: List[PreferenceAgent],
    winner: PreferenceAgent,
    rule_type: RuleType,
    deterministic: bool = False
) -> bool:
    """
    Apply fitness sharing when accumulating strategies.

    The strategy level only increases probabilistically based on niche crowding.

    Args:
        agents: All agents in population
        winner: The agent that won
        rule_type: The rule type of the task
        deterministic: If True, always accumulate (for testing)

    Returns:
        True if strategy was accumulated, False otherwise
    """
    if deterministic:
        return winner.accumulate_strategy(rule_type)

    penalty = compute_fitness_sharing_penalty(agents, winner)

    # Probabilistic accumulation based on penalty
    if random.random() < penalty:
        return winner.accumulate_strategy(rule_type)
    else:
        # Win doesn't count toward strategy accumulation
        return False


def get_diversity_bonus_rule(
    agents: List[PreferenceAgent],
    exclude_rules: List[RuleType] = None
) -> Optional[RuleType]:
    """
    Get the rule with fewest specialists (for diversity bonus).

    Useful for encouraging exploration of underrepresented niches.

    Args:
        agents: All agents
        exclude_rules: Rules to exclude from consideration

    Returns:
        Rule with fewest specialists, or None if all excluded
    """
    exclude_rules = exclude_rules or []
    counts = compute_niche_counts(agents)

    # Filter out excluded rules
    available = {r: c for r, c in counts.items() if r not in exclude_rules}

    if not available:
        return None

    # Return rule with minimum specialists
    return min(available.keys(), key=lambda r: available[r])


def compute_population_diversity_score(agents: List[PreferenceAgent]) -> float:
    """
    Compute overall diversity score for the population.

    Higher score = more evenly distributed specialists across rules.

    Returns:
        Score between 0 (all same niche) and 1 (perfectly distributed)
    """
    counts = compute_niche_counts(agents)
    total_specialists = sum(counts.values())

    if total_specialists == 0:
        return 0.0

    # Number of non-empty niches
    non_empty = sum(1 for c in counts.values() if c > 0)

    # Max possible is all 8 rules represented
    diversity_score = non_empty / len(RuleType)

    # Also penalize very uneven distributions
    if non_empty > 0:
        evenness = 1.0 - (max(counts.values()) - min(c for c in counts.values() if c > 0)) / total_specialists
        diversity_score *= evenness

    return diversity_score
