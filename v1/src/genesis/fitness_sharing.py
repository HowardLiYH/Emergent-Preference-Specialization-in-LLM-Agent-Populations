"""
Fitness Sharing Diversity Mechanism

Prevents agents from converging to the same specialty by penalizing crowded niches.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import Counter

from .domains import DomainType
from .evolution_v2 import SpecialistPrompt


@dataclass
class FitnessSharingConfig:
    """Configuration for fitness sharing."""
    sharing_radius: float = 1.0
    alpha: float = 1.0
    min_fitness: float = 0.1
    crowding_threshold: int = 2


@dataclass
class NicheStats:
    """Statistics about a niche."""
    domain: DomainType
    population: int
    total_fitness: float
    avg_fitness: float
    crowding_penalty: float


def calculate_niche_count(agents: List[SpecialistPrompt]) -> Dict[DomainType, int]:
    """Count agents in each domain."""
    counts = Counter()
    for agent in agents:
        if agent.primary_domain:
            counts[agent.primary_domain] += 1
    return dict(counts)


def sharing_function(distance: float, sigma: float = 1.0, alpha: float = 1.0) -> float:
    """Standard sharing function."""
    if distance >= sigma:
        return 0.0
    return 1.0 - (distance / sigma) ** alpha


def calculate_niche_distance(agent1: SpecialistPrompt, agent2: SpecialistPrompt) -> float:
    """Distance between agents based on specialization."""
    if agent1.primary_domain is None and agent2.primary_domain is None:
        return 0.2
    if agent1.primary_domain is None or agent2.primary_domain is None:
        return 0.5
    if agent1.primary_domain == agent2.primary_domain:
        return 0.0
    return 1.0


def apply_fitness_sharing(
    agents: List[SpecialistPrompt],
    raw_fitness: Dict[str, float],
    config: FitnessSharingConfig = None
) -> Dict[str, float]:
    """Apply fitness sharing to adjust fitness based on niche population."""
    if config is None:
        config = FitnessSharingConfig()

    adjusted = {}
    for agent in agents:
        agent_fitness = raw_fitness.get(agent.agent_id, 0.0)
        niche_count = 1.0

        for other in agents:
            if other.agent_id == agent.agent_id:
                continue
            distance = calculate_niche_distance(agent, other)
            sh = sharing_function(distance, config.sharing_radius, config.alpha)
            niche_count += sh

        shared_fitness = agent_fitness / niche_count
        adjusted[agent.agent_id] = max(shared_fitness, config.min_fitness * agent_fitness)

    return adjusted


def calculate_crowding_penalty(
    agent: SpecialistPrompt,
    agents: List[SpecialistPrompt],
    config: FitnessSharingConfig = None
) -> float:
    """Calculate crowding penalty for an agent."""
    if config is None:
        config = FitnessSharingConfig()

    if agent.primary_domain is None:
        return 0.7

    same_niche = sum(1 for other in agents if other.primary_domain == agent.primary_domain)

    if same_niche <= config.crowding_threshold:
        return 1.0

    excess = same_niche - config.crowding_threshold
    penalty = 1.0 / (1.0 + 0.3 * excess)
    return max(penalty, 0.2)


def get_underrepresented_niches(agents: List[SpecialistPrompt], threshold: int = 1) -> List[DomainType]:
    """Find underrepresented niches."""
    counts = calculate_niche_count(agents)
    return [d for d in DomainType if counts.get(d, 0) < threshold]


def get_niche_statistics(agents: List[SpecialistPrompt]) -> List[NicheStats]:
    """Get statistics for each niche."""
    stats = []
    config = FitnessSharingConfig()

    for domain in DomainType:
        niche_agents = [a for a in agents if a.primary_domain == domain]
        population = len(niche_agents)

        if population == 0:
            stats.append(NicheStats(domain, 0, 0.0, 0.0, 1.0))
        else:
            total_fitness = sum(a.get_overall_win_rate() for a in niche_agents)
            avg_fitness = total_fitness / population

            if population <= config.crowding_threshold:
                penalty = 1.0
            else:
                excess = population - config.crowding_threshold
                penalty = 1.0 / (1.0 + 0.3 * excess)

            stats.append(NicheStats(domain, population, total_fitness, avg_fitness, penalty))

    return stats


def calculate_diversity_bonus(
    agent: SpecialistPrompt,
    agents: List[SpecialistPrompt],
    bonus_scale: float = 0.2
) -> float:
    """Calculate diversity bonus for rare niches."""
    if agent.primary_domain is None:
        return 0.0

    same_niche = sum(1 for other in agents if other.primary_domain == agent.primary_domain)

    if same_niche == 1:
        return bonus_scale
    elif same_niche == 2:
        return bonus_scale * 0.5
    return 0.0


def print_niche_report(agents: List[SpecialistPrompt]):
    """Print niche distribution report."""
    stats = get_niche_statistics(agents)
    total = len(agents)

    print("\n" + "=" * 60)
    print("NICHE DISTRIBUTION REPORT")
    print("=" * 60)

    for s in sorted(stats, key=lambda x: x.population, reverse=True):
        pct = s.population / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"{s.domain.value:15} | {bar} | {s.population:2} ({pct:5.1f}%)")

    print("=" * 60)
