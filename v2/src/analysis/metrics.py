"""
Architectural Metrics for CSE Evaluation.

Implements the key metrics defined by professors:
1. Coverage Efficiency (Coverage / Tokens)
2. Scaling Exponent (α where Cost ∝ N^α)
3. Strategy Diversity (distinct specialist behaviors)
4. Equilibrium Quality (distance from optimal)
"""

import numpy as np
from typing import Dict, List, Any
from collections import Counter, defaultdict


def coverage_efficiency(coverage: float, total_tokens: int) -> float:
    """
    Coverage Efficiency = Coverage / Tokens × 1000

    Higher = more efficient architecture.
    Measures how much coverage we get per token spent.

    Args:
        coverage: Fraction of regimes covered (0-1)
        total_tokens: Total tokens used

    Returns:
        Coverage efficiency score (higher is better)
    """
    if total_tokens == 0:
        return 0.0
    return coverage / total_tokens * 1000


def compute_scaling_exponent(results_by_n: Dict[int, Dict]) -> float:
    """
    Compute scaling exponent α where Cost(N) ∝ N^α

    Lower α = better scaling (sublinear is < 1).

    Args:
        results_by_n: Dict mapping N to {'tokens': int, 'coverage': float}

    Returns:
        Scaling exponent α
    """
    ns = sorted(results_by_n.keys())
    tokens = [results_by_n[n]['tokens'] for n in ns]

    if len(ns) < 2 or min(tokens) <= 0:
        return 1.0  # Default to linear

    # Fit log-log linear regression
    log_n = np.log(ns)
    log_tokens = np.log(tokens)

    alpha, _ = np.polyfit(log_n, log_tokens, 1)
    return float(alpha)


def strategy_diversity(agents: List[Any]) -> float:
    """
    Strategy Diversity: Count distinct specialist behaviors.

    Uses specialization patterns to identify unique strategies.
    Higher = more diverse population.

    Args:
        agents: List of agents with get_specialty() method

    Returns:
        Number of distinct strategies (normalized by population size)
    """
    specialties = [a.get_specialty() for a in agents if a.get_specialty()]

    if not specialties:
        return 0.0

    unique_specialties = len(set(specialties))
    return unique_specialties / len(agents)


def equilibrium_quality(agents: List[Any], regimes: Dict, gamma: float = 0.5) -> float:
    """
    Equilibrium Quality: Distance from theoretical optimal distribution.

    Optimal: n_r ∝ (f_r × R_r × D_r)^(2/3)
    Lower distance = better equilibrium.

    Args:
        agents: List of agents
        regimes: Dict of Regime objects with frequency, reward, difficulty
        gamma: Fitness sharing parameter

    Returns:
        Mean absolute error from optimal distribution (lower is better)
    """
    # Count actual specialists per regime
    specialty_counts = Counter([a.get_specialty() for a in agents if a.get_specialty()])

    if not specialty_counts:
        return 1.0  # Worst case

    # Compute theoretical optimal distribution
    n_total = sum(specialty_counts.values())
    optimal = {}

    for name, regime in regimes.items():
        # n_r ∝ (f × R × D)^(1/(1+γ))
        exponent = 1 / (1 + gamma)
        f = getattr(regime, 'frequency', 0.2)
        R = getattr(regime, 'reward', 1.0)
        D = getattr(regime, 'difficulty', 0.5)

        optimal[name] = (f * R * D) ** exponent

    # Normalize optimal to sum to n_total
    total_optimal = sum(optimal.values())
    if total_optimal > 0:
        optimal = {k: v / total_optimal * n_total for k, v in optimal.items()}

    # Compute mean absolute error
    errors = []
    for name in regimes:
        actual = specialty_counts.get(name, 0)
        predicted = optimal.get(name, 0)
        errors.append(abs(actual - predicted))

    mae = sum(errors) / len(errors) if errors else 0
    normalized_mae = mae / n_total  # Normalize by population size

    return normalized_mae


def compute_sci(agents: List[Any], n_regimes: int) -> float:
    """
    Specialization Concentration Index (SCI).

    Based on Herfindahl-Hirschman Index.
    Lower = more diverse, Higher = more concentrated.

    Args:
        agents: List of agents
        n_regimes: Number of regimes

    Returns:
        Normalized SCI (0 = perfect diversity, 1 = single specialist)
    """
    specialty_counts = Counter([a.get_specialty() for a in agents if a.get_specialty()])

    if not specialty_counts:
        return 0.0

    total = sum(specialty_counts.values())
    hhi = sum((c / total) ** 2 for c in specialty_counts.values())

    # Normalize: (HHI - 1/n) / (1 - 1/n)
    min_hhi = 1 / n_regimes
    if n_regimes > 1:
        normalized = (hhi - min_hhi) / (1 - min_hhi)
    else:
        normalized = hhi

    return max(0, min(1, normalized))


def compute_all_metrics(agents: List[Any], regimes: Dict,
                        total_tokens: int, results_by_n: Dict = None) -> Dict:
    """
    Compute all architectural metrics.

    Returns:
        Dict with all metric values
    """
    n_regimes = len(regimes)
    coverage = strategy_diversity(agents) * n_regimes / n_regimes  # Same as compute_coverage

    # Recompute coverage properly
    covered = {a.get_specialty() for a in agents if a.get_specialty()}
    coverage = len(covered) / n_regimes if n_regimes > 0 else 0

    metrics = {
        'coverage': coverage,
        'coverage_efficiency': coverage_efficiency(coverage, total_tokens),
        'sci': compute_sci(agents, n_regimes),
        'strategy_diversity': strategy_diversity(agents),
        'equilibrium_quality': equilibrium_quality(agents, regimes),
        'total_tokens': total_tokens,
    }

    if results_by_n:
        metrics['scaling_exponent'] = compute_scaling_exponent(results_by_n)

    return metrics
