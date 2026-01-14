"""
Fitness Sharing Sensitivity Ablation

Tests different penalty functions for fitness sharing:
- No penalty (1.0)
- Linear penalty (1/n)
- Sqrt penalty (1/sqrt(n)) - current default
- Log penalty (1/log(n+1))
- Quadratic penalty (1/n^2)

Measures: Diversity, SCI, L3 coverage under each penalty function.

Ground Rules Applied:
- Checkpoints for long experiments
- API keys from .env only
- Unified model across tests
"""

import os
import sys
import json
import random
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Callable, Tuple
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genesis.synthetic_rules import RuleType
from src.genesis.neurips_metrics import compute_sci, compute_gini, compute_hhi
from src.genesis.analysis import compute_mean_std, compute_confidence_interval


# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = Path("results/fitness_sensitivity")
RESULTS_FILE = RESULTS_DIR / "fitness_sensitivity_results.json"
LOG_FILE = RESULTS_DIR / "fitness_sensitivity_log.txt"

# Experiment parameters
N_AGENTS = 12
N_GENERATIONS = 100
N_SEEDS = 5
RULES = list(RuleType)


# =============================================================================
# PENALTY FUNCTIONS
# =============================================================================

def no_penalty(n: int) -> float:
    """No fitness sharing - everyone gets full reward."""
    return 1.0


def linear_penalty(n: int) -> float:
    """Linear penalty: 1/n."""
    return 1.0 / n if n > 0 else 1.0


def sqrt_penalty(n: int) -> float:
    """Square root penalty: 1/sqrt(n). Current default."""
    return 1.0 / math.sqrt(n) if n > 0 else 1.0


def log_penalty(n: int) -> float:
    """Logarithmic penalty: 1/log(n+1)."""
    return 1.0 / math.log(n + 1) if n > 0 else 1.0


def quadratic_penalty(n: int) -> float:
    """Quadratic penalty: 1/n^2."""
    return 1.0 / (n * n) if n > 0 else 1.0


PENALTY_FUNCTIONS = {
    "none": no_penalty,
    "linear": linear_penalty,
    "sqrt": sqrt_penalty,
    "log": log_penalty,
    "quadratic": quadratic_penalty,
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PenaltyResult:
    """Result for one penalty function."""
    penalty_name: str
    n_seeds: int

    # Metrics (mean ± std across seeds)
    mean_sci: float
    std_sci: float
    mean_diversity: float
    std_diversity: float
    mean_l3_coverage: float
    std_l3_coverage: float
    mean_hhi: float
    std_hhi: float

    # Per-seed data
    per_seed_sci: List[float] = field(default_factory=list)
    per_seed_diversity: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def log(message: str):
    """Log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(log_line + "\n")


# =============================================================================
# SIMULATION
# =============================================================================

def simulate_evolution(
    penalty_fn: Callable[[int], float],
    n_agents: int = N_AGENTS,
    n_generations: int = N_GENERATIONS,
    seed: int = 42
) -> Dict:
    """
    Simulate evolution with given penalty function.

    Returns final metrics after n_generations.
    """
    rng = random.Random(seed)
    n_rules = len(RULES)

    # Initialize agents with Option B+
    agents = []
    for i in range(n_agents):
        initial_rule = rng.choice(RULES)
        levels = {r: 1 if r == initial_rule else 0 for r in RULES}
        agents.append({
            "id": f"agent_{i}",
            "levels": levels,
            "initial": initial_rule
        })

    # Run evolution
    for gen in range(n_generations):
        # Each agent competes
        for agent in agents:
            # Determine focus rule (highest level)
            max_level = max(agent["levels"].values())
            max_rules = [r for r, l in agent["levels"].items() if l == max_level]
            focus_rule = rng.choice(max_rules)

            # Count agents focusing on same rule
            n_competitors = sum(
                1 for a in agents
                if max(a["levels"].values()) > 0 and
                focus_rule in [r for r, l in a["levels"].items() if l == max(a["levels"].values())]
            )

            # Apply fitness sharing penalty
            penalty = penalty_fn(max(1, n_competitors))

            # Win probability based on level and penalty
            base_win_prob = 0.3 + 0.15 * agent["levels"][focus_rule]
            adjusted_win_prob = base_win_prob * penalty

            if rng.random() < adjusted_win_prob:
                # Level up (with exclusivity)
                current_level = agent["levels"][focus_rule]
                if current_level < 3:
                    agent["levels"][focus_rule] = current_level + 1

    # Compute final metrics
    final_scis = [compute_sci(a["levels"]) for a in agents]
    mean_sci = sum(final_scis) / len(final_scis)

    # Count L3 specialists and coverage
    l3_agents = [a for a in agents if max(a["levels"].values()) >= 3]

    covered_rules = set()
    for agent in agents:
        max_level = max(agent["levels"].values())
        if max_level >= 3:
            for r, l in agent["levels"].items():
                if l == max_level:
                    covered_rules.add(r)

    diversity = len(covered_rules) / n_rules
    l3_coverage = len(l3_agents) / n_agents

    # HHI (concentration)
    rule_counts = {}
    for agent in agents:
        max_level = max(agent["levels"].values())
        max_rules = [r for r, l in agent["levels"].items() if l == max_level]
        for r in max_rules:
            rule_counts[r.value] = rule_counts.get(r.value, 0) + 1

    total = sum(rule_counts.values())
    hhi = sum((c / total) ** 2 for c in rule_counts.values()) if total > 0 else 0

    return {
        "sci": mean_sci,
        "diversity": diversity,
        "l3_coverage": l3_coverage,
        "hhi": hhi,
        "covered_rules": len(covered_rules),
        "l3_agents": len(l3_agents)
    }


def run_penalty_experiment(
    penalty_name: str,
    penalty_fn: Callable[[int], float],
    n_seeds: int = N_SEEDS
) -> PenaltyResult:
    """Run experiment for one penalty function across multiple seeds."""

    scis = []
    diversities = []
    l3_coverages = []
    hhis = []

    for seed in range(n_seeds):
        result = simulate_evolution(penalty_fn, seed=seed)
        scis.append(result["sci"])
        diversities.append(result["diversity"])
        l3_coverages.append(result["l3_coverage"])
        hhis.append(result["hhi"])

    return PenaltyResult(
        penalty_name=penalty_name,
        n_seeds=n_seeds,
        mean_sci=sum(scis) / len(scis),
        std_sci=compute_mean_std(scis)[1],
        mean_diversity=sum(diversities) / len(diversities),
        std_diversity=compute_mean_std(diversities)[1],
        mean_l3_coverage=sum(l3_coverages) / len(l3_coverages),
        std_l3_coverage=compute_mean_std(l3_coverages)[1],
        mean_hhi=sum(hhis) / len(hhis),
        std_hhi=compute_mean_std(hhis)[1],
        per_seed_sci=scis,
        per_seed_diversity=diversities
    )


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_fitness_sensitivity_experiment() -> Dict:
    """Run full fitness sharing sensitivity analysis."""

    log("=" * 60)
    log("FITNESS SHARING SENSITIVITY ABLATION")
    log("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    for penalty_name, penalty_fn in PENALTY_FUNCTIONS.items():
        log(f"\n[{penalty_name.upper()}] Running {N_SEEDS} seeds...")

        result = run_penalty_experiment(penalty_name, penalty_fn, N_SEEDS)
        results[penalty_name] = result

        log(f"  SCI: {result.mean_sci:.3f} ± {result.std_sci:.3f}")
        log(f"  Diversity: {result.mean_diversity:.1%} ± {result.std_diversity:.1%}")
        log(f"  L3 Coverage: {result.mean_l3_coverage:.1%}")
        log(f"  HHI: {result.mean_hhi:.3f}")

    # Summary
    log("\n" + "=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)

    log("\n| Penalty | SCI | Diversity | L3 Rate | HHI |")
    log("|---------|-----|-----------|---------|-----|")
    for name, r in results.items():
        log(f"| {name:9s} | {r.mean_sci:.3f} | {r.mean_diversity:.1%} | {r.mean_l3_coverage:.1%} | {r.mean_hhi:.3f} |")

    # Find best penalty
    best_diversity = max(results.items(), key=lambda x: x[1].mean_diversity)
    best_sci = max(results.items(), key=lambda x: x[1].mean_sci)

    log(f"\nBest for Diversity: {best_diversity[0]} ({best_diversity[1].mean_diversity:.1%})")
    log(f"Best for SCI: {best_sci[0]} ({best_sci[1].mean_sci:.3f})")

    # Recommendation
    if best_diversity[0] == "sqrt":
        recommendation = "CONFIRMED: sqrt(n) penalty is optimal for diversity."
    else:
        recommendation = f"CONSIDER: {best_diversity[0]} penalty may be better than sqrt."

    log(f"\nRecommendation: {recommendation}")

    # Save results
    final_results = {
        "experiment": "fitness_sensitivity",
        "timestamp": datetime.now().isoformat(),
        "n_agents": N_AGENTS,
        "n_generations": N_GENERATIONS,
        "n_seeds": N_SEEDS,
        "results": {k: v.to_dict() for k, v in results.items()},
        "best_diversity": best_diversity[0],
        "best_sci": best_sci[0],
        "recommendation": recommendation
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(final_results, f, indent=2)

    log(f"\nResults saved to {RESULTS_FILE}")

    return final_results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_fitness_sensitivity_experiment()
