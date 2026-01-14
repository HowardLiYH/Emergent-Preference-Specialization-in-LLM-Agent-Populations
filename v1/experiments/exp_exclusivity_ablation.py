"""
Exclusivity & Fitness Sharing Ablation Study

Tests whether specialization is truly emergent or forced by design:
1. Full: exclusivity + fitness_sharing (our system)
2. No Exclusivity: fitness_sharing only
3. No Fitness Sharing: exclusivity only
4. Neither: competition only (pure emergence test)

Key question: Does competition alone produce specialization?
"""

import asyncio
import random
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genesis.synthetic_rules import RuleType
from genesis.preference_agent import PreferenceAgent, create_population
from genesis.neurips_metrics import compute_sci


@dataclass
class AblationCondition:
    """A single ablation condition."""
    name: str
    use_exclusivity: bool
    use_fitness_sharing: bool
    description: str


# Define ablation conditions
ABLATION_CONDITIONS = [
    AblationCondition(
        name="full_system",
        use_exclusivity=True,
        use_fitness_sharing=True,
        description="Complete system with exclusivity and fitness sharing"
    ),
    AblationCondition(
        name="no_exclusivity",
        use_exclusivity=False,
        use_fitness_sharing=True,
        description="Fitness sharing but no exclusivity (can be multi-specialist)"
    ),
    AblationCondition(
        name="no_fitness_sharing",
        use_exclusivity=True,
        use_fitness_sharing=False,
        description="Exclusivity but no fitness sharing (crowding allowed)"
    ),
    AblationCondition(
        name="competition_only",
        use_exclusivity=False,
        use_fitness_sharing=False,
        description="Pure competition - no constraints (key test for emergence)"
    ),
]


@dataclass
class AblationResult:
    """Result for one condition."""
    condition_name: str
    use_exclusivity: bool
    use_fitness_sharing: bool
    mean_sci: float
    std_sci: float
    coverage: float  # fraction of rules with at least one L3 specialist
    num_l3_specialists: float  # average number of L3 agents
    super_agent_count: float  # agents with L3 in multiple rules
    agents_with_l3: float  # agents with L3 in at least one rule
    per_run_sci: List[float]


@dataclass
class AblationSummary:
    """Summary of all ablation conditions."""
    results: List[Dict]
    best_condition: str
    competition_only_sci: float
    exclusivity_contribution: float
    fitness_sharing_contribution: float
    timestamp: str
    n_agents: int
    n_generations: int
    n_runs: int


def simulate_competition_round(
    agents: List[PreferenceAgent],
    rule_type: RuleType,
    use_exclusivity: bool,
    use_fitness_sharing: bool
) -> Optional[str]:
    """
    Simulate a single competition round.

    In simulation: agent with highest strategy level + noise wins.
    This mimics the LLM confidence correlation with expertise.
    """
    # Each agent "competes" - higher level = higher chance to win
    competitors = []
    for agent in agents:
        level = agent.strategy_levels.get(rule_type, 0)
        # Base probability + noise (mimics LLM confidence)
        score = level + random.random() * 0.5
        # Random chance of being "correct" (higher level = higher chance)
        correct_prob = 0.3 + 0.2 * level  # L0=30%, L3=90%
        if random.random() < correct_prob:
            competitors.append((agent, score))

    if not competitors:
        return None

    # Winner = highest score among correct
    winner_agent, _ = max(competitors, key=lambda x: x[1])

    # Apply fitness sharing probabilistically
    if use_fitness_sharing:
        # Count specialists in this rule
        specialist_count = sum(
            1 for a in agents
            if a.strategy_levels.get(rule_type, 0) >= 2
        )
        penalty = 1.0 / (1 + specialist_count * 0.3)
        if random.random() > penalty:
            return winner_agent.agent_id  # Win but no accumulation

    # Accumulate strategy (with or without exclusivity)
    winner_agent.accumulate_strategy(rule_type, exclusive=use_exclusivity)

    return winner_agent.agent_id


def run_single_simulation(
    n_agents: int,
    n_generations: int,
    use_exclusivity: bool,
    use_fitness_sharing: bool
) -> Tuple[float, float, int, int, int]:
    """
    Run a single simulation and return metrics.

    Returns:
        (mean_sci, coverage, num_l3, super_agent_count, agents_with_l3)
    """
    # Create population with seeded initialization
    agents = create_population(n_agents, prefix="agent", seed_level=1)
    rules = list(RuleType)

    # Evolution loop
    for gen in range(n_generations):
        # Sample random rule
        rule = random.choice(rules)

        # Run competition
        simulate_competition_round(
            agents, rule, use_exclusivity, use_fitness_sharing
        )

    # Compute metrics
    sci_values = []
    l3_count = 0
    super_agents = 0
    agents_with_any_l3 = 0

    for agent in agents:
        sci = compute_sci(agent.strategy_levels)
        sci_values.append(sci)

        # Count L3 specialists
        l3_rules = [r for r, l in agent.strategy_levels.items() if l >= 3]
        if len(l3_rules) > 0:
            agents_with_any_l3 += 1
            l3_count += len(l3_rules)
        if len(l3_rules) > 1:
            super_agents += 1

    mean_sci = np.mean(sci_values)

    # Coverage: how many rules have at least one L3 specialist
    covered_rules = set()
    for agent in agents:
        for rule, level in agent.strategy_levels.items():
            if level >= 3:
                covered_rules.add(rule)
    coverage = len(covered_rules) / len(rules)

    return mean_sci, coverage, l3_count, super_agents, agents_with_any_l3


def run_condition(
    condition: AblationCondition,
    n_agents: int = 12,
    n_generations: int = 100,
    n_runs: int = 5
) -> AblationResult:
    """Run multiple simulations for a condition."""
    print(f"\n  Testing: {condition.name}")
    print(f"    Exclusivity: {condition.use_exclusivity}, Fitness Sharing: {condition.use_fitness_sharing}")

    sci_values = []
    coverage_values = []
    l3_counts = []
    super_agent_counts = []
    agents_with_l3_counts = []

    for run in range(n_runs):
        sci, cov, l3, super_a, with_l3 = run_single_simulation(
            n_agents, n_generations,
            condition.use_exclusivity,
            condition.use_fitness_sharing
        )
        sci_values.append(sci)
        coverage_values.append(cov)
        l3_counts.append(l3)
        super_agent_counts.append(super_a)
        agents_with_l3_counts.append(with_l3)

    result = AblationResult(
        condition_name=condition.name,
        use_exclusivity=condition.use_exclusivity,
        use_fitness_sharing=condition.use_fitness_sharing,
        mean_sci=np.mean(sci_values),
        std_sci=np.std(sci_values),
        coverage=np.mean(coverage_values),
        num_l3_specialists=np.mean(l3_counts),
        super_agent_count=np.mean(super_agent_counts),
        agents_with_l3=np.mean(agents_with_l3_counts),
        per_run_sci=sci_values
    )

    print(f"    Mean SCI: {result.mean_sci:.3f} ± {result.std_sci:.3f}")
    print(f"    Coverage: {result.coverage:.1%}")
    print(f"    L3 Specialists: {result.num_l3_specialists:.1f}")
    print(f"    Super-agents: {result.super_agent_count:.1f}")

    return result


def run_ablation_study(
    n_agents: int = 12,
    n_generations: int = 100,
    n_runs: int = 10
) -> AblationSummary:
    """Run the full ablation study."""
    print("=" * 70)
    print("EXCLUSIVITY & FITNESS SHARING ABLATION STUDY")
    print("=" * 70)
    print(f"Agents: {n_agents}, Generations: {n_generations}, Runs per condition: {n_runs}")

    results = []
    for condition in ABLATION_CONDITIONS:
        result = run_condition(condition, n_agents, n_generations, n_runs)
        results.append(result)

    # Find results by condition
    full = next((r for r in results if r.condition_name == "full_system"), None)
    no_excl = next((r for r in results if r.condition_name == "no_exclusivity"), None)
    no_fs = next((r for r in results if r.condition_name == "no_fitness_sharing"), None)
    comp_only = next((r for r in results if r.condition_name == "competition_only"), None)

    # Calculate contributions
    baseline = comp_only.mean_sci if comp_only else 0
    excl_contrib = (no_fs.mean_sci - baseline) if no_fs else 0
    fs_contrib = (no_excl.mean_sci - baseline) if no_excl else 0

    # Best condition
    best = max(results, key=lambda r: r.mean_sci)

    summary = AblationSummary(
        results=[asdict(r) for r in results],
        best_condition=best.condition_name,
        competition_only_sci=baseline,
        exclusivity_contribution=excl_contrib,
        fitness_sharing_contribution=fs_contrib,
        timestamp=datetime.now().isoformat(),
        n_agents=n_agents,
        n_generations=n_generations,
        n_runs=n_runs
    )

    # Print summary
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    print(f"{'Condition':<25} {'SCI':>10} {'Coverage':>10} {'L3 Count':>10} {'Super-Agents':>12}")
    print("-" * 70)

    for r in results:
        print(f"{r.condition_name:<25} {r.mean_sci:>10.3f} {r.coverage:>10.1%} {r.num_l3_specialists:>10.1f} {r.super_agent_count:>12.1f}")

    print("-" * 70)
    print(f"\n🔑 KEY FINDING:")
    print(f"   Competition-only SCI: {baseline:.3f}")
    if baseline > 0.3:
        print(f"   ✅ COMPETITION ALONE produces meaningful specialization (SCI > 0.3)")
        print(f"   → Exclusivity is a SAFETY NET, not the SOURCE of diversity")
    else:
        print(f"   ⚠️  Competition alone produces weak specialization (SCI < 0.3)")
        print(f"   → Exclusivity may be necessary for strong results")

    print(f"\n   Exclusivity contribution: {excl_contrib:+.3f}")
    print(f"   Fitness sharing contribution: {fs_contrib:+.3f}")

    if comp_only and comp_only.super_agent_count > 0:
        print(f"\n   ⚠️  Without exclusivity: {comp_only.super_agent_count:.1f} super-agents emerged")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run exclusivity ablation study")
    parser.add_argument("--agents", type=int, default=12)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--runs", type=int, default=10)

    args = parser.parse_args()

    summary = run_ablation_study(
        n_agents=args.agents,
        n_generations=args.generations,
        n_runs=args.runs
    )

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "exclusivity_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(asdict(summary), f, indent=2)

    print(f"\nResults saved to {output_dir / 'ablation_results.json'}")
