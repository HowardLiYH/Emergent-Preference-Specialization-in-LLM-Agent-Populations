"""
Phase 3: Ablation Studies

Tests which components are necessary for preference specialization:
1. NO_FITNESS_SHARING: Remove diversity pressure
2. RANDOM_WINNER: Random winner selection (no merit)
3. NO_ACCUMULATION: Reset strategies each generation
4. SHUFFLED_TASKS: Random task-rule assignment
5. HANDCRAFTED_CEILING: Upper bound with perfect specialists

Each ablation removes ONE component to measure its contribution.
"""

import asyncio
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genesis.synthetic_rules import RuleType, generate_tasks, RuleTask
from genesis.preference_agent import PreferenceAgent, create_population, create_handcrafted_specialist
from genesis.competition_v3 import simulate_competition
from genesis.fitness_sharing_v3 import apply_fitness_sharing
from genesis.preference_metrics import compute_preference_diversity


@dataclass
class AblationResult:
    """Result of one ablation condition."""
    condition: str
    description: str
    preference_diversity: float
    num_unique_preferences: int
    avg_preference_strength: float
    agent_preferences: Dict[str, str]

    @property
    def summary(self) -> str:
        return f"{self.condition}: PD={self.preference_diversity:.2f}, unique={self.num_unique_preferences}, strength={self.avg_preference_strength:.2f}"


def run_baseline(num_agents: int = 12, num_generations: int = 100, seed: int = 42) -> Tuple[List[PreferenceAgent], Dict]:
    """Run the full experiment with all components (baseline)."""
    random.seed(seed)
    np.random.seed(seed)

    agents = create_population(num_agents, prefix="baseline")

    for gen in range(num_generations):
        for _ in range(8):
            rule = random.choice(list(RuleType))
            task = generate_tasks(rule, n=1, seed=seed * 1000 + gen)[0]

            winner_id, results = simulate_competition(agents, task)

            if winner_id:
                winner = next(a for a in agents if a.agent_id == winner_id)
                for agent in agents:
                    agent.record_attempt(task.rule_type, won=(agent.agent_id == winner_id))
                apply_fitness_sharing(agents, winner, task.rule_type)

    return agents, {"condition": "BASELINE"}


def run_no_fitness_sharing(num_agents: int = 12, num_generations: int = 100, seed: int = 42) -> Tuple[List[PreferenceAgent], Dict]:
    """Run WITHOUT fitness sharing - winner always gets strategy."""
    random.seed(seed)
    np.random.seed(seed)

    agents = create_population(num_agents, prefix="no_fs")

    for gen in range(num_generations):
        for _ in range(8):
            rule = random.choice(list(RuleType))
            task = generate_tasks(rule, n=1, seed=seed * 1000 + gen)[0]

            winner_id, results = simulate_competition(agents, task)

            if winner_id:
                winner = next(a for a in agents if a.agent_id == winner_id)
                for agent in agents:
                    agent.record_attempt(task.rule_type, won=(agent.agent_id == winner_id))
                # NO fitness sharing - always accumulate
                winner.accumulate_strategy(task.rule_type)

    return agents, {"condition": "NO_FITNESS_SHARING"}


def run_random_winner(num_agents: int = 12, num_generations: int = 100, seed: int = 42) -> Tuple[List[PreferenceAgent], Dict]:
    """Run with RANDOM winner selection (no merit-based competition)."""
    random.seed(seed)
    np.random.seed(seed)

    agents = create_population(num_agents, prefix="random")

    for gen in range(num_generations):
        for _ in range(8):
            rule = random.choice(list(RuleType))
            task = generate_tasks(rule, n=1, seed=seed * 1000 + gen)[0]

            # RANDOM winner instead of competition
            winner = random.choice(agents)

            for agent in agents:
                agent.record_attempt(task.rule_type, won=(agent.agent_id == winner.agent_id))
            apply_fitness_sharing(agents, winner, task.rule_type)

    return agents, {"condition": "RANDOM_WINNER"}


def run_no_accumulation(num_agents: int = 12, num_generations: int = 100, seed: int = 42) -> Tuple[List[PreferenceAgent], Dict]:
    """Run with NO strategy accumulation - reset each generation."""
    random.seed(seed)
    np.random.seed(seed)

    agents = create_population(num_agents, prefix="no_acc")

    for gen in range(num_generations):
        # Reset all strategies at start of each generation
        for agent in agents:
            agent.strategy_levels = {r: 0 for r in RuleType}

        for _ in range(8):
            rule = random.choice(list(RuleType))
            task = generate_tasks(rule, n=1, seed=seed * 1000 + gen)[0]

            winner_id, results = simulate_competition(agents, task)

            if winner_id:
                winner = next(a for a in agents if a.agent_id == winner_id)
                for agent in agents:
                    agent.record_attempt(task.rule_type, won=(agent.agent_id == winner_id))
                apply_fitness_sharing(agents, winner, task.rule_type)

    return agents, {"condition": "NO_ACCUMULATION"}


def run_shuffled_tasks(num_agents: int = 12, num_generations: int = 100, seed: int = 42) -> Tuple[List[PreferenceAgent], Dict]:
    """Run with SHUFFLED task-rule assignments (breaks rule-strategy connection)."""
    random.seed(seed)
    np.random.seed(seed)

    agents = create_population(num_agents, prefix="shuffled")

    for gen in range(num_generations):
        for _ in range(8):
            # Generate task for one rule, but RECORD as different rule
            actual_rule = random.choice(list(RuleType))
            recorded_rule = random.choice(list(RuleType))  # Disconnect!
            task = generate_tasks(actual_rule, n=1, seed=seed * 1000 + gen)[0]

            winner_id, results = simulate_competition(agents, task)

            if winner_id:
                winner = next(a for a in agents if a.agent_id == winner_id)
                for agent in agents:
                    # Record with WRONG rule
                    agent.record_attempt(recorded_rule, won=(agent.agent_id == winner_id))
                # Accumulate strategy for WRONG rule
                winner.accumulate_strategy(recorded_rule)

    return agents, {"condition": "SHUFFLED_TASKS"}


def run_handcrafted_ceiling(num_agents: int = 8) -> Tuple[List[PreferenceAgent], Dict]:
    """Create perfect handcrafted specialists (upper bound)."""
    agents = []
    rules = list(RuleType)

    for i, rule in enumerate(rules):
        agent = create_handcrafted_specialist(rule)
        agent.agent_id = f"handcrafted_{i}"
        agents.append(agent)

    return agents, {"condition": "HANDCRAFTED_CEILING"}


def analyze_results(agents: List[PreferenceAgent], condition: str, description: str) -> AblationResult:
    """Analyze agents and return ablation result."""
    # Count preferences
    preferences = {}
    strengths = []

    for agent in agents:
        pref = agent.get_primary_preference()
        strength = agent.get_preference_strength()

        pref_name = pref.value if pref else "none"
        preferences[agent.agent_id] = pref_name
        strengths.append(strength)

    unique_prefs = len(set(p for p in preferences.values() if p != "none"))
    pd = compute_preference_diversity(agents)
    avg_strength = np.mean(strengths) if strengths else 0

    return AblationResult(
        condition=condition,
        description=description,
        preference_diversity=pd,
        num_unique_preferences=unique_prefs,
        avg_preference_strength=avg_strength,
        agent_preferences=preferences,
    )


def run_all_ablations(
    num_agents: int = 12,
    num_generations: int = 100,
    seed: int = 42,
    save_results: bool = True
) -> List[AblationResult]:
    """Run all ablation conditions and compare."""
    print("=" * 60, flush=True)
    print("PHASE 3: ABLATION STUDIES", flush=True)
    print("=" * 60, flush=True)

    conditions = [
        ("BASELINE", "Full experiment with all components", run_baseline),
        ("NO_FITNESS_SHARING", "Remove diversity pressure", run_no_fitness_sharing),
        ("RANDOM_WINNER", "Random winner (no merit)", run_random_winner),
        ("NO_ACCUMULATION", "Reset strategies each gen", run_no_accumulation),
        ("SHUFFLED_TASKS", "Disconnect rule-strategy", run_shuffled_tasks),
    ]

    results = []

    for name, desc, runner in conditions:
        print(f"\nRunning {name}...", flush=True)
        agents, meta = runner(num_agents, num_generations, seed)
        result = analyze_results(agents, name, desc)
        results.append(result)
        print(f"  {result.summary}", flush=True)

    # Add handcrafted ceiling
    print(f"\nRunning HANDCRAFTED_CEILING...", flush=True)
    agents, meta = run_handcrafted_ceiling()
    result = analyze_results(agents, "HANDCRAFTED_CEILING", "Perfect specialists (upper bound)")
    results.append(result)
    print(f"  {result.summary}", flush=True)

    # Summary table
    print("\n" + "=" * 60, flush=True)
    print("ABLATION SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"{'Condition':<25} | {'PD':>5} | {'Unique':>6} | {'Strength':>8}", flush=True)
    print("-" * 55, flush=True)

    baseline_pd = results[0].preference_diversity

    for r in results:
        delta = r.preference_diversity - baseline_pd
        delta_str = f"({delta:+.2f})" if r.condition != "BASELINE" else ""
        print(f"{r.condition:<25} | {r.preference_diversity:>5.2f} | {r.num_unique_preferences:>6} | {r.avg_preference_strength:>8.2f} {delta_str}", flush=True)

    # Key findings
    print("\n" + "=" * 60, flush=True)
    print("KEY FINDINGS", flush=True)
    print("=" * 60, flush=True)

    baseline = results[0]

    for r in results[1:]:
        if r.condition == "HANDCRAFTED_CEILING":
            gap = baseline.preference_diversity / r.preference_diversity if r.preference_diversity > 0 else 0
            print(f"• Evolved agents achieve {gap:.0%} of handcrafted ceiling", flush=True)
        else:
            drop = baseline.preference_diversity - r.preference_diversity
            if drop > 0.1:
                print(f"• {r.condition}: Removing this HURTS diversity (-{drop:.2f})", flush=True)
            elif drop < -0.1:
                print(f"• {r.condition}: Removing this HELPS diversity (+{-drop:.2f})", flush=True)
            else:
                print(f"• {r.condition}: Minimal effect on diversity ({drop:+.2f})", flush=True)

    # Save results
    if save_results:
        output_dir = Path(__file__).parent.parent / "results" / "phase3_ablation"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"ablation_results_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)

        print(f"\nResults saved to {output_file}", flush=True)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 3 ablation studies")
    parser.add_argument("--agents", type=int, default=12)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_all_ablations(
        num_agents=args.agents,
        num_generations=args.generations,
        seed=args.seed,
    )
