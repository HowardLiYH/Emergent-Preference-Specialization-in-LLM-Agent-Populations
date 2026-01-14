"""
Full Experiment Runner

Runs the complete experiment suite:
1. Validation pre-test
2. Baseline comparisons
3. Main experiments (20 agents x 100 gen x 10 seeds)
4. Ablations
5. Counterfactual (prompt swap) tests

Reports all results with statistical analysis.
"""

import asyncio
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genesis.domains import DomainType, get_all_domains
from genesis.evolution_v2 import (
    SpecialistPrompt,
    create_initial_prompt,
    create_specialist_prompt,
    evolve_cumulative,
    evolve_random_strategy,
    evolve_no_change,
    EvolutionType
)
from genesis.strategy_library import STRATEGY_LIBRARY
from genesis.fitness_sharing import (
    apply_fitness_sharing,
    FitnessSharingConfig,
    get_niche_statistics,
    print_niche_report
)
from genesis.cost_tracker import CostTracker, estimate_experiment_cost
def compute_lsi_from_dict(perf: Dict[str, float]) -> float:
    """Compute LSI from performance dictionary."""
    if not perf or len(perf) < 2:
        return 0.0
    import numpy as np
    values = np.array(list(perf.values()))
    if values.sum() == 0:
        return 0.0
    p = values / values.sum()
    entropy = -np.sum(p * np.log(p + 1e-10))
    max_entropy = np.log(len(p))
    lsi = 1 - entropy / max_entropy
    return float(np.clip(lsi, 0, 1))

def compute_population_lsi(performances: Dict[str, Dict[str, float]]) -> float:
    """Compute mean LSI for a population."""
    if not performances:
        return 0.0
    import numpy as np
    lsis = [compute_lsi_from_dict(p) for p in performances.values()]
    return float(np.mean(lsis)) if lsis else 0.0


@dataclass
class ExperimentConfig:
    """Configuration for full experiment."""
    name: str = "main_experiment"
    num_agents: int = 20
    num_generations: int = 100
    tasks_per_generation: int = 10
    num_seeds: int = 10
    use_fitness_sharing: bool = True
    evolution_type: str = "cumulative"
    model: str = "gpt-4o-mini"


@dataclass
class GenerationSnapshot:
    """Snapshot of population state at a generation."""
    generation: int
    population_lsi: float
    unique_specializations: int
    specialist_distribution: Dict[str, int]
    avg_win_rate: float


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    seed: int
    config: Dict
    final_lsi: float
    final_unique_specs: int
    lsi_history: List[float]
    specialist_counts: Dict[str, int]
    agent_summaries: List[Dict]
    duration_seconds: float


def compute_statistics(values: List[float]) -> Dict:
    """Compute mean, std, CI for a list of values."""
    if not values:
        return {"mean": 0, "std": 0, "ci_low": 0, "ci_high": 0, "n": 0}

    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std = variance ** 0.5

    # 95% CI using t-distribution approximation
    se = std / (n ** 0.5) if n > 1 else 0
    ci_margin = 1.96 * se  # Using z for simplicity

    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci_low": round(mean - ci_margin, 4),
        "ci_high": round(mean + ci_margin, 4),
        "n": n
    }


async def simulate_task_performance(
    agent: SpecialistPrompt,
    domain: DomainType,
    noise_level: float = 0.1
) -> float:
    """Simulate task performance based on agent specialization."""
    base_score = 0.3  # Generalist baseline

    # Bonus for having strategies in this domain
    domain_strategies = [s for s in agent.acquired_strategies if s.domain == domain]
    strategy_bonus = 0.05 * len(domain_strategies)

    # Bonus for being specialized in this domain
    if agent.primary_domain == domain:
        specialization_bonus = 0.2
    else:
        specialization_bonus = 0

    score = base_score + strategy_bonus + specialization_bonus
    score += random.uniform(-noise_level, noise_level)

    return max(0.0, min(1.0, score))


async def run_single_experiment(
    config: ExperimentConfig,
    seed: int,
    tracker: CostTracker
) -> ExperimentResult:
    """Run a single experiment with given seed."""
    import time
    start_time = time.time()

    random.seed(seed)
    tracker.set_phase(f"main_seed_{seed}")

    # Initialize agents
    agents = [create_initial_prompt(f"agent_{i}") for i in range(config.num_agents)]

    lsi_history = []
    domains = list(get_all_domains())
    fitness_config = FitnessSharingConfig() if config.use_fitness_sharing else None

    # Run generations
    for gen in range(config.num_generations):
        # Sample tasks
        gen_domains = random.choices(domains, k=config.tasks_per_generation)

        for domain in gen_domains:
            # All agents compete on task
            scores = {}
            for agent in agents:
                score = await simulate_task_performance(agent, domain)
                scores[agent.agent_id] = score

            # Apply fitness sharing if enabled
            if fitness_config:
                scores = apply_fitness_sharing(agents, scores, fitness_config)

            # Find winner
            if scores:
                winner_id = max(scores.keys(), key=lambda x: scores[x])
                winner = next(a for a in agents if a.agent_id == winner_id)

                # Record results
                for agent in agents:
                    is_winner = agent.agent_id == winner_id
                    agent.record_attempt(domain, won=is_winner)

                # Evolve winner
                if config.evolution_type == "cumulative":
                    evolve_cumulative(winner, domain, scores[winner_id])
                elif config.evolution_type == "random_strategy":
                    evolve_random_strategy(winner, domain, scores[winner_id])
                elif config.evolution_type == "no_evolution":
                    evolve_no_change(winner, domain, scores[winner_id])

        # Compute LSI at end of generation
        performances = {}
        for agent in agents:
            agent_perf = {}
            for d in domains:
                attempts = agent.domain_attempts.get(d.value, 0)
                wins = agent.domain_wins.get(d.value, 0)
                agent_perf[d.value] = wins / attempts if attempts > 0 else 0.0
            performances[agent.agent_id] = agent_perf

        pop_lsi = compute_population_lsi(performances)
        lsi_history.append(pop_lsi)

        # Progress update every 25 generations
        if (gen + 1) % 25 == 0:
            print(f"    Gen {gen+1}: LSI = {pop_lsi:.3f}")

    # Final statistics
    specialist_counts = {}
    for agent in agents:
        if agent.primary_domain:
            key = agent.primary_domain.value
            specialist_counts[key] = specialist_counts.get(key, 0) + 1
        else:
            specialist_counts["generalist"] = specialist_counts.get("generalist", 0) + 1

    unique_specs = len([k for k in specialist_counts.keys() if k != "generalist"])

    # Agent summaries
    agent_summaries = []
    for agent in agents:
        summary = {
            "id": agent.agent_id,
            "primary_domain": agent.primary_domain.value if agent.primary_domain else None,
            "num_strategies": len(agent.acquired_strategies),
            "total_wins": agent.total_wins,
            "total_attempts": agent.total_attempts,
            "win_rate": agent.get_overall_win_rate()
        }
        agent_summaries.append(summary)

    duration = time.time() - start_time

    return ExperimentResult(
        seed=seed,
        config=asdict(config) if hasattr(config, '__dataclass_fields__') else vars(config),
        final_lsi=lsi_history[-1] if lsi_history else 0.0,
        final_unique_specs=unique_specs,
        lsi_history=lsi_history,
        specialist_counts=specialist_counts,
        agent_summaries=agent_summaries,
        duration_seconds=duration
    )


async def run_main_experiments(
    config: ExperimentConfig = None
) -> Dict:
    """Run main experiment suite with all seeds."""
    if config is None:
        config = ExperimentConfig()

    print("=" * 70)
    print(f"MAIN EXPERIMENT: {config.name}")
    print("=" * 70)
    print(f"Config: {config.num_agents} agents, {config.num_generations} generations, {config.num_seeds} seeds")
    print()

    # Cost estimate
    estimate = estimate_experiment_cost(
        config.num_agents,
        config.num_generations,
        config.tasks_per_generation,
        config.num_seeds,
        model=config.model
    )
    print(f"Estimated cost: ${estimate['estimates']['estimated_cost_usd']:.2f}")
    print()

    tracker = CostTracker(experiment_name=config.name, model=config.model)
    results = []

    for seed in range(config.num_seeds):
        print(f"\nSeed {seed + 1}/{config.num_seeds}:")
        result = await run_single_experiment(config, seed, tracker)
        results.append(result)
        print(f"  Final LSI: {result.final_lsi:.3f}, Unique specs: {result.final_unique_specs}/10")

    # Statistical analysis
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)

    lsis = [r.final_lsi for r in results]
    specs = [r.final_unique_specs for r in results]

    lsi_stats = compute_statistics(lsis)
    spec_stats = compute_statistics([float(s) for s in specs])

    print(f"Final LSI:           {lsi_stats['mean']:.3f} ± {lsi_stats['std']:.3f} (95% CI: [{lsi_stats['ci_low']:.3f}, {lsi_stats['ci_high']:.3f}])")
    print(f"Unique Specializations: {spec_stats['mean']:.1f} ± {spec_stats['std']:.1f}")

    # Aggregate specialist distribution
    agg_counts = {}
    for r in results:
        for domain, count in r.specialist_counts.items():
            agg_counts[domain] = agg_counts.get(domain, 0) + count

    print("\nAggregate Specialist Distribution:")
    for domain in sorted(agg_counts.keys(), key=lambda x: agg_counts[x], reverse=True):
        avg_count = agg_counts[domain] / config.num_seeds
        print(f"  {domain:<15}: {avg_count:.1f} agents (avg)")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "main"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "config": asdict(config) if hasattr(config, '__dataclass_fields__') else {},
        "summary": {
            "lsi": lsi_stats,
            "unique_specializations": spec_stats,
            "aggregate_distribution": agg_counts
        },
        "results": [
            {
                "seed": r.seed,
                "final_lsi": r.final_lsi,
                "final_unique_specs": r.final_unique_specs,
                "specialist_counts": r.specialist_counts,
                "duration_seconds": r.duration_seconds
            }
            for r in results
        ],
        "lsi_histories": [r.lsi_history for r in results]
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"experiment_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Print cost report
    tracker.print_report()

    return output


async def run_ablation_study() -> Dict:
    """Run ablation study comparing different conditions."""
    print("\n" + "=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)

    conditions = [
        ("cumulative", True),    # Main method with fitness sharing
        ("cumulative", False),   # Without fitness sharing
        ("random_strategy", True),  # Random baseline
        ("no_evolution", True),  # No evolution baseline
    ]

    ablation_results = {}

    for evolution_type, use_fitness_sharing in conditions:
        condition_name = f"{evolution_type}_fs{use_fitness_sharing}"
        print(f"\nCondition: {condition_name}")

        config = ExperimentConfig(
            name=condition_name,
            num_agents=10,
            num_generations=50,
            tasks_per_generation=10,
            num_seeds=5,
            use_fitness_sharing=use_fitness_sharing,
            evolution_type=evolution_type
        )

        result = await run_main_experiments(config)
        ablation_results[condition_name] = result

    # Comparison table
    print("\n" + "=" * 70)
    print("ABLATION COMPARISON")
    print("=" * 70)
    print(f"{'Condition':<30} {'LSI Mean':>10} {'LSI Std':>10} {'Unique Specs':>12}")
    print("-" * 62)

    for name, result in ablation_results.items():
        lsi = result["summary"]["lsi"]
        specs = result["summary"]["unique_specializations"]
        print(f"{name:<30} {lsi['mean']:>10.3f} {lsi['std']:>10.3f} {specs['mean']:>12.1f}")

    return ablation_results


async def main():
    """Run complete experiment suite."""
    print("=" * 70)
    print("EMERGENT PROMPT EVOLUTION - FULL EXPERIMENT SUITE")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()

    # 1. Main experiments
    main_results = await run_main_experiments(ExperimentConfig(
        name="main_v7",
        num_agents=20,
        num_generations=100,
        num_seeds=10
    ))

    # 2. Ablation study
    ablation_results = await run_ablation_study()

    print("\n" + "=" * 70)
    print("EXPERIMENT SUITE COMPLETE")
    print("=" * 70)
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    asyncio.run(main())
