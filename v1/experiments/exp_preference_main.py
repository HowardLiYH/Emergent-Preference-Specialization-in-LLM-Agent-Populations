"""
Phase 1: Main Preference Experiment

Runs the core experiment with:
- 12 agents starting identical
- Confidence-based competition
- 3-level strategy accumulation
- Fitness sharing for diversity
- 100 generations

Tracks RPI, PSI, PD, PPC metrics.
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
from genesis.preference_agent import PreferenceAgent, create_population
from genesis.competition_v3 import run_competition, simulate_competition
from genesis.fitness_sharing_v3 import apply_fitness_sharing, compute_population_diversity_score
from genesis.preference_metrics import (
    compute_rpi_variance,
    compute_population_psi,
    compute_preference_diversity,
    compute_all_metrics,
    print_metrics_summary,
    PreferenceMetricsSummary
)
from genesis.llm_client import LLMClient


@dataclass
class ExperimentConfig:
    """Configuration for Phase 1 experiment."""
    num_agents: int = 12
    num_generations: int = 100
    tasks_per_generation: int = 8
    num_seeds: int = 5

    # Fitness sharing
    use_fitness_sharing: bool = True

    # Competition
    use_real_llm: bool = True

    # Metrics checkpoints
    checkpoint_interval: int = 10

    # Phase 2 requirements
    min_unique_specializations: int = 4
    min_specialists_per_rule: int = 2


@dataclass
class SeedResult:
    """Results from one seed of the experiment."""
    seed: int
    metrics_history: List[Dict]
    final_agents: List[Dict]
    phase2_ready: bool
    phase2_msg: str
    final_rpi_variance: float
    final_psi: float
    final_pd: float


def validate_for_phase2(
    agents: List[PreferenceAgent],
    min_unique: int = 4,
    min_per_rule: int = 2
) -> Tuple[bool, str]:
    """Check if we have enough specialists for Phase 2."""
    specialist_counts = {}
    for agent in agents:
        pref = agent.get_primary_preference()
        if pref and agent.get_preference_strength() > 0.15:
            specialist_counts[pref] = specialist_counts.get(pref, 0) + 1

    unique_specs = len(specialist_counts)
    if unique_specs < min_unique:
        return False, f"Only {unique_specs} unique specializations (need {min_unique})"

    valid_rules = [r for r, c in specialist_counts.items() if c >= min_per_rule]
    if len(valid_rules) < 2:
        return False, f"Need 2+ rules with {min_per_rule}+ specialists"

    return True, f"Valid: {unique_specs} specs, {len(valid_rules)} rules with 2+ specialists"


async def run_single_seed(
    seed: int,
    config: ExperimentConfig,
    client: Optional[LLMClient] = None
) -> SeedResult:
    """Run experiment for a single seed."""
    print(f"\n{'='*60}", flush=True)
    print(f"SEED {seed}", flush=True)
    print(f"{'='*60}", flush=True)

    random.seed(seed)
    np.random.seed(seed)

    # Initialize population
    agents = create_population(config.num_agents, prefix=f"s{seed}_agent")
    metrics_history = []

    for gen in range(config.num_generations):
        # Sample tasks from all rules
        rule_types = list(RuleType)
        tasks = []
        for _ in range(config.tasks_per_generation):
            rule = random.choice(rule_types)
            task = generate_tasks(rule, n=1, seed=seed * 1000 + gen)[0]
            tasks.append(task)

        # Run competitions
        for task in tasks:
            if config.use_real_llm and client:
                winner_id, results = await run_competition(agents, task, client)
            else:
                winner_id, results = simulate_competition(agents, task)

            if winner_id:
                winner = next(a for a in agents if a.agent_id == winner_id)

                # Record attempts
                for agent in agents:
                    agent.record_attempt(task.rule_type, won=(agent.agent_id == winner_id))

                # Apply fitness sharing
                if config.use_fitness_sharing:
                    apply_fitness_sharing(agents, winner, task.rule_type)
                else:
                    winner.accumulate_strategy(task.rule_type)

        # Checkpoint
        if gen % config.checkpoint_interval == 0:
            for agent in agents:
                agent.snapshot_preferences()

            metrics = compute_all_metrics(agents, gen)
            metrics_history.append(metrics.to_dict())

            if gen % 10 == 0:
                print(f"  Gen {gen}: RPI_var={metrics.rpi_variance:.4f}, PSI={metrics.population_psi:.2f}, PD={metrics.preference_diversity:.2f}", flush=True)

            # INCREMENTAL SAVE - save checkpoint to disk for crash recovery
            checkpoint_dir = Path(__file__).parent.parent / "results" / "phase1_checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_file = checkpoint_dir / f"checkpoint_seed{seed}_gen{gen}.json"

            agent_snapshot = []
            for agent in agents:
                pref = agent.get_primary_preference()
                agent_snapshot.append({
                    "agent_id": agent.agent_id,
                    "strategy_levels": {r.value: l for r, l in agent.strategy_levels.items()},
                    "primary_preference": pref.value if pref else None,
                    "preference_strength": agent.get_preference_strength(),
                })

            with open(checkpoint_file, "w") as f:
                json.dump({
                    "seed": seed,
                    "generation": gen,
                    "metrics": metrics.to_dict(),
                    "agents": agent_snapshot
                }, f, indent=2)

            # Keep only last 3 checkpoints per seed to save space
            old_checkpoints = sorted(checkpoint_dir.glob(f"checkpoint_seed{seed}_gen*.json"))
            for old_cp in old_checkpoints[:-3]:
                old_cp.unlink()

    # Final metrics
    final_metrics = compute_all_metrics(agents, config.num_generations)
    print_metrics_summary(final_metrics)

    # Validate for Phase 2
    phase2_ready, phase2_msg = validate_for_phase2(
        agents,
        config.min_unique_specializations,
        config.min_specialists_per_rule
    )

    # Serialize agents
    agent_dicts = []
    for agent in agents:
        agent_dicts.append({
            "agent_id": agent.agent_id,
            "strategy_levels": {r.value: l for r, l in agent.strategy_levels.items()},
            "primary_preference": agent.get_primary_preference().value if agent.get_primary_preference() else None,
            "preference_strength": agent.get_preference_strength(),
            "total_levels": agent.get_total_strategy_levels(),
        })

    return SeedResult(
        seed=seed,
        metrics_history=metrics_history,
        final_agents=agent_dicts,
        phase2_ready=phase2_ready,
        phase2_msg=phase2_msg,
        final_rpi_variance=final_metrics.rpi_variance,
        final_psi=final_metrics.population_psi,
        final_pd=final_metrics.preference_diversity,
    )


async def run_phase1(
    config: ExperimentConfig,
    client: Optional[LLMClient] = None,
    save_results: bool = True
) -> List[SeedResult]:
    """Run Phase 1 across all seeds."""
    print("=" * 60, flush=True)
    print("PHASE 1: MAIN PREFERENCE EXPERIMENT", flush=True)
    print("=" * 60, flush=True)
    print(f"Agents: {config.num_agents}", flush=True)
    print(f"Generations: {config.num_generations}", flush=True)
    print(f"Seeds: {config.num_seeds}", flush=True)
    print(f"Fitness sharing: {config.use_fitness_sharing}", flush=True)
    print(f"Real LLM: {config.use_real_llm}", flush=True)

    results = []
    for seed in range(config.num_seeds):
        result = await run_single_seed(seed, config, client)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 1 SUMMARY")
    print("=" * 60)

    avg_rpi = np.mean([r.final_rpi_variance for r in results])
    avg_psi = np.mean([r.final_psi for r in results])
    avg_pd = np.mean([r.final_pd for r in results])
    phase2_ready_count = sum(1 for r in results if r.phase2_ready)

    print(f"Average RPI Variance: {avg_rpi:.4f} {'✅' if avg_rpi > 0.15 else '❌'}")
    print(f"Average PSI: {avg_psi:.2f} {'✅' if avg_psi > 0.7 else '❌'}")
    print(f"Average PD: {avg_pd:.2f} {'✅' if avg_pd > 0.5 else '❌'}")
    print(f"Seeds ready for Phase 2: {phase2_ready_count}/{config.num_seeds}")

    # Save results
    if save_results:
        output_dir = Path(__file__).parent.parent / "results" / "phase1_main"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"phase1_results_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)

        print(f"\nResults saved to {output_file}")

    return results


async def run_phase1_main(
    api_key: str = None,
    model: str = "gemini-2.5-flash",
    num_seeds: int = 3,
    num_generations: int = 50,
    use_real_llm: bool = True
) -> List[SeedResult]:
    """Main entry point."""
    config = ExperimentConfig(
        num_agents=12,
        num_generations=num_generations,
        num_seeds=num_seeds,
        use_real_llm=use_real_llm,
    )

    client = None
    if use_real_llm and api_key:
        client = LLMClient.for_gemini(api_key=api_key, model=model)

    try:
        return await run_phase1(config, client)
    finally:
        if client:
            await client.close()


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Run Phase 1 experiment")
    parser.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY", ""))
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--simulate", action="store_true", help="Use simulation instead of LLM")

    args = parser.parse_args()

    asyncio.run(run_phase1_main(
        api_key=args.api_key,
        model=args.model,
        num_seeds=args.seeds,
        num_generations=args.generations,
        use_real_llm=not args.simulate,
    ))
