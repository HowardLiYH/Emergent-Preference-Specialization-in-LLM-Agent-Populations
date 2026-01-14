"""
Baseline Experiments

5 baseline conditions:
1. No evolution - Agents keep initial prompt
2. Random evolution - LLM makes arbitrary changes
3. Random strategy - Random strategies from library (not domain-matched)
4. Single best - All agents get top-performing strategy
5. Hand-crafted specialists - Expert-designed prompts (upper bound)
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import random
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
from genesis.fitness_sharing import apply_fitness_sharing, FitnessSharingConfig
from genesis.cost_tracker import CostTracker
from genesis.llm_client import create_client
import numpy as np

def compute_lsi_from_dict(perf: Dict[str, float]) -> float:
    """Compute LSI from performance dictionary."""
    if not perf or len(perf) < 2:
        return 0.0
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
    lsis = [compute_lsi_from_dict(p) for p in performances.values()]
    return float(np.mean(lsis)) if lsis else 0.0


class BaselineType(Enum):
    """Types of baseline experiments."""
    CUMULATIVE = "cumulative"  # Main method
    NO_EVOLUTION = "no_evolution"
    RANDOM_EVOLUTION = "random_evolution"
    RANDOM_STRATEGY = "random_strategy"
    SINGLE_BEST = "single_best"
    HANDCRAFTED = "handcrafted"
    # NEW: Negative controls
    SHUFFLED_DOMAINS = "shuffled_domains"
    RANDOM_WINNER = "random_winner"
    LENGTH_MATCHED = "length_matched"


@dataclass
class BaselineConfig:
    """Configuration for baseline experiment."""
    baseline_type: BaselineType
    num_agents: int = 10
    num_generations: int = 50
    tasks_per_generation: int = 10
    use_fitness_sharing: bool = True
    seed: int = 42


@dataclass
class BaselineResult:
    """Result from a baseline experiment."""
    baseline_type: str
    config: Dict
    final_lsi: float
    lsi_history: List[float]
    specialist_counts: Dict[str, int]
    total_cost: float
    duration_seconds: float


async def simulate_task_performance(
    agent: SpecialistPrompt,
    domain: DomainType
) -> float:
    """Simulate task performance (for baseline testing without API calls)."""
    # Specialists perform better in their domain
    if agent.primary_domain == domain:
        # Check how many strategies from this domain the agent has
        domain_strategies = [s for s in agent.acquired_strategies if s.domain == domain]
        base_score = 0.6 + 0.04 * len(domain_strategies)
        return min(0.95, base_score + random.uniform(-0.1, 0.1))
    else:
        # Generalists or out-of-domain
        return 0.3 + random.uniform(-0.1, 0.1)


async def run_baseline_experiment(
    config: BaselineConfig,
    tracker: CostTracker
) -> BaselineResult:
    """Run a single baseline experiment."""
    import time
    start_time = time.time()

    random.seed(config.seed)
    tracker.set_phase(config.baseline_type.value)

    # Initialize agents
    agents = []

    if config.baseline_type == BaselineType.HANDCRAFTED:
        # Create one specialist per domain
        for domain in get_all_domains():
            agent = create_specialist_prompt(
                f"handcrafted_{domain.value}",
                domain,
                num_strategies=8,
                num_examples=3
            )
            agents.append(agent)
    else:
        # Create initial generalist agents
        for i in range(config.num_agents):
            agents.append(create_initial_prompt(f"agent_{i}"))

    lsi_history = []
    domains = list(get_all_domains())

    # Run generations
    for gen in range(config.num_generations):
        # Sample tasks for this generation
        gen_domains = random.choices(domains, k=config.tasks_per_generation)

        for domain in gen_domains:
            # Each agent attempts the task
            scores = {}
            for agent in agents:
                score = await simulate_task_performance(agent, domain)
                scores[agent.agent_id] = score
                agent.record_attempt(domain, won=False)  # Will update winner below

            # Find winner
            if scores:
                winner_id = max(scores.keys(), key=lambda x: scores[x])
                winner = next(a for a in agents if a.agent_id == winner_id)
                winner.record_attempt(domain, won=True)
                # Fix the previous False recording
                winner.domain_wins[domain.value] = winner.domain_wins.get(domain.value, 0) + 1
                winner.total_wins += 1

                # Evolution based on baseline type
                if config.baseline_type == BaselineType.CUMULATIVE:
                    evolve_cumulative(winner, domain, scores[winner_id])
                elif config.baseline_type == BaselineType.RANDOM_STRATEGY:
                    evolve_random_strategy(winner, domain, scores[winner_id])
                elif config.baseline_type == BaselineType.NO_EVOLUTION:
                    evolve_no_change(winner, domain, scores[winner_id])
                elif config.baseline_type == BaselineType.RANDOM_EVOLUTION:
                    # Random strategy from any domain
                    evolve_random_strategy(winner, domain, scores[winner_id])
                # HANDCRAFTED and SINGLE_BEST don't evolve

        # Calculate LSI at end of generation
        if agents:
            # Build performance matrix for LSI
            performances = {}
            for agent in agents:
                agent_perf = {}
                for d in domains:
                    attempts = agent.domain_attempts.get(d.value, 0)
                    wins = agent.domain_wins.get(d.value, 0)
                    agent_perf[d.value] = wins / attempts if attempts > 0 else 0.0
                performances[agent.agent_id] = agent_perf

            # Compute population LSI
            pop_lsi = compute_population_lsi(performances)
            lsi_history.append(pop_lsi)

    # Final statistics
    specialist_counts = {}
    for agent in agents:
        if agent.primary_domain:
            domain_key = agent.primary_domain.value
            specialist_counts[domain_key] = specialist_counts.get(domain_key, 0) + 1
        else:
            specialist_counts["generalist"] = specialist_counts.get("generalist", 0) + 1

    duration = time.time() - start_time

    return BaselineResult(
        baseline_type=config.baseline_type.value,
        config={
            "num_agents": config.num_agents,
            "num_generations": config.num_generations,
            "tasks_per_generation": config.tasks_per_generation,
            "seed": config.seed
        },
        final_lsi=lsi_history[-1] if lsi_history else 0.0,
        lsi_history=lsi_history,
        specialist_counts=specialist_counts,
        total_cost=0.0,  # Simulated, no API costs
        duration_seconds=duration
    )


async def run_all_baselines(
    num_seeds: int = 5
) -> Dict[str, List[BaselineResult]]:
    """Run all baseline experiments."""
    print("=" * 70)
    print("BASELINE EXPERIMENTS")
    print("=" * 70)

    tracker = CostTracker(experiment_name="baselines")
    results = {}

    baseline_types = [
        BaselineType.CUMULATIVE,
        BaselineType.NO_EVOLUTION,
        BaselineType.RANDOM_STRATEGY,
        BaselineType.HANDCRAFTED,
    ]

    for baseline in baseline_types:
        print(f"\nRunning {baseline.value}...")
        baseline_results = []

        for seed in range(num_seeds):
            config = BaselineConfig(
                baseline_type=baseline,
                num_agents=10,
                num_generations=50,
                tasks_per_generation=10,
                seed=seed
            )

            result = await run_baseline_experiment(config, tracker)
            baseline_results.append(result)
            print(f"  Seed {seed}: LSI = {result.final_lsi:.3f}")

        results[baseline.value] = baseline_results

        # Summary for this baseline
        avg_lsi = sum(r.final_lsi for r in baseline_results) / len(baseline_results)
        print(f"  Average LSI: {avg_lsi:.3f}")

    # Print comparison
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)
    print(f"{'Baseline':<20} {'Avg LSI':>10} {'Std':>10}")
    print("-" * 40)

    for baseline, baseline_results in results.items():
        lsis = [r.final_lsi for r in baseline_results]
        avg = sum(lsis) / len(lsis)
        std = (sum((x - avg) ** 2 for x in lsis) / len(lsis)) ** 0.5
        print(f"{baseline:<20} {avg:>10.3f} {std:>10.3f}")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_results = {}
    for baseline, baseline_results in results.items():
        json_results[baseline] = [
            {
                "baseline_type": r.baseline_type,
                "config": r.config,
                "final_lsi": r.final_lsi,
                "specialist_counts": r.specialist_counts,
                "duration_seconds": r.duration_seconds
            }
            for r in baseline_results
        ]

    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'baseline_results.json'}")

    return results


# ============================================================================
# NEW NEGATIVE CONTROLS (Professor's Review)
# ============================================================================

async def run_shuffled_domains_baseline(
    config: BaselineConfig,
    tracker: CostTracker
) -> BaselineResult:
    """
    SHUFFLED DOMAINS: Evolve with WRONG domain labels.

    Tests whether specialization is due to domain structure or random drift.
    If LSI is high here, the method isn't learning domain structure.
    """
    import time
    start_time = time.time()

    random.seed(config.seed)
    tracker.set_phase("shuffled_domains")

    agents = [create_initial_prompt(f"agent_{i}") for i in range(config.num_agents)]
    lsi_history = []
    domains = list(get_all_domains())

    for gen in range(config.num_generations):
        gen_domains = random.choices(domains, k=config.tasks_per_generation)

        for real_domain in gen_domains:
            # LIE about the domain - use random domain instead
            fake_domain = random.choice(domains)

            scores = {}
            for agent in agents:
                # Simulate performance based on REAL domain
                if agent.primary_domain == real_domain:
                    score = 0.6 + random.uniform(-0.1, 0.1)
                else:
                    score = 0.3 + random.uniform(-0.1, 0.1)
                scores[agent.agent_id] = score
                agent.record_attempt(fake_domain, won=False)  # Record with FAKE domain

            if scores:
                winner_id = max(scores.keys(), key=lambda x: scores[x])
                winner = next(a for a in agents if a.agent_id == winner_id)
                winner.domain_wins[fake_domain.value] = winner.domain_wins.get(fake_domain.value, 0) + 1
                winner.total_wins += 1
                # Evolve with FAKE domain
                evolve_cumulative(winner, fake_domain, scores[winner_id])

        # Calculate LSI
        performances = {}
        for agent in agents:
            agent_perf = {}
            for d in domains:
                attempts = agent.domain_attempts.get(d.value, 0)
                wins = agent.domain_wins.get(d.value, 0)
                agent_perf[d.value] = wins / attempts if attempts > 0 else 0.0
            performances[agent.agent_id] = agent_perf
        lsi_history.append(compute_population_lsi(performances))

    duration = time.time() - start_time

    specialist_counts = {}
    for agent in agents:
        if agent.primary_domain:
            specialist_counts[agent.primary_domain.value] = specialist_counts.get(agent.primary_domain.value, 0) + 1

    return BaselineResult(
        baseline_type="shuffled_domains",
        config={"num_agents": config.num_agents, "num_generations": config.num_generations, "seed": config.seed},
        final_lsi=lsi_history[-1] if lsi_history else 0.0,
        lsi_history=lsi_history,
        specialist_counts=specialist_counts,
        total_cost=0.0,
        duration_seconds=duration
    )


async def run_random_winner_baseline(
    config: BaselineConfig,
    tracker: CostTracker
) -> BaselineResult:
    """
    RANDOM WINNER: Random agent evolves, not the best performer.

    Tests whether competition (winner-take-all) is necessary.
    If LSI is high here, competition doesn't matter.
    """
    import time
    start_time = time.time()

    random.seed(config.seed)
    tracker.set_phase("random_winner")

    agents = [create_initial_prompt(f"agent_{i}") for i in range(config.num_agents)]
    lsi_history = []
    domains = list(get_all_domains())

    for gen in range(config.num_generations):
        gen_domains = random.choices(domains, k=config.tasks_per_generation)

        for domain in gen_domains:
            # All agents attempt
            for agent in agents:
                agent.record_attempt(domain, won=False)

            # Pick RANDOM winner, not best performer
            winner = random.choice(agents)
            winner.domain_wins[domain.value] = winner.domain_wins.get(domain.value, 0) + 1
            winner.total_wins += 1
            evolve_cumulative(winner, domain, 0.5)

        # Calculate LSI
        performances = {}
        for agent in agents:
            agent_perf = {d.value: agent.domain_wins.get(d.value, 0) / max(1, agent.domain_attempts.get(d.value, 1))
                         for d in domains}
            performances[agent.agent_id] = agent_perf
        lsi_history.append(compute_population_lsi(performances))

    duration = time.time() - start_time

    specialist_counts = {}
    for agent in agents:
        if agent.primary_domain:
            specialist_counts[agent.primary_domain.value] = specialist_counts.get(agent.primary_domain.value, 0) + 1

    return BaselineResult(
        baseline_type="random_winner",
        config={"num_agents": config.num_agents, "num_generations": config.num_generations, "seed": config.seed},
        final_lsi=lsi_history[-1] if lsi_history else 0.0,
        lsi_history=lsi_history,
        specialist_counts=specialist_counts,
        total_cost=0.0,
        duration_seconds=duration
    )


async def run_length_matched_baseline(
    config: BaselineConfig,
    tracker: CostTracker
) -> BaselineResult:
    """
    LENGTH MATCHED: Add random text instead of strategies.

    Tests whether it's the strategies or just prompt length that matters.
    If LSI is high here, it's not about strategy content.
    """
    import time
    import string
    start_time = time.time()

    random.seed(config.seed)
    tracker.set_phase("length_matched")

    # Track agents with random text instead of strategies
    agents = [create_initial_prompt(f"agent_{i}") for i in range(config.num_agents)]
    agent_random_text = {a.agent_id: [] for a in agents}

    lsi_history = []
    domains = list(get_all_domains())

    def generate_random_text(length: int = 50) -> str:
        """Generate random text of similar length to a strategy."""
        words = [''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
                 for _ in range(length // 5)]
        return ' '.join(words)

    for gen in range(config.num_generations):
        gen_domains = random.choices(domains, k=config.tasks_per_generation)

        for domain in gen_domains:
            scores = {}
            for agent in agents:
                # Length of random text doesn't help performance
                score = 0.3 + random.uniform(-0.1, 0.1)
                scores[agent.agent_id] = score
                agent.record_attempt(domain, won=False)

            if scores:
                winner_id = max(scores.keys(), key=lambda x: scores[x])
                winner = next(a for a in agents if a.agent_id == winner_id)
                winner.domain_wins[domain.value] = winner.domain_wins.get(domain.value, 0) + 1
                winner.total_wins += 1

                # Add RANDOM TEXT instead of strategy
                agent_random_text[winner_id].append(generate_random_text())

        # Calculate LSI
        performances = {}
        for agent in agents:
            agent_perf = {d.value: agent.domain_wins.get(d.value, 0) / max(1, agent.domain_attempts.get(d.value, 1))
                         for d in domains}
            performances[agent.agent_id] = agent_perf
        lsi_history.append(compute_population_lsi(performances))

    duration = time.time() - start_time

    specialist_counts = {}
    for agent in agents:
        if agent.primary_domain:
            specialist_counts[agent.primary_domain.value] = specialist_counts.get(agent.primary_domain.value, 0) + 1

    return BaselineResult(
        baseline_type="length_matched",
        config={"num_agents": config.num_agents, "num_generations": config.num_generations, "seed": config.seed},
        final_lsi=lsi_history[-1] if lsi_history else 0.0,
        lsi_history=lsi_history,
        specialist_counts=specialist_counts,
        total_cost=0.0,
        duration_seconds=duration
    )


async def run_all_baselines_v2(num_seeds: int = 5) -> Dict[str, List[BaselineResult]]:
    """Run all baselines including new negative controls."""
    print("=" * 70)
    print("BASELINE EXPERIMENTS V2 (with negative controls)")
    print("=" * 70)

    tracker = CostTracker(experiment_name="baselines_v2")
    results = {}

    # Original baselines
    baseline_types = [
        BaselineType.CUMULATIVE,
        BaselineType.NO_EVOLUTION,
        BaselineType.RANDOM_STRATEGY,
        BaselineType.HANDCRAFTED,
    ]

    for baseline in baseline_types:
        print(f"\nRunning {baseline.value}...")
        baseline_results = []
        for seed in range(num_seeds):
            config = BaselineConfig(baseline_type=baseline, seed=seed)
            result = await run_baseline_experiment(config, tracker)
            baseline_results.append(result)
            print(f"  Seed {seed}: LSI = {result.final_lsi:.3f}")
        results[baseline.value] = baseline_results

    # NEW: Negative controls
    print("\n--- NEGATIVE CONTROLS ---")

    for name, func in [
        ("shuffled_domains", run_shuffled_domains_baseline),
        ("random_winner", run_random_winner_baseline),
        ("length_matched", run_length_matched_baseline),
    ]:
        print(f"\nRunning {name}...")
        baseline_results = []
        for seed in range(num_seeds):
            config = BaselineConfig(baseline_type=BaselineType.CUMULATIVE, seed=seed)
            result = await func(config, tracker)
            baseline_results.append(result)
            print(f"  Seed {seed}: LSI = {result.final_lsi:.3f}")
        results[name] = baseline_results

    # Print comparison
    print("\n" + "=" * 70)
    print("FULL COMPARISON (including negative controls)")
    print("=" * 70)
    print(f"{'Baseline':<20} {'Avg LSI':>10} {'Std':>10} {'Type':>15}")
    print("-" * 55)

    for baseline, baseline_results in results.items():
        lsis = [r.final_lsi for r in baseline_results]
        avg = sum(lsis) / len(lsis)
        std = (sum((x - avg) ** 2 for x in lsis) / len(lsis)) ** 0.5
        ctrl_type = "NEGATIVE" if baseline in ["shuffled_domains", "random_winner", "length_matched"] else "baseline"
        print(f"{baseline:<20} {avg:>10.3f} {std:>10.3f} {ctrl_type:>15}")

    return results


if __name__ == "__main__":
    asyncio.run(run_all_baselines_v2(num_seeds=3))
