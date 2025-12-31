#!/usr/bin/env python3
"""
Experiment 1: Baseline Specialization

Tests whether LLM agents develop specialization through competition
and directed prompt evolution.

Expected outcome:
- LSI increases from ~0.2 to >0.6 over 100 generations
- Agents differentiate into distinct task-type specialists
- Population develops full coverage of task types
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from genesis import (
    GenesisSimulation,
    LLMClient,
    create_client,
)
from genesis.simulation import SimulationConfig, run_experiment
from genesis.metrics import compute_all_metrics


async def run_baseline_experiment(
    n_runs: int = 10,
    n_agents: int = 16,
    n_generations: int = 100,
    tasks_per_generation: int = 10,
    output_dir: str = "results/baseline",
    verbose: bool = True
):
    """
    Run the baseline specialization experiment.

    Args:
        n_runs: Number of independent runs for statistical significance
        n_agents: Number of agents in the population
        n_generations: Number of generations to run
        tasks_per_generation: Tasks per generation
        output_dir: Directory to save results
        verbose: Print progress
    """
    print("=" * 60)
    print("EXPERIMENT 1: BASELINE SPECIALIZATION")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Agents: {n_agents}")
    print(f"  Generations: {n_generations}")
    print(f"  Tasks/Generation: {tasks_per_generation}")
    print(f"  Runs: {n_runs}")
    print(f"  Output: {output_dir}")
    print()

    # Create configuration
    config = SimulationConfig(
        n_agents=n_agents,
        n_generations=n_generations,
        tasks_per_generation=tasks_per_generation,
        evolution_type="directed",
        winner_takes_all=True,
        log_every=10 if verbose else 100,
    )

    # Create LLM client
    client = create_client()

    try:
        # Run experiment
        results = await run_experiment(
            config=config,
            llm_client=client,
            n_runs=n_runs,
            output_dir=output_dir
        )

        # Print summary
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"\nFinal LSI across {n_runs} runs:")
        print(f"  Mean: {results['final_lsi']['mean']:.3f}")
        print(f"  Std:  {results['final_lsi']['std']:.3f}")
        print(f"  Min:  {results['final_lsi']['min']:.3f}")
        print(f"  Max:  {results['final_lsi']['max']:.3f}")

        # Check success criteria
        print("\n" + "-" * 40)
        print("SUCCESS CRITERIA:")
        target_lsi = 0.6
        achieved_lsi = results['final_lsi']['mean']
        if achieved_lsi >= target_lsi:
            print(f"  ✓ LSI target ({target_lsi}) ACHIEVED: {achieved_lsi:.3f}")
        else:
            print(f"  ✗ LSI target ({target_lsi}) NOT MET: {achieved_lsi:.3f}")

        # Cost report
        print("\n" + "-" * 40)
        print("COST REPORT:")
        print(client.get_usage_report())

        return results

    finally:
        await client.close()


async def run_single_demo(
    n_agents: int = 4,
    n_generations: int = 10,
    verbose: bool = True
):
    """
    Run a quick demo to verify the system works.

    Uses fewer agents and generations for quick testing.
    """
    print("=" * 60)
    print("DEMO RUN: Quick verification")
    print("=" * 60)

    config = SimulationConfig(
        n_agents=n_agents,
        n_generations=n_generations,
        tasks_per_generation=5,
        evolution_type="directed",
        winner_takes_all=True,
        log_every=1,
    )

    client = create_client()

    try:
        sim = GenesisSimulation(config, client)
        result = await sim.run()

        print("\n" + "=" * 60)
        print("DEMO RESULTS")
        print("=" * 60)
        print(f"\nFinal LSI: {result.final_metrics['lsi']['mean']:.3f}")
        print(f"Duration: {result.duration_seconds:.1f}s")

        print("\nAgent Specializations:")
        for agent in result.final_agents:
            perf = agent.get_performance_by_type()
            best = agent.get_best_task_type()
            print(f"  {agent.id}: {best} ({perf})")

        print("\n" + client.get_usage_report())

        return result

    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline specialization experiment"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "demo"],
        default="demo",
        help="Run mode: 'full' for complete experiment, 'demo' for quick test"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs (full mode only)"
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=16,
        help="Number of agents"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="Number of generations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/baseline",
        help="Output directory"
    )

    args = parser.parse_args()

    if args.mode == "demo":
        asyncio.run(run_single_demo(
            n_agents=min(args.agents, 4),
            n_generations=min(args.generations, 10)
        ))
    else:
        asyncio.run(run_baseline_experiment(
            n_runs=args.runs,
            n_agents=args.agents,
            n_generations=args.generations,
            output_dir=args.output
        ))


if __name__ == "__main__":
    main()
