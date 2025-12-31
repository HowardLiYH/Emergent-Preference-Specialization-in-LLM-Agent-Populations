#!/usr/bin/env python3
"""
Ablation Experiments for Genesis

Tests which components are necessary for emergent specialization:

Experiment 2: No Evolution
- evolution_type = "none"
- Expected: LSI stays flat (~0.2)

Experiment 3: Random Evolution  
- evolution_type = "random"
- Expected: LSI increases less than directed (~0.3-0.4)

Experiment 4: No Competition (All Update)
- winner_takes_all = False
- Expected: Less differentiation, prompts converge

Experiment 5: Temperature Ablation
- Test at T=0.3, 0.7, 1.0
- Expected: Moderate temperature (0.7) works best
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from genesis import create_client
from genesis.simulation import SimulationConfig, GenesisSimulation
import numpy as np


async def run_ablation(
    name: str,
    config: SimulationConfig,
    client,
    n_runs: int = 5,
    output_dir: str = "results/ablations"
) -> Dict[str, Any]:
    """Run a single ablation condition."""
    
    print(f"\n{'='*60}")
    print(f"ABLATION: {name}")
    print(f"{'='*60}")
    
    all_results = []
    
    for run in range(n_runs):
        print(f"\n--- Run {run+1}/{n_runs} ---")
        
        config.seed = run
        sim = GenesisSimulation(config, client)
        result = await sim.run()
        all_results.append(result)
        
        print(f"Final LSI: {result.final_metrics['lsi']['mean']:.3f}")
    
    # Aggregate
    final_lsi = [r.final_metrics['lsi']['mean'] for r in all_results]
    
    summary = {
        "name": name,
        "config": config.__dict__,
        "n_runs": n_runs,
        "final_lsi": {
            "mean": float(np.mean(final_lsi)),
            "std": float(np.std(final_lsi)),
            "min": float(np.min(final_lsi)),
            "max": float(np.max(final_lsi)),
        }
    }
    
    # Save results
    output_path = Path(output_dir) / name
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary


async def run_all_ablations(
    n_runs: int = 5,
    n_agents: int = 16,
    n_generations: int = 50,  # Shorter for ablations
    output_dir: str = "results/ablations"
):
    """Run all ablation experiments."""
    
    print("=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)
    
    client = create_client()
    results = {}
    
    try:
        # Baseline (for comparison)
        baseline_config = SimulationConfig(
            n_agents=n_agents,
            n_generations=n_generations,
            tasks_per_generation=10,
            evolution_type="directed",
            winner_takes_all=True,
        )
        results["baseline"] = await run_ablation(
            "baseline", baseline_config, client, n_runs, output_dir
        )
        
        # Ablation 1: No Evolution
        no_evo_config = SimulationConfig(
            n_agents=n_agents,
            n_generations=n_generations,
            tasks_per_generation=10,
            evolution_type="none",  # No evolution
            winner_takes_all=True,
        )
        results["no_evolution"] = await run_ablation(
            "no_evolution", no_evo_config, client, n_runs, output_dir
        )
        
        # Ablation 2: Random Evolution
        random_evo_config = SimulationConfig(
            n_agents=n_agents,
            n_generations=n_generations,
            tasks_per_generation=10,
            evolution_type="random",  # Random evolution
            winner_takes_all=True,
        )
        results["random_evolution"] = await run_ablation(
            "random_evolution", random_evo_config, client, n_runs, output_dir
        )
        
        # Ablation 3: No Competition (All Update)
        no_comp_config = SimulationConfig(
            n_agents=n_agents,
            n_generations=n_generations,
            tasks_per_generation=10,
            evolution_type="directed",
            winner_takes_all=False,  # All agents evolve
        )
        results["no_competition"] = await run_ablation(
            "no_competition", no_comp_config, client, n_runs, output_dir
        )
        
        # Ablation 4: Temperature variations
        for temp in [0.3, 0.7, 1.0]:
            temp_config = SimulationConfig(
                n_agents=n_agents,
                n_generations=n_generations,
                tasks_per_generation=10,
                evolution_type="directed",
                winner_takes_all=True,
                evolution_temperature=temp,
            )
            results[f"temp_{temp}"] = await run_ablation(
                f"temp_{temp}", temp_config, client, n_runs, output_dir
            )
        
        # Print comparison
        print("\n" + "=" * 60)
        print("ABLATION COMPARISON")
        print("=" * 60)
        print(f"\n{'Condition':<25} {'LSI Mean':>10} {'LSI Std':>10}")
        print("-" * 50)
        for name, result in results.items():
            lsi = result['final_lsi']
            print(f"{name:<25} {lsi['mean']:>10.3f} {lsi['std']:>10.3f}")
        
        # Save comparison
        with open(Path(output_dir) / "comparison.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{client.get_usage_report()}")
        
        return results
        
    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--runs", type=int, default=5, help="Runs per condition")
    parser.add_argument("--agents", type=int, default=16, help="Number of agents")
    parser.add_argument("--generations", type=int, default=50, help="Generations")
    parser.add_argument("--output", type=str, default="results/ablations")
    
    args = parser.parse_args()
    
    asyncio.run(run_all_ablations(
        n_runs=args.runs,
        n_agents=args.agents,
        n_generations=args.generations,
        output_dir=args.output
    ))


if __name__ == "__main__":
    main()

