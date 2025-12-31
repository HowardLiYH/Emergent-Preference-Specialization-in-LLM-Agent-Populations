#!/usr/bin/env python3
"""
Experiment 6: Prompt Swap Counterfactual Test

The critical causal validation experiment.

Protocol:
1. Train a population until specialization emerges
2. Identify specialists (e.g., math specialist, language specialist)
3. Swap their prompts
4. Re-evaluate: Does performance follow the prompt or the agent?

If prompts CAUSE specialization:
- After swap, the "math specialist" (now with language prompt) 
  should become better at language tasks

Transfer Coefficient:
- 0: Performance follows agent identity (prompts don't matter)
- 1: Performance follows prompt completely (prompts cause behavior)
- Target: > 0.5 for causal claim
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from genesis import create_client, GenesisSimulation
from genesis.simulation import SimulationConfig
from genesis.counterfactual import run_prompt_swap_test
from genesis.tasks import TaskPool


async def run_counterfactual_experiment(
    n_pre_training_generations: int = 50,
    n_agents: int = 16,
    n_eval_tasks: int = 20,
    n_runs: int = 5,
    output_dir: str = "results/counterfactual"
):
    """
    Run the prompt swap counterfactual experiment.
    
    Steps:
    1. Train population to develop specialists
    2. Run prompt swap test
    3. Compute transfer coefficient
    """
    
    print("=" * 60)
    print("EXPERIMENT 6: PROMPT SWAP COUNTERFACTUAL")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Pre-training generations: {n_pre_training_generations}")
    print(f"  Agents: {n_agents}")
    print(f"  Eval tasks per type: {n_eval_tasks}")
    print(f"  Runs: {n_runs}")
    
    client = create_client()
    task_pool = TaskPool()
    
    all_results = []
    
    try:
        for run in range(n_runs):
            print(f"\n{'='*40}")
            print(f"RUN {run+1}/{n_runs}")
            print(f"{'='*40}")
            
            # Step 1: Train population
            print("\nPhase 1: Training population...")
            config = SimulationConfig(
                n_agents=n_agents,
                n_generations=n_pre_training_generations,
                tasks_per_generation=10,
                evolution_type="directed",
                winner_takes_all=True,
                seed=run,
            )
            
            sim = GenesisSimulation(config, client)
            train_result = await sim.run()
            
            final_lsi = train_result.final_metrics['lsi']['mean']
            print(f"Training complete. Final LSI: {final_lsi:.3f}")
            
            # Check if we have specialists
            from genesis.metrics import identify_specialists
            specialists = identify_specialists(train_result.final_agents, lsi_threshold=0.4)
            print(f"Specialists found: {list(specialists.keys())}")
            
            if len(specialists) < 2:
                print("Warning: Less than 2 specialist types emerged. Swap test may be limited.")
            
            # Step 2: Run prompt swap test
            print("\nPhase 2: Running prompt swap test...")
            swap_result = await run_prompt_swap_test(
                agents=train_result.final_agents,
                task_pool=task_pool,
                llm_client=client,
                n_eval_tasks=n_eval_tasks
            )
            
            print(swap_result.analysis)
            
            all_results.append({
                "run": run,
                "final_lsi": final_lsi,
                "transfer_coefficient": swap_result.transfer_coefficient,
                "baseline": swap_result.baseline,
                "swapped": swap_result.swapped,
            })
        
        # Aggregate results
        import numpy as np
        transfer_coeffs = [r["transfer_coefficient"] for r in all_results]
        
        print("\n" + "=" * 60)
        print("AGGREGATE RESULTS")
        print("=" * 60)
        print(f"\nTransfer Coefficient across {n_runs} runs:")
        print(f"  Mean: {np.mean(transfer_coeffs):.3f}")
        print(f"  Std:  {np.std(transfer_coeffs):.3f}")
        print(f"  Min:  {np.min(transfer_coeffs):.3f}")
        print(f"  Max:  {np.max(transfer_coeffs):.3f}")
        
        # Success criterion
        mean_tc = np.mean(transfer_coeffs)
        print("\n" + "-" * 40)
        print("CAUSAL CLAIM VALIDATION:")
        if mean_tc >= 0.75:
            print(f"  ✓ STRONG EVIDENCE: TC = {mean_tc:.3f} >= 0.75")
            print("    Prompts strongly CAUSE specialization.")
        elif mean_tc >= 0.5:
            print(f"  ✓ MODERATE EVIDENCE: TC = {mean_tc:.3f} >= 0.50")
            print("    Prompts influence specialization.")
        else:
            print(f"  ✗ WEAK EVIDENCE: TC = {mean_tc:.3f} < 0.50")
            print("    Cannot establish causal link between prompts and performance.")
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        summary = {
            "n_runs": n_runs,
            "config": {
                "n_pre_training_generations": n_pre_training_generations,
                "n_agents": n_agents,
                "n_eval_tasks": n_eval_tasks,
            },
            "transfer_coefficient": {
                "mean": float(np.mean(transfer_coeffs)),
                "std": float(np.std(transfer_coeffs)),
                "values": transfer_coeffs,
            },
            "runs": all_results,
        }
        
        with open(output_path / "results.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nResults saved to {output_path}")
        print(f"\n{client.get_usage_report()}")
        
        return summary
        
    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run prompt swap counterfactual experiment"
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--agents", type=int, default=16)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--eval-tasks", type=int, default=20)
    parser.add_argument("--output", type=str, default="results/counterfactual")
    
    args = parser.parse_args()
    
    asyncio.run(run_counterfactual_experiment(
        n_pre_training_generations=args.generations,
        n_agents=args.agents,
        n_eval_tasks=args.eval_tasks,
        n_runs=args.runs,
        output_dir=args.output
    ))


if __name__ == "__main__":
    main()

