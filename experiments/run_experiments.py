#!/usr/bin/env python3
"""
Run all Genesis experiments with proper rate limiting.

This script runs experiments sequentially with delays to avoid
overwhelming the API proxy.
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from genesis.llm_client import LLMClient
from genesis.config import DEFAULT_PROXY_CONFIG, LLMConfig
from genesis.tasks import TaskPool, TaskType
from genesis.agent import GenesisAgent, create_population
from genesis.metrics import compute_lsi
import numpy as np


# Use environment variables for API credentials
import os

EXPERIMENT_CONFIG = LLMConfig(
    provider=os.getenv("LLM_PROVIDER", "gemini"),
    api_key=os.getenv("OPENAI_API_KEY", os.getenv("GEMINI_API_KEY", "")),
    api_base=os.getenv("OPENAI_API_BASE", ""),
    model=os.getenv("LLM_MODEL", "gemini-2.0-flash"),
    max_retries=5,
    timeout=180,  # 3 minutes timeout
)


async def call_llm(client: LLMClient, prompt: str, system: str = None,
                   max_tokens: int = 200, delay: float = 0.5) -> str:
    """Make an LLM call with rate limiting."""
    await asyncio.sleep(delay)  # Rate limit
    return await client.generate(prompt, max_tokens=max_tokens, system=system)


async def evaluate_agent_on_task(client: LLMClient, agent: GenesisAgent,
                                  task, delay: float = 0.5) -> float:
    """Evaluate an agent on a single task."""
    prompt = f"{task.prompt}"
    system = agent.system_prompt

    try:
        response = await call_llm(client, prompt, system, max_tokens=200, delay=delay)
        score = task.evaluate(response)
        agent.record_performance(task.task_type.value, score)
        return score, response
    except Exception as e:
        print(f"    Error: {e}")
        return 0.0, ""


async def evolve_prompt(client: LLMClient, agent: GenesisAgent,
                        task_type: str, score: float, delay: float = 1.0) -> str:
    """Evolve an agent's prompt based on success."""
    evolution_prompt = f"""You are an AI agent evolution system.

This agent just SUCCEEDED at a {task_type} task with score {score:.2f}.

Current agent role description:
{agent.system_prompt}

Update the agent's role description to:
1. Reinforce the skills that led to this success
2. Become more specialized in {task_type} tasks
3. Keep the new prompt under 200 words

Output ONLY the new system prompt, nothing else."""

    try:
        new_prompt = await call_llm(client, evolution_prompt, max_tokens=300, delay=delay)
        return new_prompt.strip()
    except Exception as e:
        print(f"    Evolution error: {e}")
        return agent.system_prompt


async def run_baseline_experiment(n_agents=4, n_generations=10, tasks_per_gen=3, n_runs=2):
    """Run the baseline specialization experiment."""
    print("=" * 70)
    print("EXPERIMENT 1: BASELINE SPECIALIZATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Agents: {n_agents}")
    print(f"  Generations: {n_generations}")
    print(f"  Tasks/Generation: {tasks_per_gen}")
    print(f"  Runs: {n_runs}")
    print()

    all_results = []
    task_pool = TaskPool(seed=42)

    for run in range(n_runs):
        print(f"\n{'='*50}")
        print(f"RUN {run + 1}/{n_runs}")
        print(f"{'='*50}")

        client = LLMClient(EXPERIMENT_CONFIG)
        agents = create_population(n_agents)

        lsi_trajectory = []

        try:
            for gen in range(n_generations):
                tasks = task_pool.sample(tasks_per_gen)

                # Evaluate all agents on all tasks (sequentially to avoid timeouts)
                scores = {a.id: [] for a in agents}

                for task in tasks:
                    print(f"  Gen {gen}, Task: {task.task_type.value[:4]}...", end=" ")

                    best_score = -1
                    winner = None

                    for agent in agents:
                        score, _ = await evaluate_agent_on_task(client, agent, task, delay=0.3)
                        scores[agent.id].append(score)

                        if score > best_score:
                            best_score = score
                            winner = agent

                    print(f"Winner: {winner.id[:4]} (score={best_score:.2f})")

                    # Evolve winner's prompt
                    if winner and best_score > 0:
                        new_prompt = await evolve_prompt(
                            client, winner, task.task_type.value, best_score, delay=0.5
                        )
                        winner.system_prompt = new_prompt
                        winner.generation += 1

                # Compute LSI for this generation
                lsi_values = [compute_lsi(a) for a in agents]
                mean_lsi = np.mean(lsi_values)
                lsi_trajectory.append(mean_lsi)

                if gen % 3 == 0:
                    print(f"  --> Generation {gen}: Mean LSI = {mean_lsi:.3f}")

            # Final results for this run
            final_lsi = np.mean([compute_lsi(a) for a in agents])

            run_result = {
                'run': run,
                'final_lsi': float(final_lsi),
                'lsi_trajectory': [float(x) for x in lsi_trajectory],
                'agents': [
                    {
                        'id': a.id,
                        'best_type': a.get_best_task_type(),
                        'lsi': float(compute_lsi(a)),
                        'prompt_length': len(a.system_prompt),
                    }
                    for a in agents
                ],
                'tokens': {
                    'prompt': client.usage.prompt_tokens,
                    'completion': client.usage.completion_tokens,
                }
            }
            all_results.append(run_result)

            print(f"\n  Run {run + 1} complete: Final LSI = {final_lsi:.3f}")
            print(f"  {client.get_usage_report()}")

        except Exception as e:
            print(f"  Run error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await client.close()

    # Aggregate results
    if all_results:
        final_lsi_values = [r['final_lsi'] for r in all_results]

        print("\n" + "=" * 70)
        print("EXPERIMENT 1 RESULTS")
        print("=" * 70)
        print(f"Final LSI across {len(all_results)} runs:")
        print(f"  Mean: {np.mean(final_lsi_values):.3f}")
        print(f"  Std:  {np.std(final_lsi_values):.3f}")

        # Show agent specializations from last run
        print("\nAgent Specializations (Final Run):")
        for agent in all_results[-1]['agents']:
            print(f"  {agent['id'][:8]}: {agent['best_type']} (LSI={agent['lsi']:.3f})")

        # Calculate total cost
        total_prompt = sum(r['tokens']['prompt'] for r in all_results)
        total_completion = sum(r['tokens']['completion'] for r in all_results)
        est_cost = (total_prompt / 1_000_000 * 0.15) + (total_completion / 1_000_000 * 0.60)

        print(f"\nTotal tokens: {total_prompt:,} prompt + {total_completion:,} completion")
        print(f"Estimated cost: ${est_cost:.4f}")

        # Save results
        output_dir = Path(__file__).parent.parent / 'results' / 'baseline'
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            'experiment': 'baseline_specialization',
            'timestamp': datetime.now().isoformat(),
            'config': {
                'n_agents': n_agents,
                'n_generations': n_generations,
                'tasks_per_generation': tasks_per_gen,
                'n_runs': n_runs,
            },
            'results': {
                'final_lsi_mean': float(np.mean(final_lsi_values)),
                'final_lsi_std': float(np.std(final_lsi_values)),
                'final_lsi_values': final_lsi_values,
            },
            'runs': all_results,
            'cost': {
                'prompt_tokens': total_prompt,
                'completion_tokens': total_completion,
                'estimated_usd': est_cost,
            }
        }

        output_file = output_dir / 'experiment1_results.json'
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {output_file}")
        return summary

    return None


async def run_ablation_no_evolution(n_agents=4, n_generations=10, tasks_per_gen=3):
    """Ablation: No prompt evolution (baseline comparison)."""
    print("\n" + "=" * 70)
    print("ABLATION: NO EVOLUTION")
    print("=" * 70)

    client = LLMClient(EXPERIMENT_CONFIG)
    agents = create_population(n_agents)
    task_pool = TaskPool(seed=42)

    lsi_trajectory = []

    try:
        for gen in range(n_generations):
            tasks = task_pool.sample(tasks_per_gen)

            for task in tasks:
                for agent in agents:
                    score, _ = await evaluate_agent_on_task(client, agent, task, delay=0.3)
                # NO evolution - just evaluate

            lsi_values = [compute_lsi(a) for a in agents]
            mean_lsi = np.mean(lsi_values)
            lsi_trajectory.append(mean_lsi)

            if gen % 3 == 0:
                print(f"  Generation {gen}: Mean LSI = {mean_lsi:.3f}")

        final_lsi = np.mean([compute_lsi(a) for a in agents])
        print(f"\nFinal LSI (No Evolution): {final_lsi:.3f}")
        print(client.get_usage_report())

        return {
            'condition': 'no_evolution',
            'final_lsi': float(final_lsi),
            'lsi_trajectory': [float(x) for x in lsi_trajectory],
        }

    finally:
        await client.close()


async def run_ablation_all_evolve(n_agents=4, n_generations=10, tasks_per_gen=3):
    """Ablation: All agents evolve (not just winner)."""
    print("\n" + "=" * 70)
    print("ABLATION: ALL AGENTS EVOLVE")
    print("=" * 70)

    client = LLMClient(EXPERIMENT_CONFIG)
    agents = create_population(n_agents)
    task_pool = TaskPool(seed=42)

    lsi_trajectory = []

    try:
        for gen in range(n_generations):
            tasks = task_pool.sample(tasks_per_gen)

            for task in tasks:
                for agent in agents:
                    score, _ = await evaluate_agent_on_task(client, agent, task, delay=0.3)

                    # ALL agents evolve (not just winner)
                    if score > 0:
                        new_prompt = await evolve_prompt(
                            client, agent, task.task_type.value, score, delay=0.5
                        )
                        agent.system_prompt = new_prompt

            lsi_values = [compute_lsi(a) for a in agents]
            mean_lsi = np.mean(lsi_values)
            lsi_trajectory.append(mean_lsi)

            if gen % 3 == 0:
                print(f"  Generation {gen}: Mean LSI = {mean_lsi:.3f}")

        final_lsi = np.mean([compute_lsi(a) for a in agents])
        print(f"\nFinal LSI (All Evolve): {final_lsi:.3f}")
        print(client.get_usage_report())

        return {
            'condition': 'all_evolve',
            'final_lsi': float(final_lsi),
            'lsi_trajectory': [float(x) for x in lsi_trajectory],
        }

    finally:
        await client.close()


async def main():
    """Run all experiments."""
    print("=" * 70)
    print("GENESIS EXPERIMENT SUITE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # Experiment 1: Baseline (winner-take-all with evolution)
    baseline = await run_baseline_experiment(
        n_agents=4,
        n_generations=10,
        tasks_per_gen=3,
        n_runs=2
    )
    if baseline:
        results['baseline'] = baseline

    # Ablation 1: No evolution
    no_evo = await run_ablation_no_evolution(
        n_agents=4,
        n_generations=10,
        tasks_per_gen=3
    )
    if no_evo:
        results['no_evolution'] = no_evo

    # Ablation 2: All evolve
    all_evo = await run_ablation_all_evolve(
        n_agents=4,
        n_generations=10,
        tasks_per_gen=3
    )
    if all_evo:
        results['all_evolve'] = all_evo

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Condition':<20} {'Final LSI':>10}")
    print("-" * 35)

    if 'baseline' in results:
        print(f"{'Baseline (WTA)':<20} {results['baseline']['results']['final_lsi_mean']:>10.3f}")
    if 'no_evolution' in results:
        print(f"{'No Evolution':<20} {results['no_evolution']['final_lsi']:>10.3f}")
    if 'all_evolve' in results:
        print(f"{'All Evolve':<20} {results['all_evolve']['final_lsi']:>10.3f}")

    # Save all results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'all_experiments.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nAll results saved to {output_dir / 'all_experiments.json'}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    asyncio.run(main())
