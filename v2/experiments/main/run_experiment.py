"""
Main experiment runner for Emergent Prompt Evolution v2.

Runs the complete experiment pipeline:
1. Initialize population with tool levels
2. Run competition for n generations
3. Collect metrics and save results
4. Run baseline comparisons
5. Generate figures for paper
"""

import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.llm_client import LLMClient
from src.core.population import Population
from src.core.curriculum import TaskCurriculum
from src.regimes.sampler import RegimeSampler
from src.regimes.config import REGIMES, compute_equilibrium_distribution
from src.benchmark.runner import BenchmarkRunner
from src.benchmark.tasks import generate_l0_batch, generate_l1_batch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run v2 experiment')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--n-agents', type=int, default=12, help='Number of agents')
    parser.add_argument('--n-generations', type=int, default=100, help='Number of generations')
    parser.add_argument('--output-dir', type=str, default='v2/results', help='Output directory')
    parser.add_argument('--run-baselines', action='store_true', help='Run baseline comparisons')
    parser.add_argument('--run-benchmark', action='store_true', help='Run ESB benchmark')
    return parser.parse_args()


def create_task_generator(curriculum: TaskCurriculum):
    """Create a task generator function."""
    def generate_task(regime: str):
        level = curriculum.sample_task_level()

        if level == 'L0':
            tasks = generate_l0_batch(1)
            return tasks[0] if tasks else None
        elif level == 'L1':
            tasks = generate_l1_batch(1)
            return tasks[0] if tasks else None
        else:
            # Placeholder for other levels
            from src.benchmark.tasks.l0_tasks import Task
            return Task(
                question=f"Complete this {regime} task at level {level}.",
                answer="placeholder",
                regime=regime,
                tool_level=level
            )

    return generate_task


def create_evaluator():
    """Create an evaluator function."""
    def evaluate(response: str, task) -> tuple:
        if not response:
            return False, 0.0

        # Check if answer is in response
        answer_lower = task.answer.lower() if hasattr(task, 'answer') else ""
        response_lower = response.lower()

        is_correct = answer_lower in response_lower if answer_lower else len(response) > 10
        confidence = 0.8 if is_correct else 0.3

        return is_correct, confidence

    return evaluate


def run_experiment(
    seed: int = 0,
    n_agents: int = 12,
    n_generations: int = 100,
    output_dir: str = 'v2/results'
) -> Dict[str, Any]:
    """
    Run the main experiment.

    Args:
        seed: Random seed
        n_agents: Number of agents
        n_generations: Number of generations
        output_dir: Output directory

    Returns:
        Experiment results
    """
    logger.info(f"Starting experiment with seed={seed}, n_agents={n_agents}, n_gen={n_generations}")
    start_time = time.time()

    # Create output directory
    output_path = Path(output_dir) / f"experiment_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize LLM client
    try:
        llm_client = LLMClient()
        logger.info(f"LLM client initialized: {llm_client.model}")
    except Exception as e:
        logger.warning(f"LLM client init failed: {e}. Running in dry-run mode.")
        llm_client = None

    # Initialize population
    population = Population(
        n_agents=n_agents,
        seed=seed,
        llm_client=llm_client
    )

    # Initialize curriculum
    curriculum = TaskCurriculum(seed=seed)

    # Create task generator and evaluator
    task_generator = create_task_generator(curriculum)
    evaluator = create_evaluator()

    # Track metrics
    metrics = {
        'sci_history': [],
        'coverage_history': [],
        'generation_results': [],
    }

    # Run generations
    logger.info(f"Running {n_generations} generations...")

    for gen in range(n_generations):
        # Run one generation
        result = population.run_generation(
            task_generator=task_generator,
            evaluator=evaluator
        )

        # Record metrics
        stats = population.get_stats()
        metrics['sci_history'].append(stats['sci'])
        metrics['coverage_history'].append(stats['coverage'])
        metrics['generation_results'].append({
            'generation': gen + 1,
            'winner': result.winner_id,
            'regime': result.regime,
            'tool': result.winner_tool,
        })

        # Advance curriculum
        curriculum.advance()

        # Log progress
        if (gen + 1) % 10 == 0:
            logger.info(
                f"Gen {gen + 1}: SCI={stats['sci']:.3f}, "
                f"Coverage={stats['coverage']:.1%}"
            )

    # Final stats
    elapsed = time.time() - start_time
    final_stats = population.get_stats()

    # Compute theoretical equilibrium
    regime_config = {
        name: {
            'frequency': r.frequency,
            'reward': r.reward,
            'difficulty': r.difficulty
        }
        for name, r in REGIMES.items()
    }
    expected_distribution = compute_equilibrium_distribution(regime_config, n_agents)

    # Compile results
    results = {
        'seed': seed,
        'n_agents': n_agents,
        'n_generations': n_generations,
        'elapsed_seconds': elapsed,
        'final_sci': final_stats['sci'],
        'final_coverage': final_stats['coverage'],
        'specialist_distribution': final_stats['specialist_distribution'],
        'expected_distribution': expected_distribution,
        'llm_stats': llm_client.get_stats() if llm_client else {},
        'metrics': metrics,
    }

    # Save results
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save population state
    population.save(output_path / 'population.json')

    logger.info(f"Experiment completed in {elapsed:.1f}s")
    logger.info(f"Final SCI: {final_stats['sci']:.3f}, Coverage: {final_stats['coverage']:.1%}")
    logger.info(f"Results saved to {output_path}")

    return results


def run_multiple_seeds(
    n_runs: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """Run experiment with multiple seeds."""
    all_results = []

    for seed in range(n_runs):
        logger.info(f"\n{'='*60}\nRunning seed {seed}/{n_runs}\n{'='*60}")
        result = run_experiment(seed=seed, **kwargs)
        all_results.append(result)

    # Aggregate
    import numpy as np
    sci_values = [r['final_sci'] for r in all_results]
    coverage_values = [r['final_coverage'] for r in all_results]

    aggregate = {
        'n_runs': n_runs,
        'mean_sci': float(np.mean(sci_values)),
        'std_sci': float(np.std(sci_values)),
        'mean_coverage': float(np.mean(coverage_values)),
        'all_results': all_results,
    }

    # Save aggregate
    output_path = Path(kwargs.get('output_dir', 'v2/results'))
    with open(output_path / 'aggregate_results.json', 'w') as f:
        json.dump(aggregate, f, indent=2, default=str)

    return aggregate


if __name__ == '__main__':
    args = parse_args()

    result = run_experiment(
        seed=args.seed,
        n_agents=args.n_agents,
        n_generations=args.n_generations,
        output_dir=args.output_dir
    )

    print(f"\nFinal Results:")
    print(f"  SCI: {result['final_sci']:.3f}")
    print(f"  Coverage: {result['final_coverage']:.1%}")
    print(f"  Time: {result['elapsed_seconds']:.1f}s")
