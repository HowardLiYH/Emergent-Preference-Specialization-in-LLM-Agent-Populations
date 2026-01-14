"""
Ablation experiment runner.

Tests which components are essential for specialization.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for an ablation experiment."""
    name: str
    description: str
    modify_config: Dict[str, Any]


# Ablation experiments to run
ABLATION_EXPERIMENTS = [
    AblationConfig(
        name='no_fitness_sharing',
        description='Remove fitness sharing penalty',
        modify_config={'fitness_sharing_gamma': 0.0}
    ),
    AblationConfig(
        name='no_memory',
        description='Disable agent memory system',
        modify_config={'memory_enabled': False}
    ),
    AblationConfig(
        name='no_curriculum',
        description='Disable task curriculum',
        modify_config={'curriculum_enabled': False}
    ),
    AblationConfig(
        name='random_tools',
        description='Random tool selection (no Thompson Sampling)',
        modify_config={'tool_policy': 'random'}
    ),
    AblationConfig(
        name='uniform_regimes',
        description='Uniform regime distribution (no non-uniform rewards)',
        modify_config={'uniform_regimes': True}
    ),
]


def run_ablation(
    ablation: AblationConfig,
    base_runner: callable,
    n_runs: int = 10,
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run a single ablation experiment.

    Args:
        ablation: Ablation configuration
        base_runner: Function that runs experiment with config
        n_runs: Number of runs
        output_path: Path to save results

    Returns:
        Ablation results
    """
    logger.info(f"Running ablation: {ablation.name}")

    results = {
        'name': ablation.name,
        'description': ablation.description,
        'config_changes': ablation.modify_config,
        'runs': [],
    }

    for seed in range(n_runs):
        try:
            run_result = base_runner(
                seed=seed,
                **ablation.modify_config
            )
            results['runs'].append({
                'seed': seed,
                'success': True,
                **run_result
            })
        except Exception as e:
            logger.error(f"Ablation {ablation.name} seed {seed} failed: {e}")
            results['runs'].append({
                'seed': seed,
                'success': False,
                'error': str(e)
            })

    # Aggregate
    successful_runs = [r for r in results['runs'] if r.get('success')]
    if successful_runs:
        import numpy as np

        sci_values = [r.get('sci', 0) for r in successful_runs]
        results['mean_sci'] = float(np.mean(sci_values))
        results['std_sci'] = float(np.std(sci_values))

        coverage_values = [r.get('coverage', 0) for r in successful_runs]
        results['mean_coverage'] = float(np.mean(coverage_values))

    # Save if path provided
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / f'{ablation.name}.json', 'w') as f:
            json.dump(results, f, indent=2)

    return results


def run_all_ablations(
    base_runner: callable,
    output_path: Optional[Path] = None,
    n_runs: int = 10
) -> Dict[str, Dict[str, Any]]:
    """
    Run all ablation experiments.

    Args:
        base_runner: Base experiment runner
        output_path: Path for results
        n_runs: Runs per ablation

    Returns:
        All ablation results
    """
    all_results = {}

    # First run baseline
    logger.info("Running baseline (full system)...")
    baseline_results = run_ablation(
        AblationConfig(
            name='baseline',
            description='Full system (no ablation)',
            modify_config={}
        ),
        base_runner,
        n_runs,
        output_path
    )
    all_results['baseline'] = baseline_results

    # Run each ablation
    for ablation in ABLATION_EXPERIMENTS:
        results = run_ablation(
            ablation,
            base_runner,
            n_runs,
            output_path
        )
        all_results[ablation.name] = results

    # Save summary
    if output_path:
        with open(output_path / 'ablation_summary.json', 'w') as f:
            summary = {
                name: {
                    'mean_sci': r.get('mean_sci', 0),
                    'std_sci': r.get('std_sci', 0),
                    'mean_coverage': r.get('mean_coverage', 0),
                }
                for name, r in all_results.items()
            }
            json.dump(summary, f, indent=2)

    return all_results


def generate_ablation_report(results: Dict[str, Dict[str, Any]]) -> str:
    """Generate ablation report."""
    lines = [
        "# Ablation Study Results",
        "",
        "| Ablation | Mean SCI | Std SCI | Coverage | Essential? |",
        "|----------|----------|---------|----------|------------|",
    ]

    baseline_sci = results.get('baseline', {}).get('mean_sci', 0)

    for name, data in results.items():
        sci = data.get('mean_sci', 0)
        std = data.get('std_sci', 0)
        cov = data.get('mean_coverage', 0)

        # Component is essential if SCI drops significantly
        essential = "Yes" if (baseline_sci - sci) > 0.2 else "No"
        if name == 'baseline':
            essential = "N/A"

        lines.append(
            f"| {name} | {sci:.3f} | {std:.3f} | {cov:.1%} | {essential} |"
        )

    return "\n".join(lines)
