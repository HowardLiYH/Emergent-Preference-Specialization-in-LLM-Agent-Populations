"""
Wall-clock time measurement for cost comparison.
"""

import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class WallClockResult:
    """Result of wall-clock measurement."""
    method: str
    time_seconds: float
    tokens_used: int
    final_accuracy: float
    reached_target: bool
    seed: int


def measure_wallclock(
    method_runner: Callable,
    method_name: str,
    target_perf: float = 0.80,
    n_runs: int = 5,
    seeds: Optional[list] = None
) -> Dict[str, Any]:
    """
    Measure wall-clock time for a method.

    Args:
        method_runner: Function that runs the method (seed) -> result_dict
        method_name: Name of the method
        target_perf: Target performance threshold
        n_runs: Number of runs
        seeds: Optional list of seeds

    Returns:
        Aggregated results
    """
    if seeds is None:
        seeds = list(range(n_runs))

    results = []

    for seed in seeds:
        start = time.time()

        try:
            result = method_runner(seed)
            elapsed = time.time() - start

            results.append(WallClockResult(
                method=method_name,
                time_seconds=elapsed,
                tokens_used=result.get('tokens', 0),
                final_accuracy=result.get('accuracy', 0),
                reached_target=result.get('reached_target', False),
                seed=seed
            ))

        except Exception as e:
            logger.error(f"Run {seed} failed: {e}")
            results.append(WallClockResult(
                method=method_name,
                time_seconds=float('inf'),
                tokens_used=0,
                final_accuracy=0,
                reached_target=False,
                seed=seed
            ))

    # Aggregate
    import numpy as np

    times = [r.time_seconds for r in results if r.reached_target]
    tokens = [r.tokens_used for r in results if r.reached_target]

    return {
        'method': method_name,
        'n_runs': n_runs,
        'success_rate': len(times) / n_runs,
        'mean_time': np.mean(times) if times else float('inf'),
        'std_time': np.std(times) if times else 0,
        'mean_tokens': np.mean(tokens) if tokens else 0,
        'std_tokens': np.std(tokens) if tokens else 0,
        'results': results,
    }


def compare_methods(
    method_runners: Dict[str, Callable],
    target_perf: float = 0.80,
    n_runs: int = 10
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple methods.

    Args:
        method_runners: Dict[method_name -> runner_function]
        target_perf: Target performance
        n_runs: Number of runs per method

    Returns:
        Comparison results
    """
    results = {}

    for name, runner in method_runners.items():
        logger.info(f"Measuring {name}...")
        results[name] = measure_wallclock(
            runner, name, target_perf, n_runs
        )

    return results


def generate_comparison_report(
    results: Dict[str, Dict[str, Any]]
) -> str:
    """Generate comparison report."""
    lines = [
        "# Wall-Clock Comparison Report",
        "",
        "| Method | Success Rate | Mean Time (s) | Mean Tokens | Std Tokens |",
        "|--------|--------------|---------------|-------------|------------|",
    ]

    for name, data in results.items():
        lines.append(
            f"| {name} | {data['success_rate']:.0%} | "
            f"{data['mean_time']:.1f} | "
            f"{data['mean_tokens']:.0f} | {data['std_tokens']:.0f} |"
        )

    return "\n".join(lines)
