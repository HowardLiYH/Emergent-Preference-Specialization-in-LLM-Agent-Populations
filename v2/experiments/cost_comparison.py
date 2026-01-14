"""
Cost comparison experiment runner.

Compares population approach against baselines:
- UCB1 Bandit
- LinUCB (contextual bandit)
- Simplified PPO
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging
import numpy as np

from .baselines.bandit import UCB1Bandit
from .baselines.contextual_bandit import LinUCBBandit
from .baselines.ppo import SimplePPO
from .wallclock import measure_wallclock, compare_methods

logger = logging.getLogger(__name__)


def create_population_runner(llm_client, task_generator, evaluator):
    """Create a runner for the population method."""
    def run(seed: int, **kwargs) -> Dict[str, Any]:
        from ..src.core.population import Population

        pop = Population(n_agents=12, seed=seed, llm_client=llm_client)

        tokens = 0
        for _ in range(100):  # 100 generations
            result = pop.run_generation(task_generator, evaluator)
            tokens += 100  # Rough estimate per generation

            if pop.get_stats()['sci'] > 0.75:
                break

        return {
            'tokens': tokens,
            'accuracy': pop.get_stats()['sci'],
            'reached_target': pop.get_stats()['sci'] > 0.75,
        }

    return run


def create_bandit_runner(llm_client, task_generator, evaluator):
    """Create a runner for UCB1 bandit."""
    def run(seed: int, **kwargs) -> Dict[str, Any]:
        bandit = UCB1Bandit(n_arms=5)
        result = bandit.train_to_target(
            task_generator, evaluator, llm_client,
            target_accuracy=0.80, max_iterations=500
        )
        return result

    return run


def create_contextual_bandit_runner(llm_client, task_generator, evaluator):
    """Create a runner for contextual bandit."""
    def run(seed: int, **kwargs) -> Dict[str, Any]:
        bandit = LinUCBBandit(n_arms=5, context_dim=10)
        result = bandit.train_to_target(
            task_generator, evaluator, llm_client,
            target_accuracy=0.80, max_iterations=500
        )
        return result

    return run


def create_ppo_runner(llm_client, task_generator, evaluator):
    """Create a runner for simplified PPO."""
    def run(seed: int, **kwargs) -> Dict[str, Any]:
        ppo = SimplePPO(n_actions=5)
        result = ppo.train_to_target(
            task_generator, evaluator, llm_client,
            target_accuracy=0.80, max_iterations=500
        )
        return result

    return run


def run_cost_comparison(
    llm_client,
    task_generator,
    evaluator,
    n_runs: int = 10,
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run full cost comparison experiment.

    Args:
        llm_client: LLM client
        task_generator: Task generation function
        evaluator: Evaluation function
        n_runs: Number of runs per method
        output_path: Path to save results

    Returns:
        Comparison results
    """
    logger.info(f"Running cost comparison with {n_runs} runs per method...")

    method_runners = {
        'population': create_population_runner(llm_client, task_generator, evaluator),
        'ucb1_bandit': create_bandit_runner(llm_client, task_generator, evaluator),
        'contextual_bandit': create_contextual_bandit_runner(llm_client, task_generator, evaluator),
        'ppo': create_ppo_runner(llm_client, task_generator, evaluator),
    }

    results = compare_methods(method_runners, target_perf=0.80, n_runs=n_runs)

    # Calculate savings
    if 'ucb1_bandit' in results and 'population' in results:
        bandit_tokens = results['ucb1_bandit']['mean_tokens']
        pop_tokens = results['population']['mean_tokens']
        if bandit_tokens > 0:
            savings = (bandit_tokens - pop_tokens) / bandit_tokens * 100
            results['savings_vs_bandit'] = savings

    # Save results
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / 'cost_comparison.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

    return results


def run_statistical_analysis(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run statistical analysis on comparison results.

    Returns:
        Statistical analysis including t-tests and effect sizes
    """
    from scipy import stats

    analysis = {}

    pop_results = results.get('population', {}).get('results', [])
    pop_tokens = [r.tokens_used for r in pop_results if r.reached_target]

    for method in ['ucb1_bandit', 'contextual_bandit', 'ppo']:
        method_results = results.get(method, {}).get('results', [])
        method_tokens = [r.tokens_used for r in method_results if r.reached_target]

        if len(pop_tokens) >= 2 and len(method_tokens) >= 2:
            # T-test
            t_stat, p_value = stats.ttest_ind(pop_tokens, method_tokens)

            # Cohen's d
            pooled_std = np.sqrt(
                (np.var(pop_tokens) + np.var(method_tokens)) / 2
            )
            cohens_d = (np.mean(method_tokens) - np.mean(pop_tokens)) / pooled_std if pooled_std > 0 else 0

            analysis[method] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'cohens_d': float(cohens_d),
                'significant': p_value < 0.05,
                'effect_size': (
                    'huge' if abs(cohens_d) > 1.2 else
                    'large' if abs(cohens_d) > 0.8 else
                    'medium' if abs(cohens_d) > 0.5 else
                    'small'
                ),
            }

    return analysis


def generate_cost_report(
    results: Dict[str, Any],
    analysis: Optional[Dict[str, Any]] = None
) -> str:
    """Generate cost comparison report."""
    lines = [
        "# Cost Comparison Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Token Usage to Reach 80% Accuracy",
        "",
        "| Method | Mean Tokens | Std | Success Rate |",
        "|--------|-------------|-----|--------------|",
    ]

    for method, data in results.items():
        if isinstance(data, dict) and 'mean_tokens' in data:
            lines.append(
                f"| {method} | {data['mean_tokens']:.0f} | "
                f"{data.get('std_tokens', 0):.0f} | "
                f"{data.get('success_rate', 0):.0%} |"
            )

    if 'savings_vs_bandit' in results:
        lines.extend([
            "",
            f"**Population saves {results['savings_vs_bandit']:.0f}% tokens vs UCB1 bandit.**",
        ])

    if analysis:
        lines.extend([
            "",
            "## Statistical Significance",
            "",
            "| Comparison | p-value | Cohen's d | Effect |",
            "|------------|---------|-----------|--------|",
        ])

        for method, stats_data in analysis.items():
            lines.append(
                f"| Pop vs {method} | {stats_data['p_value']:.4f} | "
                f"{stats_data['cohens_d']:.2f} | {stats_data['effect_size']} |"
            )

    return "\n".join(lines)
