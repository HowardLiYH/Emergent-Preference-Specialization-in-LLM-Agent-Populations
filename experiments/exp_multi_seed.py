"""
Multi-Seed Experiment Runner

Runs experiments with multiple seeds and aggregates results with:
- Mean and standard deviation
- 95% confidence intervals
- Error bars for visualization
"""

import sys
sys.path.insert(0, 'src')

import asyncio
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
import math

from genesis.llm_client import LLMClient
from genesis.synthetic_rules import RuleType, generate_tasks
from genesis.preference_agent import create_handcrafted_specialist


@dataclass
class MultiSeedResult:
    """Aggregated results across multiple seeds."""
    metric_name: str
    mean: float
    std: float
    ci_low: float
    ci_high: float
    n_seeds: int
    raw_values: List[float]


def compute_stats(values: List[float]) -> Tuple[float, float, float, float]:
    """Compute mean, std, and 95% CI."""
    if not values:
        return 0, 0, 0, 0

    n = len(values)
    mean = sum(values) / n

    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
        se = std / math.sqrt(n)
        ci_margin = 1.96 * se  # 95% CI
    else:
        std = 0
        ci_margin = 0

    return mean, std, mean - ci_margin, mean + ci_margin


async def run_single_seed_swap_test(
    client: LLMClient,
    seed: int,
    n_tasks: int = 5
) -> Dict[str, float]:
    """Run swap test with a specific seed."""
    results = {}
    rules = list(RuleType)

    for spec_rule in rules:
        spec = create_handcrafted_specialist(spec_rule)

        for test_rule in rules:
            if spec_rule == test_rule:
                continue

            # Generate tasks with this seed
            tasks = generate_tasks(test_rule, n=n_tasks, seed=seed, opaque=True)
            correct_spec = create_handcrafted_specialist(test_rule)

            correct_scores = []
            wrong_scores = []

            for task in tasks:
                try:
                    # Correct specialist
                    resp = await asyncio.wait_for(
                        client.generate(task.prompt, system=correct_spec.get_prompt(),
                                       temperature=0.1, max_tokens=100),
                        timeout=30
                    )
                    correct_scores.append(task.evaluate(resp))

                    # Wrong specialist
                    resp = await asyncio.wait_for(
                        client.generate(task.prompt, system=spec.get_prompt(),
                                       temperature=0.1, max_tokens=100),
                        timeout=30
                    )
                    wrong_scores.append(task.evaluate(resp))
                except:
                    pass

            if correct_scores and wrong_scores:
                correct_avg = sum(correct_scores) / len(correct_scores)
                wrong_avg = sum(wrong_scores) / len(wrong_scores)
                effect = correct_avg - wrong_avg
                passed = 1.0 if effect > 0.1 else 0.0

                key = f"{spec_rule.value}->{test_rule.value}"
                results[key] = passed

    return results


async def run_multi_seed_swap_test(
    api_key: str,
    n_seeds: int = 5,
    n_tasks: int = 3
) -> Dict[str, MultiSeedResult]:
    """Run swap test with multiple seeds and aggregate."""
    client = LLMClient.for_gemini(api_key=api_key, model='gemini-2.0-flash')

    print(f"\n{'='*60}")
    print(f"MULTI-SEED SWAP TEST ({n_seeds} seeds)")
    print(f"{'='*60}\n")

    all_results = []

    for seed in range(n_seeds):
        print(f"Seed {seed+1}/{n_seeds}...", flush=True)
        seed_results = await run_single_seed_swap_test(client, seed=seed*100, n_tasks=n_tasks)
        all_results.append(seed_results)

        # Quick summary
        pass_rate = sum(seed_results.values()) / len(seed_results) if seed_results else 0
        print(f"  Pass rate: {pass_rate:.1%}", flush=True)

    await client.close()

    # Aggregate across seeds
    aggregated = {}

    # Overall pass rate
    pass_rates = []
    for seed_result in all_results:
        if seed_result:
            pass_rates.append(sum(seed_result.values()) / len(seed_result))

    mean, std, ci_low, ci_high = compute_stats(pass_rates)
    aggregated["overall_pass_rate"] = MultiSeedResult(
        metric_name="Overall Pass Rate",
        mean=mean, std=std, ci_low=ci_low, ci_high=ci_high,
        n_seeds=n_seeds, raw_values=pass_rates
    )

    # Per-rule pass rates
    rules = list(RuleType)
    for rule in rules:
        rule_pass_rates = []
        for seed_result in all_results:
            rule_pairs = [v for k, v in seed_result.items() if k.startswith(f"{rule.value}->")]
            if rule_pairs:
                rule_pass_rates.append(sum(rule_pairs) / len(rule_pairs))

        if rule_pass_rates:
            mean, std, ci_low, ci_high = compute_stats(rule_pass_rates)
            aggregated[f"{rule.value}_pass_rate"] = MultiSeedResult(
                metric_name=f"{rule.value.upper()} Pass Rate",
                mean=mean, std=std, ci_low=ci_low, ci_high=ci_high,
                n_seeds=n_seeds, raw_values=rule_pass_rates
            )

    return aggregated


def print_multi_seed_results(results: Dict[str, MultiSeedResult]):
    """Print formatted multi-seed results."""
    print(f"\n{'='*60}")
    print("MULTI-SEED RESULTS WITH CONFIDENCE INTERVALS")
    print(f"{'='*60}\n")

    print(f"{'Metric':<25} {'Mean':<8} {'Std':<8} {'95% CI':<20}")
    print("-" * 60)

    for key, result in results.items():
        ci_str = f"[{result.ci_low:.3f}, {result.ci_high:.3f}]"
        print(f"{result.metric_name:<25} {result.mean:<8.3f} {result.std:<8.3f} {ci_str:<20}")


async def main():
    """Run multi-seed experiments."""
    GEMINI_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_KEY:
        raise RuntimeError('GEMINI_API_KEY environment variable not set')

    print("Multi-Seed Swap Test")
    print("Running Phase 2 with 5 seeds for statistical rigor")

    results = await run_multi_seed_swap_test(
        api_key=GEMINI_KEY,
        n_seeds=5,
        n_tasks=3  # Fewer tasks per seed to save cost
    )

    print_multi_seed_results(results)

    # Save results
    save_data = {}
    for key, result in results.items():
        save_data[key] = {
            "metric_name": result.metric_name,
            "mean": result.mean,
            "std": result.std,
            "ci_low": result.ci_low,
            "ci_high": result.ci_high,
            "n_seeds": result.n_seeds,
            "raw_values": result.raw_values
        }

    with open('results/multi_seed_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)

    print("\nResults saved to results/multi_seed_results.json")


if __name__ == "__main__":
    asyncio.run(main())
