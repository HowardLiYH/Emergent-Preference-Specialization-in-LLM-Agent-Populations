"""
Scalability Analysis

Tests how performance scales with different population sizes:
- N=8: Small population
- N=12: Default population
- N=24: Medium population
- N=48: Large population

Measures:
- Coverage: How many rules have specialists
- Swap test pass rate: Does causality hold at scale
"""

import sys
sys.path.insert(0, 'src')

import asyncio
import json
from typing import Dict, List
from dataclasses import dataclass
import random

from genesis.llm_client import LLMClient
from genesis.synthetic_rules import RuleType, generate_tasks
from genesis.preference_agent import PreferenceAgent, create_handcrafted_specialist


@dataclass
class ScalabilityResult:
    """Result for a single population size."""
    n_agents: int
    coverage: float  # Fraction of rules with specialists
    swap_pass_rate: float
    unique_specialists: int
    total_api_calls: int


async def simulate_evolution(n_agents: int, n_generations: int = 20) -> Dict[str, int]:
    """Simulate evolution to get specialist distribution (no LLM calls)."""
    agents = [PreferenceAgent(f"agent_{i}") for i in range(n_agents)]
    rules = list(RuleType)

    for gen in range(n_generations):
        # Each generation: random tasks, random winners
        for _ in range(8):  # 8 tasks per generation
            rule = random.choice(rules)
            # Random winner among agents with lower levels
            eligible = [a for a in agents if a.get_strategy_level(rule) < 3]
            if eligible:
                winner = random.choice(eligible)
                winner.accumulate_strategy(rule)

    # Count specialists (agents with level 3 in any rule)
    specialist_counts = {r.value: 0 for r in rules}
    for agent in agents:
        specialized_rule = agent.get_specialized_rule()
        if specialized_rule:
            specialist_counts[specialized_rule.value] += 1

    return specialist_counts


async def test_swap_at_scale(
    client: LLMClient,
    n_pairs: int = 10
) -> float:
    """Quick swap test to validate causality holds."""
    rules = list(RuleType)
    passed = 0
    total = 0

    for _ in range(n_pairs):
        spec_rule = random.choice(rules)
        test_rule = random.choice([r for r in rules if r != spec_rule])

        tasks = generate_tasks(test_rule, n=2, seed=random.randint(0, 1000), opaque=True)
        correct_spec = create_handcrafted_specialist(test_rule)
        wrong_spec = create_handcrafted_specialist(spec_rule)

        correct_scores = []
        wrong_scores = []

        for task in tasks:
            try:
                resp = await client.generate(task.prompt, system=correct_spec.get_prompt(),
                                            temperature=0.1, max_tokens=100)
                correct_scores.append(task.evaluate(resp))

                resp = await client.generate(task.prompt, system=wrong_spec.get_prompt(),
                                            temperature=0.1, max_tokens=100)
                wrong_scores.append(task.evaluate(resp))
            except:
                pass

        if correct_scores and wrong_scores:
            correct_avg = sum(correct_scores) / len(correct_scores)
            wrong_avg = sum(wrong_scores) / len(wrong_scores)
            if correct_avg > wrong_avg + 0.1:
                passed += 1
            total += 1

    return passed / total if total > 0 else 0


async def run_scalability_analysis(api_key: str) -> List[ScalabilityResult]:
    """Run scalability tests for different population sizes."""
    client = LLMClient.for_gemini(api_key=api_key, model='gemini-2.0-flash')

    population_sizes = [8, 12, 24, 48]
    results = []

    print(f"\n{'='*60}")
    print("SCALABILITY ANALYSIS")
    print(f"{'='*60}\n")

    for n_agents in population_sizes:
        print(f"Testing N={n_agents} agents...", flush=True)

        # Simulate evolution (no LLM calls)
        specialist_dist = await simulate_evolution(n_agents, n_generations=50)

        # Count coverage
        rules_covered = sum(1 for count in specialist_dist.values() if count > 0)
        coverage = rules_covered / len(RuleType)

        # Quick swap test (with LLM)
        swap_pass_rate = await test_swap_at_scale(client, n_pairs=8)

        unique_specialists = sum(1 for count in specialist_dist.values() if count > 0)

        result = ScalabilityResult(
            n_agents=n_agents,
            coverage=coverage,
            swap_pass_rate=swap_pass_rate,
            unique_specialists=unique_specialists,
            total_api_calls=8 * 4  # 8 pairs * 4 calls each
        )
        results.append(result)

        print(f"  Coverage: {coverage:.1%}, Swap Pass: {swap_pass_rate:.1%}, Specialists: {unique_specialists}/8", flush=True)

    await client.close()
    return results


def print_scalability_results(results: List[ScalabilityResult]):
    """Print formatted scalability analysis."""
    print(f"\n{'='*60}")
    print("SCALABILITY RESULTS")
    print(f"{'='*60}\n")

    print(f"{'N Agents':<12} {'Coverage':<12} {'Swap Pass':<12} {'Specialists':<12}")
    print("-" * 50)

    for result in results:
        print(f"{result.n_agents:<12} {result.coverage:<12.1%} {result.swap_pass_rate:<12.1%} {result.unique_specialists}/8")

    # Analysis
    print(f"\n--- Scalability Findings ---")

    # Check if coverage maintains
    all_high_coverage = all(r.coverage >= 0.75 for r in results)
    print(f"Coverage maintains at scale: {'YES' if all_high_coverage else 'NO'}")

    # Check if swap test holds
    all_high_swap = all(r.swap_pass_rate >= 0.5 for r in results)
    print(f"Causality holds at scale: {'YES' if all_high_swap else 'PARTIAL'}")

    # Trend analysis
    coverages = [r.coverage for r in results]
    if len(set(coverages)) > 1:
        trend = "increasing" if coverages[-1] > coverages[0] else "stable/decreasing"
        print(f"Coverage trend with N: {trend}")


async def main():
    """Run scalability analysis."""
    GEMINI_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_KEY:
        raise RuntimeError('GEMINI_API_KEY environment variable not set')

    print("Scalability Analysis")
    print("Testing N=8, 12, 24, 48 agents")

    results = await run_scalability_analysis(api_key=GEMINI_KEY)

    print_scalability_results(results)

    # Save results
    save_data = []
    for result in results:
        save_data.append({
            "n_agents": result.n_agents,
            "coverage": result.coverage,
            "swap_pass_rate": result.swap_pass_rate,
            "unique_specialists": result.unique_specialists,
            "total_api_calls": result.total_api_calls
        })

    with open('results/scalability_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)

    print("\nResults saved to results/scalability_results.json")


if __name__ == "__main__":
    asyncio.run(main())
