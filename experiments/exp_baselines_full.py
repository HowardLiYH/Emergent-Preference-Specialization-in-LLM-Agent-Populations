"""
Baseline Comparison Experiments

Compares our method against:
1. NO_PROMPT: No system prompt (baseline LLM)
2. RANDOM_PROMPT: Random specialist prompt
3. WRONG_PROMPT: Intentionally wrong specialist
4. CORRECT_PROMPT: Correct specialist (our method)
"""

import sys
sys.path.insert(0, 'src')

import asyncio
import json
import random
from typing import Dict, List
from dataclasses import dataclass

from genesis.llm_client import LLMClient
from genesis.synthetic_rules import RuleType, generate_tasks
from genesis.preference_agent import create_handcrafted_specialist


@dataclass
class BaselineResult:
    """Result for a single baseline condition."""
    condition: str
    mean_accuracy: float
    std_accuracy: float
    per_rule: Dict[str, float]


async def test_condition(
    client: LLMClient,
    condition: str,
    n_tasks: int = 5,
    seed: int = 42
) -> BaselineResult:
    """Test a single baseline condition across all rules."""
    rules = list(RuleType)
    per_rule = {}
    all_scores = []

    for rule in rules:
        tasks = generate_tasks(rule, n=n_tasks, seed=seed, opaque=True)

        # Get prompt based on condition
        if condition == "NO_PROMPT":
            system_prompt = "You are a helpful assistant. Answer the question."
        elif condition == "RANDOM_PROMPT":
            random_rule = random.choice([r for r in rules if r != rule])
            system_prompt = create_handcrafted_specialist(random_rule).get_prompt()
        elif condition == "WRONG_PROMPT":
            # Systematically wrong: use next rule in sequence
            wrong_idx = (rules.index(rule) + 4) % len(rules)
            system_prompt = create_handcrafted_specialist(rules[wrong_idx]).get_prompt()
        elif condition == "CORRECT_PROMPT":
            system_prompt = create_handcrafted_specialist(rule).get_prompt()
        else:
            system_prompt = ""

        scores = []
        for task in tasks:
            try:
                resp = await asyncio.wait_for(
                    client.generate(task.prompt, system=system_prompt,
                                   temperature=0.1, max_tokens=100),
                    timeout=30
                )
                scores.append(task.evaluate(resp))
            except:
                pass

        if scores:
            rule_avg = sum(scores) / len(scores)
            per_rule[rule.value] = rule_avg
            all_scores.extend(scores)

    # Compute overall stats
    if all_scores:
        mean = sum(all_scores) / len(all_scores)
        variance = sum((x - mean) ** 2 for x in all_scores) / len(all_scores)
        std = variance ** 0.5
    else:
        mean, std = 0, 0

    return BaselineResult(
        condition=condition,
        mean_accuracy=mean,
        std_accuracy=std,
        per_rule=per_rule
    )


async def run_baseline_comparison(api_key: str, n_tasks: int = 5) -> List[BaselineResult]:
    """Run all baseline conditions."""
    client = LLMClient.for_gemini(api_key=api_key, model='gemini-2.0-flash')

    conditions = ["NO_PROMPT", "RANDOM_PROMPT", "WRONG_PROMPT", "CORRECT_PROMPT"]
    results = []

    print(f"\n{'='*60}")
    print("BASELINE COMPARISON EXPERIMENTS")
    print(f"{'='*60}\n")

    for i, condition in enumerate(conditions):
        print(f"[{i+1}/{len(conditions)}] Testing {condition}...", flush=True)
        result = await test_condition(client, condition, n_tasks)
        results.append(result)
        print(f"  Mean accuracy: {result.mean_accuracy:.3f} (std: {result.std_accuracy:.3f})", flush=True)

    await client.close()
    return results


def print_baseline_results(results: List[BaselineResult]):
    """Print formatted baseline comparison."""
    print(f"\n{'='*60}")
    print("BASELINE COMPARISON RESULTS")
    print(f"{'='*60}\n")

    print(f"{'Condition':<20} {'Mean Acc':<12} {'Std':<10} {'vs Correct':<15}")
    print("-" * 60)

    correct_mean = next((r.mean_accuracy for r in results if r.condition == "CORRECT_PROMPT"), 0)

    for result in results:
        diff = result.mean_accuracy - correct_mean
        diff_str = f"{diff:+.3f}" if result.condition != "CORRECT_PROMPT" else "-"
        print(f"{result.condition:<20} {result.mean_accuracy:<12.3f} {result.std_accuracy:<10.3f} {diff_str:<15}")

    print(f"\n--- Per-Rule Breakdown ---")
    print(f"{'Rule':<15}", end="")
    for result in results:
        print(f"{result.condition[:8]:<12}", end="")
    print()
    print("-" * 60)

    rules = list(RuleType)
    for rule in rules:
        print(f"{rule.value:<15}", end="")
        for result in results:
            score = result.per_rule.get(rule.value, 0)
            print(f"{score:<12.2f}", end="")
        print()

    # Statistical significance
    print(f"\n--- Key Findings ---")
    no_prompt = next((r for r in results if r.condition == "NO_PROMPT"), None)
    correct = next((r for r in results if r.condition == "CORRECT_PROMPT"), None)

    if no_prompt and correct:
        improvement = correct.mean_accuracy - no_prompt.mean_accuracy
        print(f"CORRECT vs NO_PROMPT: {improvement:+.3f} ({improvement/no_prompt.mean_accuracy*100:+.1f}%)")

    random_prompt = next((r for r in results if r.condition == "RANDOM_PROMPT"), None)
    if random_prompt and correct:
        improvement = correct.mean_accuracy - random_prompt.mean_accuracy
        print(f"CORRECT vs RANDOM: {improvement:+.3f} ({improvement/random_prompt.mean_accuracy*100:+.1f}%)")


async def main():
    """Run baseline comparison."""
    GEMINI_KEY = 'AIzaSyDngzJmPnKNrc-jXz5y5xUuDDlwhCWDRic'

    print("Baseline Comparison Experiments")
    print("Testing: NO_PROMPT, RANDOM_PROMPT, WRONG_PROMPT, CORRECT_PROMPT")

    results = await run_baseline_comparison(api_key=GEMINI_KEY, n_tasks=5)

    print_baseline_results(results)

    # Save results
    save_data = []
    for result in results:
        save_data.append({
            "condition": result.condition,
            "mean_accuracy": result.mean_accuracy,
            "std_accuracy": result.std_accuracy,
            "per_rule": result.per_rule
        })

    with open('results/baseline_comparison.json', 'w') as f:
        json.dump(save_data, f, indent=2)

    print("\nResults saved to results/baseline_comparison.json")


if __name__ == "__main__":
    asyncio.run(main())
