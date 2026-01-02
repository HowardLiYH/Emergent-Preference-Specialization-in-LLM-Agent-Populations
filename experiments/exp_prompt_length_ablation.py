"""
Prompt Length Ablation Study

Compares performance of:
- SHORT prompts (~30 chars): Minimal instruction
- ENHANCED prompts (~800+ chars): Full persona, steps, examples

Tests hypothesis: Longer, more detailed prompts lead to better specialization.
"""

import sys
sys.path.insert(0, 'src')

import asyncio
import json
from typing import Dict, List
from dataclasses import dataclass

from genesis.llm_client import LLMClient
from genesis.synthetic_rules import RuleType, generate_tasks
from genesis.rule_strategies import STRATEGY_LEVELS, SHORT_STRATEGY_LEVELS


@dataclass
class AblationResult:
    """Result for a single rule with both prompt types."""
    rule: str
    short_prompt_score: float
    enhanced_prompt_score: float
    improvement: float  # enhanced - short
    short_prompt_len: int
    enhanced_prompt_len: int


async def test_prompt_type(
    client: LLMClient,
    rule_type: RuleType,
    use_short: bool,
    n_tasks: int = 10
) -> float:
    """Test a single rule with either short or enhanced prompts."""
    # Get the Level 3 prompt
    if use_short:
        prompt = SHORT_STRATEGY_LEVELS[rule_type].level_3
    else:
        prompt = STRATEGY_LEVELS[rule_type].level_3

    # Generate opaque tasks (no hints)
    tasks = generate_tasks(rule_type, n=n_tasks, seed=42, opaque=True)

    scores = []
    for task in tasks:
        try:
            resp = await client.generate(
                task.prompt,
                system=prompt,
                temperature=0.1,
                max_tokens=50
            )
            scores.append(task.evaluate(resp))
        except Exception as e:
            print(f"    Error: {e}")
            continue

    return sum(scores) / len(scores) if scores else 0


async def run_ablation(api_key: str, n_tasks: int = 10) -> List[AblationResult]:
    """Run ablation study comparing short vs enhanced prompts."""
    client = LLMClient.for_gemini(api_key=api_key, model='gemini-2.0-flash')

    results = []
    rules = list(RuleType)

    print("\n" + "="*60)
    print("PROMPT LENGTH ABLATION STUDY")
    print("Comparing SHORT (~30 chars) vs ENHANCED (~800+ chars) prompts")
    print("="*60 + "\n")

    for i, rule in enumerate(rules):
        print(f"[{i+1}/{len(rules)}] Testing {rule.value}...", flush=True)

        # Get prompt lengths
        short_len = len(SHORT_STRATEGY_LEVELS[rule].level_3)
        enhanced_len = len(STRATEGY_LEVELS[rule].level_3)

        # Test short prompt
        short_score = await test_prompt_type(client, rule, use_short=True, n_tasks=n_tasks)

        # Test enhanced prompt
        enhanced_score = await test_prompt_type(client, rule, use_short=False, n_tasks=n_tasks)

        improvement = enhanced_score - short_score

        result = AblationResult(
            rule=rule.value,
            short_prompt_score=short_score,
            enhanced_prompt_score=enhanced_score,
            improvement=improvement,
            short_prompt_len=short_len,
            enhanced_prompt_len=enhanced_len
        )
        results.append(result)

        mark = "↑" if improvement > 0.1 else ("↓" if improvement < -0.1 else "=")
        print(f"    Short ({short_len}c): {short_score:.2f}, Enhanced ({enhanced_len}c): {enhanced_score:.2f}, Δ: {improvement:+.2f} {mark}", flush=True)

    await client.close()
    return results


def analyze_ablation(results: List[AblationResult]):
    """Analyze ablation study results."""
    print("\n" + "="*60)
    print("ABLATION ANALYSIS")
    print("="*60 + "\n")

    # Overall stats
    avg_short = sum(r.short_prompt_score for r in results) / len(results)
    avg_enhanced = sum(r.enhanced_prompt_score for r in results) / len(results)
    avg_improvement = sum(r.improvement for r in results) / len(results)

    print(f"Average Short Prompt Score: {avg_short:.3f}")
    print(f"Average Enhanced Prompt Score: {avg_enhanced:.3f}")
    print(f"Average Improvement: {avg_improvement:+.3f}")

    # Per-rule breakdown
    print("\n--- Per-Rule Results ---")
    print(f"{'Rule':<12} {'Short':<8} {'Enhanced':<10} {'Δ':<8} {'Verdict'}")
    print("-" * 50)

    improved = 0
    same = 0
    worse = 0

    for r in results:
        if r.improvement > 0.1:
            verdict = "IMPROVED"
            improved += 1
        elif r.improvement < -0.1:
            verdict = "WORSE"
            worse += 1
        else:
            verdict = "SAME"
            same += 1

        print(f"{r.rule:<12} {r.short_prompt_score:<8.2f} {r.enhanced_prompt_score:<10.2f} {r.improvement:+<8.2f} {verdict}")

    print("\n--- Summary ---")
    print(f"Improved with enhanced prompts: {improved}/{len(results)}")
    print(f"Same performance: {same}/{len(results)}")
    print(f"Worse with enhanced prompts: {worse}/{len(results)}")

    # Correlation between prompt length and improvement
    print("\n--- Prompt Length Stats ---")
    avg_short_len = sum(r.short_prompt_len for r in results) / len(results)
    avg_enhanced_len = sum(r.enhanced_prompt_len for r in results) / len(results)
    print(f"Average Short Prompt Length: {avg_short_len:.0f} chars")
    print(f"Average Enhanced Prompt Length: {avg_enhanced_len:.0f} chars")
    print(f"Length Multiplier: {avg_enhanced_len/avg_short_len:.1f}x")

    # Conclusion
    print("\n--- CONCLUSION ---")
    if avg_improvement > 0.15:
        print("✓ ENHANCED PROMPTS HELP: Longer prompts improve specialization")
    elif avg_improvement > 0.05:
        print("~ MODEST BENEFIT: Some improvement from detailed prompts")
    elif avg_improvement > -0.05:
        print("= NO CLEAR DIFFERENCE: Prompt length may not matter")
    else:
        print("✗ SHORT PROMPTS BETTER: Concise instructions may be clearer")


async def main():
    """Run prompt length ablation study."""
    GEMINI_KEY = 'AIzaSyDngzJmPnKNrc-jXz5y5xUuDDlwhCWDRic'

    print("Prompt Length Ablation Study")
    print("Testing: SHORT (~30 chars) vs ENHANCED (~800+ chars)")
    print("="*50)

    results = await run_ablation(api_key=GEMINI_KEY, n_tasks=10)

    analyze_ablation(results)

    # Save results
    with open('results/prompt_length_ablation.json', 'w') as f:
        json.dump([{
            'rule': r.rule,
            'short_prompt_score': r.short_prompt_score,
            'enhanced_prompt_score': r.enhanced_prompt_score,
            'improvement': r.improvement,
            'short_prompt_len': r.short_prompt_len,
            'enhanced_prompt_len': r.enhanced_prompt_len
        } for r in results], f, indent=2)

    print("\nResults saved to results/prompt_length_ablation.json")


if __name__ == "__main__":
    asyncio.run(main())
