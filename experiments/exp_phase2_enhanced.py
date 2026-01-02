"""
Phase 2: Enhanced Prompt Swap Tests

This experiment validates causality using enhanced 500+ char prompts:
- Tests all 8 rules × 8 rules = 64 pairs (28 unique swaps)
- Compares performance with original vs. swapped prompts
- Uses opaque task prompts (no rule hints)
"""

import sys
sys.path.insert(0, 'src')

import asyncio
from typing import Dict, List, Tuple
from dataclasses import dataclass

from genesis.llm_client import LLMClient
from genesis.synthetic_rules import RuleType, generate_tasks
from genesis.preference_agent import create_handcrafted_specialist


@dataclass
class SwapTestResult:
    """Result of a bidirectional swap test."""
    specialist_rule: str
    test_rule: str
    original_score: float
    swapped_score: float
    swap_effect: float  # original - swapped
    passed: bool  # True if original > swapped by margin


async def run_single_test(
    client: LLMClient,
    specialist_rule: RuleType,
    test_rule: RuleType,
    n_tasks: int = 5,
    use_opaque: bool = True
) -> Tuple[float, float]:
    """
    Run a single specialist on a test rule with both original and swapped prompts.

    Returns:
        (original_score, swapped_score)
    """
    # Create specialists
    original_specialist = create_handcrafted_specialist(specialist_rule)
    swapped_specialist = create_handcrafted_specialist(test_rule)

    # Generate opaque tasks
    tasks = generate_tasks(test_rule, n=n_tasks, seed=42, opaque=use_opaque)

    original_scores = []
    swapped_scores = []

    for task in tasks:
        try:
            # Test with original prompt
            resp1 = await client.generate(
                task.prompt,
                system=original_specialist.get_prompt(),
                temperature=0.1,
                max_tokens=50
            )
            original_scores.append(task.evaluate(resp1))

            # Test with swapped prompt
            resp2 = await client.generate(
                task.prompt,
                system=swapped_specialist.get_prompt(),
                temperature=0.1,
                max_tokens=50
            )
            swapped_scores.append(task.evaluate(resp2))

        except Exception as e:
            print(f"    Error: {e}")
            continue

    original_avg = sum(original_scores) / len(original_scores) if original_scores else 0
    swapped_avg = sum(swapped_scores) / len(swapped_scores) if swapped_scores else 0

    return original_avg, swapped_avg


async def run_full_swap_test(
    api_key: str,
    n_tasks: int = 5,
    timeout_seconds: int = 600
) -> Dict[str, SwapTestResult]:
    """
    Run bidirectional swap test for all rule pairs.

    Tests: For each (specialist_rule, test_rule) pair:
    - Original: specialist_rule prompt on test_rule tasks
    - Swapped: test_rule prompt on test_rule tasks

    If original > swapped for specialist on THEIR OWN rule, prompts are causal.
    """
    client = LLMClient.for_gemini(api_key=api_key, model='gemini-2.0-flash')

    results = {}
    rules = list(RuleType)

    total_pairs = len(rules) * (len(rules) - 1)  # 8 × 7 = 56 pairs
    current = 0

    print(f"\n{'='*60}")
    print("PHASE 2: ENHANCED PROMPT SWAP TEST")
    print(f"Testing {total_pairs} specialist-task pairs with enhanced prompts")
    print(f"{'='*60}\n")

    for specialist_rule in rules:
        for test_rule in rules:
            if specialist_rule == test_rule:
                continue  # Skip same-rule (would be trivially 0 effect)

            current += 1
            print(f"[{current}/{total_pairs}] {specialist_rule.value} specialist → {test_rule.value} tasks...", flush=True)

            try:
                original, swapped = await asyncio.wait_for(
                    run_single_test(
                        client, specialist_rule, test_rule, n_tasks, use_opaque=True
                    ),
                    timeout=60  # 60s per pair
                )

                swap_effect = original - swapped

                # Pass if specialist does worse on wrong rule than the correct specialist
                # i.e., swapped > original (the correct specialist beats the wrong one)
                passed = swapped > original + 0.1  # 10% margin

                result = SwapTestResult(
                    specialist_rule=specialist_rule.value,
                    test_rule=test_rule.value,
                    original_score=original,
                    swapped_score=swapped,
                    swap_effect=swap_effect,
                    passed=passed
                )
                results[f"{specialist_rule.value}→{test_rule.value}"] = result

                mark = "✓" if passed else "✗"
                print(f"    Original: {original:.2f}, Swapped: {swapped:.2f}, Effect: {swap_effect:+.2f} {mark}", flush=True)

            except asyncio.TimeoutError:
                print(f"    TIMEOUT", flush=True)
            except Exception as e:
                print(f"    ERROR: {e}", flush=True)

    await client.close()
    return results


def analyze_results(results: Dict[str, SwapTestResult]):
    """Analyze and summarize swap test results."""
    print("\n" + "="*60)
    print("SWAP TEST ANALYSIS")
    print("="*60 + "\n")

    if not results:
        print("No results to analyze!")
        return

    passed = [r for r in results.values() if r.passed]
    failed = [r for r in results.values() if not r.passed]

    print(f"Total pairs tested: {len(results)}")
    print(f"Passed: {len(passed)} ({len(passed)/len(results)*100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")

    # Average swap effect
    effects = [r.swap_effect for r in results.values()]
    avg_effect = sum(effects) / len(effects) if effects else 0
    print(f"\nAverage swap effect: {avg_effect:+.3f}")

    # Group by specialist rule
    print("\n--- By Specialist Rule ---")
    for rule in RuleType:
        rule_results = [r for r in results.values() if r.specialist_rule == rule.value]
        if rule_results:
            passed_count = sum(1 for r in rule_results if r.passed)
            avg_original = sum(r.original_score for r in rule_results) / len(rule_results)
            avg_swapped = sum(r.swapped_score for r in rule_results) / len(rule_results)
            print(f"{rule.value:10}: {passed_count}/{len(rule_results)} passed, "
                  f"orig={avg_original:.2f}, swap={avg_swapped:.2f}")

    # Conclusion
    print("\n--- CONCLUSION ---")
    pass_rate = len(passed) / len(results) if results else 0
    if pass_rate >= 0.7:
        print("✓ STRONG CAUSALITY: Prompts significantly affect performance")
    elif pass_rate >= 0.5:
        print("~ MODERATE CAUSALITY: Some prompt effect detected")
    else:
        print("✗ WEAK CAUSALITY: Prompts may not be distinct enough")


async def main():
    """Run Phase 2 enhanced swap test."""
    # Get API key
    GEMINI_KEY = 'AIzaSyDngzJmPnKNrc-jXz5y5xUuDDlwhCWDRic'

    print("Phase 2: Enhanced Prompt Swap Test")
    print("Using enhanced 500+ char prompts with opaque tasks")
    print("="*50)

    results = await run_full_swap_test(
        api_key=GEMINI_KEY,
        n_tasks=5,  # 5 tasks per pair
        timeout_seconds=600
    )

    analyze_results(results)

    # Save results
    import json
    with open('results/phase2_enhanced_results.json', 'w') as f:
        json.dump({k: {
            'specialist_rule': v.specialist_rule,
            'test_rule': v.test_rule,
            'original_score': v.original_score,
            'swapped_score': v.swapped_score,
            'swap_effect': v.swap_effect,
            'passed': v.passed
        } for k, v in results.items()}, f, indent=2)

    print("\nResults saved to results/phase2_enhanced_results.json")


if __name__ == "__main__":
    asyncio.run(main())
