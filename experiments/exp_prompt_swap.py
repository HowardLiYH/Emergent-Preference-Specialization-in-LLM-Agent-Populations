"""
Bidirectional Prompt Swap Test

Tests whether specialized prompts CAUSALLY drive domain-specific performance.

Key insight: If prompts are truly specialized, then:
1. Agent A (arithmetic specialist) with B's prompt should DROP on arithmetic
2. Agent A with B's prompt should IMPROVE on coding (B's specialty)
3. The improvement should match B's original level

This rules out the "mismatch effect" confound.
"""

import asyncio
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genesis.domains import DomainType, get_all_domains
from genesis.evolution_v2 import (
    PortfolioAgent,
    create_portfolio_agent,
    create_specialist_prompt,
    SpecialistPrompt
)
from genesis.benchmark_tasks import BENCHMARK_LOADER, BenchmarkTask
from genesis.llm_client import LLMClient


@dataclass
class PairwiseSwapResult:
    """Result from swapping prompts between two specialists."""
    agent_a_domain: str
    agent_b_domain: str

    # Original performance
    a_on_a: float  # A on its own domain
    a_on_b: float  # A on B's domain
    b_on_a: float  # B on A's domain
    b_on_b: float  # B on its own domain

    # Swapped performance
    a_with_b_on_a: float  # A using B's prompt, on A's domain
    a_with_b_on_b: float  # A using B's prompt, on B's domain
    b_with_a_on_a: float  # B using A's prompt, on A's domain
    b_with_a_on_b: float  # B using A's prompt, on B's domain

    # Validation checks
    a_drops_on_own: bool       # A gets worse at A's domain with B's prompt
    a_improves_on_other: bool  # A gets better at B's domain with B's prompt
    a_matches_b_level: bool    # A with B's prompt ≈ B's original level
    b_drops_on_own: bool       # Symmetric for B
    b_improves_on_other: bool
    b_matches_a_level: bool

    @property
    def passes_causality_test(self) -> bool:
        """Check if bidirectional transfer confirms causality."""
        return (
            self.a_drops_on_own and
            self.a_improves_on_other and
            self.b_drops_on_own and
            self.b_improves_on_other
        )

    @property
    def num_checks_passed(self) -> int:
        """Count how many of 6 checks passed."""
        return sum([
            self.a_drops_on_own,
            self.a_improves_on_other,
            self.a_matches_b_level,
            self.b_drops_on_own,
            self.b_improves_on_other,
            self.b_matches_a_level
        ])


@dataclass
class PromptSwapSummary:
    """Summary of all pairwise swap tests."""
    num_pairs_tested: int
    num_pairs_passed: int
    pass_rate: float
    avg_checks_passed: float
    pairwise_results: List[Dict]
    timestamp: str
    model_used: str


async def evaluate_agent_on_tasks(
    client: LLMClient,
    agent_prompt: str,
    tasks: List[BenchmarkTask]
) -> float:
    """Evaluate an agent prompt on a set of tasks."""
    scores = []

    for task in tasks:
        try:
            response = await client.generate(
                prompt=task.prompt,
                system=agent_prompt,
                temperature=0.3,
                max_tokens=300
            )
            score = task.evaluate(response)
            scores.append(score)
        except Exception as e:
            print(f"    Error: {e}")
            scores.append(0.0)

    return np.mean(scores) if scores else 0.0


async def run_pairwise_swap(
    client: LLMClient,
    domain_a: DomainType,
    domain_b: DomainType,
    tasks_per_domain: int = 10
) -> PairwiseSwapResult:
    """
    Run prompt swap test between two domain specialists.

    Creates:
    - Specialist A (expert in domain_a)
    - Specialist B (expert in domain_b)

    Tests all 4 combinations of agent × domain, then swaps prompts and retests.
    """
    # Create specialists
    spec_a = create_specialist_prompt(f"spec_{domain_a.value}", domain_a, num_strategies=5)
    spec_b = create_specialist_prompt(f"spec_{domain_b.value}", domain_b, num_strategies=5)

    prompt_a = spec_a.to_prompt_text()
    prompt_b = spec_b.to_prompt_text()

    # Load tasks
    all_tasks = BENCHMARK_LOADER.load_all_domains(n_per_domain=tasks_per_domain)
    tasks_a = all_tasks.get(domain_a, [])[:tasks_per_domain]
    tasks_b = all_tasks.get(domain_b, [])[:tasks_per_domain]

    print(f"  Testing {domain_a.value} ↔ {domain_b.value}...")

    # Original performance (4 conditions)
    a_on_a = await evaluate_agent_on_tasks(client, prompt_a, tasks_a)
    a_on_b = await evaluate_agent_on_tasks(client, prompt_a, tasks_b)
    b_on_a = await evaluate_agent_on_tasks(client, prompt_b, tasks_a)
    b_on_b = await evaluate_agent_on_tasks(client, prompt_b, tasks_b)

    # Swapped performance (4 conditions)
    a_with_b_on_a = await evaluate_agent_on_tasks(client, prompt_b, tasks_a)  # A uses B's prompt on A's tasks
    a_with_b_on_b = await evaluate_agent_on_tasks(client, prompt_b, tasks_b)  # A uses B's prompt on B's tasks
    b_with_a_on_a = await evaluate_agent_on_tasks(client, prompt_a, tasks_a)  # B uses A's prompt on A's tasks
    b_with_a_on_b = await evaluate_agent_on_tasks(client, prompt_a, tasks_b)  # B uses A's prompt on B's tasks

    # Validation checks
    TOLERANCE = 0.15  # 15% tolerance for "matching" levels

    result = PairwiseSwapResult(
        agent_a_domain=domain_a.value,
        agent_b_domain=domain_b.value,

        a_on_a=a_on_a,
        a_on_b=a_on_b,
        b_on_a=b_on_a,
        b_on_b=b_on_b,

        a_with_b_on_a=a_with_b_on_a,
        a_with_b_on_b=a_with_b_on_b,
        b_with_a_on_a=b_with_a_on_a,
        b_with_a_on_b=b_with_a_on_b,

        # A drops on own domain with B's prompt
        a_drops_on_own=(a_with_b_on_a < a_on_a),

        # A improves on B's domain with B's prompt
        a_improves_on_other=(a_with_b_on_b > a_on_b),

        # A with B's prompt matches B's level on B's domain
        a_matches_b_level=(abs(a_with_b_on_b - b_on_b) < TOLERANCE),

        # Symmetric for B
        b_drops_on_own=(b_with_a_on_b < b_on_b),
        b_improves_on_other=(b_with_a_on_a > b_on_a),
        b_matches_a_level=(abs(b_with_a_on_a - a_on_a) < TOLERANCE),
    )

    return result


def print_swap_result(result: PairwiseSwapResult):
    """Pretty print a swap result."""
    print(f"\n  {result.agent_a_domain} ↔ {result.agent_b_domain}:")
    print(f"    Original:")
    print(f"      A on A: {result.a_on_a:.3f}  |  A on B: {result.a_on_b:.3f}")
    print(f"      B on A: {result.b_on_a:.3f}  |  B on B: {result.b_on_b:.3f}")
    print(f"    Swapped:")
    print(f"      A(B's prompt) on A: {result.a_with_b_on_a:.3f}  |  on B: {result.a_with_b_on_b:.3f}")
    print(f"      B(A's prompt) on A: {result.b_with_a_on_a:.3f}  |  on B: {result.b_with_a_on_b:.3f}")
    print(f"    Checks passed: {result.num_checks_passed}/6")
    print(f"    Causality: {'✅ PASS' if result.passes_causality_test else '❌ FAIL'}")


async def run_prompt_swap_test(
    client: LLMClient,
    domains_to_test: List[DomainType] = None,
    tasks_per_domain: int = 5
) -> PromptSwapSummary:
    """
    Run prompt swap tests on multiple domain pairs.

    Args:
        client: LLM client
        domains_to_test: Domains to include (default: first 4)
        tasks_per_domain: Tasks per evaluation

    Returns:
        Summary of all pairwise tests
    """
    if domains_to_test is None:
        # Default to first 4 domains for faster testing
        all_domains = list(get_all_domains())
        domains_to_test = all_domains[:4]

    print("=" * 60)
    print("BIDIRECTIONAL PROMPT SWAP TEST")
    print("=" * 60)
    print(f"Domains: {[d.value for d in domains_to_test]}")
    print(f"Tasks per domain: {tasks_per_domain}")

    results = []

    # Test all pairs
    for i, domain_a in enumerate(domains_to_test):
        for domain_b in domains_to_test[i+1:]:
            result = await run_pairwise_swap(
                client, domain_a, domain_b, tasks_per_domain
            )
            results.append(result)
            print_swap_result(result)

    # Summary
    num_passed = sum(1 for r in results if r.passes_causality_test)
    avg_checks = np.mean([r.num_checks_passed for r in results])

    summary = PromptSwapSummary(
        num_pairs_tested=len(results),
        num_pairs_passed=num_passed,
        pass_rate=num_passed / len(results) if results else 0,
        avg_checks_passed=avg_checks,
        pairwise_results=[asdict(r) for r in results],
        timestamp=datetime.now().isoformat(),
        model_used=client.config.model
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Pairs tested: {summary.num_pairs_tested}")
    print(f"Pairs passed: {summary.num_pairs_passed}")
    print(f"Pass rate: {summary.pass_rate:.1%}")
    print(f"Avg checks passed: {summary.avg_checks_passed:.1f}/6")

    if summary.pass_rate >= 0.8:
        print("\n✅ CAUSALITY CONFIRMED: Prompts drive specialization")
    elif summary.pass_rate >= 0.5:
        print("\n⚠️  PARTIAL SUPPORT: Some evidence for prompt causality")
    else:
        print("\n❌ CAUSALITY NOT CONFIRMED: Prompts may not drive specialization")

    return summary


async def run_full_swap_test(
    api_key: str = None,
    model: str = "gemini-2.0-flash",
    tasks_per_domain: int = 5,
    num_domains: int = 4
) -> PromptSwapSummary:
    """Run the full prompt swap test."""

    if api_key:
        client = LLMClient.for_gemini(api_key=api_key, model=model)
    else:
        client = LLMClient.for_gemini(model=model)

    try:
        domains = list(get_all_domains())[:num_domains]
        summary = await run_prompt_swap_test(client, domains, tasks_per_domain)

        # Save results
        output_dir = Path(__file__).parent.parent / "results" / "prompt_swap"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "swap_results.json", "w") as f:
            json.dump(asdict(summary), f, indent=2)

        print(f"\nResults saved to {output_dir / 'swap_results.json'}")

        return summary

    finally:
        await client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run prompt swap test")
    parser.add_argument("--api-key", help="Gemini API key")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Model")
    parser.add_argument("--tasks", type=int, default=3, help="Tasks per domain")
    parser.add_argument("--domains", type=int, default=3, help="Number of domains")

    args = parser.parse_args()

    asyncio.run(run_full_swap_test(
        api_key=args.api_key,
        model=args.model,
        tasks_per_domain=args.tasks,
        num_domains=args.domains
    ))
