"""
Component Ablation Experiment

Tests which component of strategies drives specialization:
1. Full: reasoning + examples
2. Reasoning only: description + steps, no examples
3. Examples only: just worked examples, no reasoning
4. Neither: no strategy content (control)

Key question: Is specialization due to meta-cognitive reasoning or just few-shot priming?
"""

import asyncio
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genesis.domains import DomainType, get_all_domains
from genesis.evolution_v2 import create_specialist_prompt, SpecialistPrompt
from genesis.benchmark_tasks import BENCHMARK_LOADER, BenchmarkTask
from genesis.llm_client import LLMClient


@dataclass
class AblationCondition:
    """A single ablation condition."""
    name: str
    use_reasoning: bool
    use_example: bool
    description: str


# Define ablation conditions
ABLATION_CONDITIONS = [
    AblationCondition(
        name="full",
        use_reasoning=True,
        use_example=True,
        description="Complete strategies with reasoning and examples"
    ),
    AblationCondition(
        name="reasoning_only",
        use_reasoning=True,
        use_example=False,
        description="Reasoning (description + steps) without worked examples"
    ),
    AblationCondition(
        name="example_only",
        use_reasoning=False,
        use_example=True,
        description="Only worked examples, no meta-cognitive reasoning"
    ),
    AblationCondition(
        name="neither",
        use_reasoning=False,
        use_example=False,
        description="Just strategy names, no content (control)"
    ),
]


@dataclass
class AblationResult:
    """Result for one condition."""
    condition_name: str
    use_reasoning: bool
    use_example: bool
    avg_in_domain_score: float
    avg_cross_domain_score: float
    specialization_gap: float  # in_domain - cross_domain
    domain_scores: Dict[str, float]


@dataclass
class AblationSummary:
    """Summary of all ablation conditions."""
    results: List[Dict]
    best_condition: str
    reasoning_contribution: float  # How much reasoning adds
    example_contribution: float    # How much examples add
    interaction_effect: float      # Synergy between components
    timestamp: str
    model_used: str


def create_ablated_prompt(
    domain: DomainType,
    use_reasoning: bool,
    use_example: bool,
    num_strategies: int = 5
) -> str:
    """Create a specialist prompt with ablated strategy components."""
    from genesis.strategy_library import STRATEGY_LIBRARY
    from genesis.domains import get_domain_spec

    spec = get_domain_spec(domain)

    # Header
    prompt_parts = [
        f"# {spec.name} Specialist",
        f"Core Operation: {spec.core_operation}",
        f"Description: {spec.description}",
        "",
        "## Active Strategies"
    ]

    # Get strategies for this domain
    strategies = STRATEGY_LIBRARY.get_by_domain(domain)[:num_strategies]

    for strategy in strategies:
        strategy_text = strategy.to_prompt_text(
            use_reasoning=use_reasoning,
            use_example=use_example
        )
        prompt_parts.append(strategy_text)
        prompt_parts.append("")

    return "\n".join(prompt_parts)


async def evaluate_condition_on_domain(
    client: LLMClient,
    condition: AblationCondition,
    target_domain: DomainType,
    eval_domain: DomainType,
    tasks_per_domain: int = 5
) -> float:
    """Evaluate a condition's performance on a specific domain."""
    # Create prompt for target domain with ablation settings
    prompt = create_ablated_prompt(
        target_domain,
        use_reasoning=condition.use_reasoning,
        use_example=condition.use_example
    )

    # Get tasks for eval domain
    all_tasks = BENCHMARK_LOADER.load_all_domains(n_per_domain=tasks_per_domain)
    tasks = all_tasks.get(eval_domain, [])[:tasks_per_domain]

    scores = []
    for task in tasks:
        try:
            response = await client.generate(
                prompt=task.prompt,
                system=prompt,
                temperature=0.3,
                max_tokens=300
            )
            score = task.evaluate(response)
            scores.append(score)
        except Exception as e:
            scores.append(0.0)

    return np.mean(scores) if scores else 0.0


async def run_condition(
    client: LLMClient,
    condition: AblationCondition,
    domains: List[DomainType],
    tasks_per_domain: int = 5
) -> AblationResult:
    """Run a single ablation condition across all domains."""
    print(f"\n  Testing condition: {condition.name}")
    print(f"    Reasoning: {condition.use_reasoning}, Examples: {condition.use_example}")

    in_domain_scores = []
    cross_domain_scores = []
    domain_scores = {}

    for target_domain in domains:
        # In-domain score
        in_score = await evaluate_condition_on_domain(
            client, condition, target_domain, target_domain, tasks_per_domain
        )
        in_domain_scores.append(in_score)
        domain_scores[target_domain.value] = in_score

        # Cross-domain scores (average over other domains)
        cross_scores = []
        for eval_domain in domains:
            if eval_domain != target_domain:
                score = await evaluate_condition_on_domain(
                    client, condition, target_domain, eval_domain, tasks_per_domain
                )
                cross_scores.append(score)

        if cross_scores:
            cross_domain_scores.append(np.mean(cross_scores))

    avg_in = np.mean(in_domain_scores) if in_domain_scores else 0.0
    avg_cross = np.mean(cross_domain_scores) if cross_domain_scores else 0.0

    result = AblationResult(
        condition_name=condition.name,
        use_reasoning=condition.use_reasoning,
        use_example=condition.use_example,
        avg_in_domain_score=avg_in,
        avg_cross_domain_score=avg_cross,
        specialization_gap=avg_in - avg_cross,
        domain_scores=domain_scores
    )

    print(f"    In-domain: {avg_in:.3f}, Cross-domain: {avg_cross:.3f}, Gap: {result.specialization_gap:.3f}")

    return result


async def run_component_ablation(
    client: LLMClient,
    num_domains: int = 4,
    tasks_per_domain: int = 3
) -> AblationSummary:
    """
    Run full component ablation study.

    Tests all 4 conditions:
    - full (R+E)
    - reasoning_only (R)
    - example_only (E)
    - neither (control)
    """
    print("=" * 60)
    print("COMPONENT ABLATION STUDY")
    print("=" * 60)
    print(f"Conditions: {[c.name for c in ABLATION_CONDITIONS]}")
    print(f"Domains: {num_domains}")
    print(f"Tasks per domain: {tasks_per_domain}")

    domains = list(get_all_domains())[:num_domains]
    results = []

    for condition in ABLATION_CONDITIONS:
        result = await run_condition(client, condition, domains, tasks_per_domain)
        results.append(result)

    # Analyze contributions
    # Find results by condition
    full = next((r for r in results if r.condition_name == "full"), None)
    reasoning = next((r for r in results if r.condition_name == "reasoning_only"), None)
    example = next((r for r in results if r.condition_name == "example_only"), None)
    neither = next((r for r in results if r.condition_name == "neither"), None)

    # Calculate contributions
    baseline = neither.specialization_gap if neither else 0

    reasoning_contribution = (reasoning.specialization_gap - baseline) if reasoning else 0
    example_contribution = (example.specialization_gap - baseline) if example else 0

    # Interaction effect (synergy or redundancy)
    expected_full = baseline + reasoning_contribution + example_contribution
    actual_full = full.specialization_gap if full else 0
    interaction_effect = actual_full - expected_full

    # Find best condition
    best = max(results, key=lambda r: r.specialization_gap)

    summary = AblationSummary(
        results=[asdict(r) for r in results],
        best_condition=best.condition_name,
        reasoning_contribution=reasoning_contribution,
        example_contribution=example_contribution,
        interaction_effect=interaction_effect,
        timestamp=datetime.now().isoformat(),
        model_used=client.config.model
    )

    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    print(f"{'Condition':<20} {'In-Domain':>12} {'Cross':>12} {'Gap':>12}")
    print("-" * 56)

    for r in results:
        print(f"{r.condition_name:<20} {r.avg_in_domain_score:>12.3f} {r.avg_cross_domain_score:>12.3f} {r.specialization_gap:>12.3f}")

    print("-" * 56)
    print(f"\nBest condition: {summary.best_condition}")
    print(f"Reasoning contribution: {summary.reasoning_contribution:+.3f}")
    print(f"Example contribution: {summary.example_contribution:+.3f}")
    print(f"Interaction effect: {summary.interaction_effect:+.3f}")

    if summary.reasoning_contribution > summary.example_contribution:
        print("\n✅ Reasoning (meta-cognitive) > Examples (few-shot)")
        print("   → Specialization is NOT just few-shot learning")
    else:
        print("\n⚠️  Examples (few-shot) >= Reasoning (meta-cognitive)")
        print("   → May be confounded with few-shot learning")

    return summary


async def run_full_ablation(
    api_key: str = None,
    model: str = "gemini-2.0-flash",
    num_domains: int = 4,
    tasks_per_domain: int = 3
) -> AblationSummary:
    """Run the full ablation study."""

    if api_key:
        client = LLMClient.for_gemini(api_key=api_key, model=model)
    else:
        client = LLMClient.for_gemini(model=model)

    try:
        summary = await run_component_ablation(
            client, num_domains, tasks_per_domain
        )

        # Save results
        output_dir = Path(__file__).parent.parent / "results" / "component_ablation"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "ablation_results.json", "w") as f:
            json.dump(asdict(summary), f, indent=2)

        print(f"\nResults saved to {output_dir / 'ablation_results.json'}")

        return summary

    finally:
        await client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run component ablation")
    parser.add_argument("--api-key", help="Gemini API key")
    parser.add_argument("--model", default="gemini-2.0-flash")
    parser.add_argument("--domains", type=int, default=3)
    parser.add_argument("--tasks", type=int, default=2)

    args = parser.parse_args()

    asyncio.run(run_full_ablation(
        api_key=args.api_key,
        model=args.model,
        num_domains=args.domains,
        tasks_per_domain=args.tasks
    ))
