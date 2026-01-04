"""
Domain Overlap Experiment

Measures how distinct the 10 domains are by testing specialists on cross-domain tasks.

Success Criteria:
- Diagonal (in-domain) > 0.7
- Off-diagonal (cross-domain) < 0.4
- Gap > 0.3

If domains overlap too much, agents can't meaningfully specialize.
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
class OverlapResult:
    """Result of domain overlap analysis."""
    overlap_matrix: List[List[float]]  # 10x10 matrix
    domain_names: List[str]
    diagonal_mean: float
    off_diagonal_mean: float
    gap: float
    is_orthogonal: bool
    timestamp: str
    model_used: str
    tasks_per_domain: int


async def evaluate_specialist_on_task(
    client: LLMClient,
    specialist: SpecialistPrompt,
    task: BenchmarkTask
) -> float:
    """Evaluate a specialist on a single task."""
    try:
        prompt_text = specialist.to_prompt_text()

        response = await client.generate(
            prompt=task.prompt,
            system=prompt_text,
            temperature=0.3,
            max_tokens=300
        )

        score = task.evaluate(response)
        return score

    except Exception as e:
        print(f"    Error evaluating task: {e}")
        return 0.0


async def measure_domain_overlap(
    client: LLMClient,
    tasks_per_domain: int = 10,
    verbose: bool = True
) -> OverlapResult:
    """
    Measure performance overlap between domain specialists.

    Creates a 10x10 matrix where [i,j] = specialist_i's score on domain_j tasks.

    Args:
        client: LLM client for evaluation
        tasks_per_domain: Number of tasks to test per domain
        verbose: Print progress

    Returns:
        OverlapResult with full analysis
    """
    domains = list(get_all_domains())
    n_domains = len(domains)
    overlap_matrix = np.zeros((n_domains, n_domains))

    # Load tasks for all domains
    if verbose:
        print("Loading benchmark tasks...")
    all_tasks = BENCHMARK_LOADER.load_all_domains(n_per_domain=tasks_per_domain)

    # Create specialists for all domains
    if verbose:
        print("Creating domain specialists...")
    specialists = {}
    for domain in domains:
        specialists[domain] = create_specialist_prompt(
            f"specialist_{domain.value}",
            domain,
            num_strategies=5,
            num_examples=2
        )

    # Evaluate each specialist on each domain's tasks
    if verbose:
        print(f"\nEvaluating {n_domains}x{n_domains} = {n_domains**2} specialist-domain pairs...")
        print("=" * 60)

    for i, spec_domain in enumerate(domains):
        specialist = specialists[spec_domain]

        if verbose:
            print(f"\n[{i+1}/{n_domains}] Testing {spec_domain.value} specialist:")

        for j, task_domain in enumerate(domains):
            tasks = all_tasks.get(task_domain, [])[:tasks_per_domain]

            if not tasks:
                overlap_matrix[i, j] = 0.5  # Default if no tasks
                continue

            # Evaluate on all tasks
            scores = []
            for task in tasks:
                score = await evaluate_specialist_on_task(client, specialist, task)
                scores.append(score)

            mean_score = np.mean(scores) if scores else 0.0
            overlap_matrix[i, j] = mean_score

            marker = "✓" if i == j else ""
            if verbose:
                print(f"  → {task_domain.value}: {mean_score:.3f} {marker}")

    # Compute statistics
    diagonal = np.diag(overlap_matrix)
    diagonal_mean = np.mean(diagonal)

    # Off-diagonal elements
    mask = ~np.eye(n_domains, dtype=bool)
    off_diagonal = overlap_matrix[mask]
    off_diagonal_mean = np.mean(off_diagonal)

    gap = diagonal_mean - off_diagonal_mean
    is_orthogonal = diagonal_mean > 0.7 and off_diagonal_mean < 0.4

    result = OverlapResult(
        overlap_matrix=overlap_matrix.tolist(),
        domain_names=[d.value for d in domains],
        diagonal_mean=float(diagonal_mean),
        off_diagonal_mean=float(off_diagonal_mean),
        gap=float(gap),
        is_orthogonal=is_orthogonal,
        timestamp=datetime.now().isoformat(),
        model_used=client.config.model,
        tasks_per_domain=tasks_per_domain
    )

    return result


def print_overlap_matrix(result: OverlapResult):
    """Pretty print the overlap matrix."""
    print("\n" + "=" * 80)
    print("DOMAIN OVERLAP MATRIX")
    print("=" * 80)

    domains = result.domain_names
    matrix = np.array(result.overlap_matrix)

    # Header
    print(f"{'':>12}", end="")
    for d in domains:
        print(f"{d[:8]:>10}", end="")
    print()
    print("-" * (12 + 10 * len(domains)))

    # Rows
    for i, row_domain in enumerate(domains):
        print(f"{row_domain[:12]:>12}", end="")
        for j, val in enumerate(matrix[i]):
            marker = "*" if i == j else " "
            print(f"{val:>9.2f}{marker}", end="")
        print()

    print("-" * (12 + 10 * len(domains)))

    # Summary
    print(f"\nDiagonal (in-domain) mean:    {result.diagonal_mean:.3f}")
    print(f"Off-diagonal (cross) mean:    {result.off_diagonal_mean:.3f}")
    print(f"Gap:                          {result.gap:.3f}")
    print(f"Orthogonal (success):         {'✅ YES' if result.is_orthogonal else '❌ NO'}")

    if not result.is_orthogonal:
        print("\n⚠️  WARNING: Domains not sufficiently distinct!")
        print("   Consider merging similar domains or using harder tasks.")


def validate_domain_orthogonality(result: OverlapResult) -> Dict:
    """
    Check if domains meet orthogonality criteria.

    Returns detailed validation results.
    """
    matrix = np.array(result.overlap_matrix)
    domains = result.domain_names

    # Find problematic domain pairs
    problematic_pairs = []
    for i in range(len(domains)):
        for j in range(len(domains)):
            if i != j and matrix[i, j] > 0.5:  # High cross-domain performance
                problematic_pairs.append({
                    "specialist": domains[i],
                    "task_domain": domains[j],
                    "score": matrix[i, j],
                    "issue": "High cross-domain performance"
                })

    # Find weak specialties
    weak_specialties = []
    for i, domain in enumerate(domains):
        if matrix[i, i] < 0.6:
            weak_specialties.append({
                "domain": domain,
                "in_domain_score": matrix[i, i],
                "issue": "Low in-domain performance"
            })

    return {
        "passes": result.is_orthogonal,
        "diagonal_mean": result.diagonal_mean,
        "off_diagonal_mean": result.off_diagonal_mean,
        "gap": result.gap,
        "criteria": {
            "diagonal_threshold": 0.7,
            "off_diagonal_threshold": 0.4,
            "gap_threshold": 0.3
        },
        "problematic_pairs": problematic_pairs,
        "weak_specialties": weak_specialties,
        "recommendation": "Domains are distinct enough" if result.is_orthogonal
            else "Consider merging or refining domains"
    }


async def run_domain_overlap_test(
    api_key: str = None,
    model: str = "gemini-2.5-flash",
    tasks_per_domain: int = 5
) -> OverlapResult:
    """
    Run the full domain overlap test.

    Args:
        api_key: Gemini API key (uses env if not provided)
        model: Model to use
        tasks_per_domain: Tasks per domain (fewer = faster)

    Returns:
        OverlapResult with full analysis
    """
    print("=" * 60)
    print("DOMAIN OVERLAP TEST")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Tasks per domain: {tasks_per_domain}")
    print()

    # Create client
    if api_key:
        client = LLMClient.for_gemini(api_key=api_key, model=model)
    else:
        client = LLMClient.for_gemini(model=model)

    try:
        result = await measure_domain_overlap(
            client,
            tasks_per_domain=tasks_per_domain,
            verbose=True
        )

        # Print results
        print_overlap_matrix(result)

        # Validate
        validation = validate_domain_orthogonality(result)

        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        print(json.dumps(validation, indent=2))

        # Save results
        output_dir = Path(__file__).parent.parent / "results" / "domain_overlap"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "overlap_results.json", "w") as f:
            json.dump(asdict(result), f, indent=2)

        print(f"\nResults saved to {output_dir / 'overlap_results.json'}")

        return result

    finally:
        await client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run domain overlap test")
    parser.add_argument("--api-key", help="Gemini API key")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Model to use")
    parser.add_argument("--tasks", type=int, default=5, help="Tasks per domain")

    args = parser.parse_args()

    asyncio.run(run_domain_overlap_test(
        api_key=args.api_key,
        model=args.model,
        tasks_per_domain=args.tasks
    ))
