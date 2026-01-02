"""
Phase 0: Rule Validation Experiment

Tests:
1. Rule Orthogonality - specialists perform well on own rule, poorly on others
2. Handcrafted Ceiling - maximum achievable performance per rule
3. Competition Mechanism - confidence parsing works correctly

Success Criteria:
- Orthogonality gap > 30%
- All ceiling scores > 85%
- Competition produces clear winners
"""

import asyncio
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genesis.synthetic_rules import RuleType, generate_tasks, RuleTask
from genesis.preference_agent import PreferenceAgent, create_handcrafted_specialist
from genesis.competition_v3 import run_competition, parse_answer_confidence
from genesis.llm_client import LLMClient


@dataclass
class ValidationResults:
    """Results from Phase 0 validation."""
    # Orthogonality
    overlap_matrix: Dict[str, Dict[str, float]]
    diagonal_mean: float
    off_diagonal_mean: float
    orthogonality_gap: float
    orthogonality_pass: bool

    # Ceiling
    ceiling_scores: Dict[str, float]
    ceiling_mean: float
    ceiling_pass: bool

    # Competition
    competition_winner_rate: float
    competition_confidence_variance: float
    competition_pass: bool

    # Overall
    all_pass: bool
    timestamp: str
    model_used: str

    def to_dict(self) -> dict:
        return asdict(self)


async def test_orthogonality(
    client: LLMClient,
    tasks_per_rule: int = 5
) -> Tuple[Dict[str, Dict[str, float]], float, float, float]:
    """
    Test 1: Rule Orthogonality

    Create specialist for each rule, test on all rules.
    Compute 8x8 overlap matrix.
    """
    print("\n" + "=" * 60)
    print("TEST 1: RULE ORTHOGONALITY")
    print("=" * 60)

    overlap_matrix = {r.value: {} for r in RuleType}

    for spec_rule in RuleType:
        print(f"\nTesting {spec_rule.value} specialist...")
        specialist = create_handcrafted_specialist(spec_rule)

        for test_rule in RuleType:
            tasks = generate_tasks(test_rule, n=tasks_per_rule)
            scores = []

            for task in tasks:
                try:
                    response = await client.generate(
                        prompt=task.prompt,
                        system=specialist.get_prompt(),
                        temperature=0.1,
                        max_tokens=100
                    )
                    score = task.evaluate(response)
                    scores.append(score)
                except Exception as e:
                    print(f"    Error on {test_rule.value}: {e}")
                    scores.append(0.0)

            avg_score = np.mean(scores) if scores else 0.0
            overlap_matrix[spec_rule.value][test_rule.value] = avg_score

        # Print row
        row_scores = [f"{overlap_matrix[spec_rule.value][r.value]:.2f}" for r in RuleType]
        print(f"  {spec_rule.value:12} | {' | '.join(row_scores)}")

    # Compute metrics
    diagonal_scores = [overlap_matrix[r.value][r.value] for r in RuleType]
    off_diagonal_scores = [
        overlap_matrix[r1.value][r2.value]
        for r1 in RuleType for r2 in RuleType
        if r1 != r2
    ]

    diagonal_mean = np.mean(diagonal_scores)
    off_diagonal_mean = np.mean(off_diagonal_scores)
    gap = (diagonal_mean - off_diagonal_mean) / diagonal_mean if diagonal_mean > 0 else 0

    print(f"\nDiagonal mean: {diagonal_mean:.3f}")
    print(f"Off-diagonal mean: {off_diagonal_mean:.3f}")
    print(f"Gap: {gap:.1%}")
    print(f"Pass: {'✅' if gap > 0.30 else '❌'} (threshold: 30%)")

    return overlap_matrix, diagonal_mean, off_diagonal_mean, gap


async def test_ceiling_baseline(
    client: LLMClient,
    tasks_per_rule: int = 10
) -> Tuple[Dict[str, float], float]:
    """
    Test 2: Handcrafted Ceiling Baseline

    Measure maximum achievable performance with perfect Level 3 strategies.
    """
    print("\n" + "=" * 60)
    print("TEST 2: HANDCRAFTED CEILING BASELINE")
    print("=" * 60)

    ceiling_scores = {}

    for rule_type in RuleType:
        specialist = create_handcrafted_specialist(rule_type)
        tasks = generate_tasks(rule_type, n=tasks_per_rule)

        scores = []
        for task in tasks:
            try:
                response = await client.generate(
                    prompt=task.prompt,
                    system=specialist.get_prompt(),
                    temperature=0.1,
                    max_tokens=100
                )
                score = task.evaluate(response)
                scores.append(score)
            except Exception as e:
                scores.append(0.0)

        avg_score = np.mean(scores) if scores else 0.0
        ceiling_scores[rule_type.value] = avg_score

        status = "✅" if avg_score >= 0.85 else "❌"
        print(f"  {rule_type.value:12}: {avg_score:.3f} {status}")

    ceiling_mean = np.mean(list(ceiling_scores.values()))
    all_pass = all(s >= 0.85 for s in ceiling_scores.values())

    print(f"\nCeiling mean: {ceiling_mean:.3f}")
    print(f"All rules >= 85%: {'✅' if all_pass else '❌'}")

    return ceiling_scores, ceiling_mean


async def test_competition_mechanism(
    client: LLMClient,
    num_tests: int = 10
) -> Tuple[float, float]:
    """
    Test 3: Competition Mechanism

    Verify that confidence parsing works and produces clear winners.
    """
    print("\n" + "=" * 60)
    print("TEST 3: COMPETITION MECHANISM")
    print("=" * 60)

    # Create a few test agents with different strategy levels
    agents = [
        PreferenceAgent(agent_id="agent_0"),  # No strategies
        PreferenceAgent(agent_id="agent_1"),  # Level 1 on POSITION
        PreferenceAgent(agent_id="agent_2"),  # Level 3 on POSITION
    ]
    agents[1].strategy_levels[RuleType.POSITION] = 1
    agents[2].strategy_levels[RuleType.POSITION] = 3

    winners = []
    confidences = []

    for i in range(num_tests):
        task = generate_tasks(RuleType.POSITION, n=1)[0]

        winner_id, results = await run_competition(agents, task, client)

        if winner_id:
            winners.append(winner_id)

        for r in results:
            confidences.append(r.confidence)

        print(f"  Test {i+1}: Winner = {winner_id}, Confidences = {[f'{r.confidence:.2f}' for r in results]}")

    winner_rate = len(winners) / num_tests
    confidence_variance = np.var(confidences) if confidences else 0

    print(f"\nWinner rate: {winner_rate:.1%}")
    print(f"Confidence variance: {confidence_variance:.4f}")
    print(f"Competition working: {'✅' if winner_rate > 0.5 and confidence_variance > 0.01 else '❌'}")

    return winner_rate, confidence_variance


async def run_phase0_validation(
    client: LLMClient,
    tasks_per_rule: int = 5
) -> ValidationResults:
    """
    Run all Phase 0 validation tests.
    """
    print("=" * 60)
    print("PHASE 0: COMPREHENSIVE VALIDATION")
    print("=" * 60)
    print(f"Model: {client.config.model}")
    print(f"Tasks per rule: {tasks_per_rule}")

    # Test 1: Orthogonality
    overlap_matrix, diag_mean, off_diag_mean, gap = await test_orthogonality(
        client, tasks_per_rule
    )
    orthogonality_pass = gap > 0.30

    # Test 2: Ceiling
    ceiling_scores, ceiling_mean = await test_ceiling_baseline(
        client, tasks_per_rule * 2  # More samples for ceiling
    )
    ceiling_pass = all(s >= 0.85 for s in ceiling_scores.values())

    # Test 3: Competition
    winner_rate, conf_var = await test_competition_mechanism(client, 10)
    competition_pass = winner_rate > 0.5 and conf_var > 0.01

    # Overall
    all_pass = orthogonality_pass and ceiling_pass and competition_pass

    results = ValidationResults(
        overlap_matrix=overlap_matrix,
        diagonal_mean=diag_mean,
        off_diagonal_mean=off_diag_mean,
        orthogonality_gap=gap,
        orthogonality_pass=orthogonality_pass,
        ceiling_scores=ceiling_scores,
        ceiling_mean=ceiling_mean,
        ceiling_pass=ceiling_pass,
        competition_winner_rate=winner_rate,
        competition_confidence_variance=conf_var,
        competition_pass=competition_pass,
        all_pass=all_pass,
        timestamp=datetime.now().isoformat(),
        model_used=client.config.model
    )

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 0 SUMMARY")
    print("=" * 60)
    print(f"Orthogonality (gap > 30%):     {gap:.1%} {'✅' if orthogonality_pass else '❌'}")
    print(f"Ceiling (all >= 85%):          {ceiling_mean:.1%} {'✅' if ceiling_pass else '❌'}")
    print(f"Competition (winners > 50%):   {winner_rate:.1%} {'✅' if competition_pass else '❌'}")
    print(f"\nOVERALL: {'✅ PASS - Proceed to Phase 1' if all_pass else '❌ FAIL - Review pivot tree'}")

    return results


async def run_phase0(
    api_key: str = None,
    model: str = "gemini-2.0-flash",
    tasks_per_rule: int = 5,
    save_results: bool = True
) -> ValidationResults:
    """Main entry point for Phase 0 validation."""

    # Create client
    if api_key:
        client = LLMClient.for_gemini(api_key=api_key, model=model)
    else:
        client = LLMClient.for_gemini(model=model)

    try:
        results = await run_phase0_validation(client, tasks_per_rule)

        # Save results
        if save_results:
            output_dir = Path(__file__).parent.parent / "results" / "phase0_validation"
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"validation_{timestamp}.json"

            with open(output_file, "w") as f:
                json.dump(results.to_dict(), f, indent=2)

            print(f"\nResults saved to {output_file}")

        return results

    finally:
        await client.close()


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Run Phase 0 validation")
    parser.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY", ""), help="Gemini API key")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Model to use")
    parser.add_argument("--tasks", type=int, default=5, help="Tasks per rule")

    args = parser.parse_args()

    asyncio.run(run_phase0(
        api_key=args.api_key,
        model=args.model,
        tasks_per_rule=args.tasks
    ))
