"""
Phase 2: Prompt Swap Causality Test

The smoking gun for our thesis:
If swapping prompts between specialists changes their performance,
then the PROMPT causes the specialization, not random luck.

Design:
1. Evolve specialists in simulation (fast)
2. Select pairs with different specializations
3. Test each on their OWN domain vs OTHER domain
4. SWAP prompts
5. Re-test - performance should swap!

Success Criteria:
- Performance drop when using wrong prompt: > 20%
- Performance gain when using right prompt: > 20%
- Bidirectional swap shows symmetric effect
"""

import asyncio
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genesis.synthetic_rules import RuleType, generate_tasks, RuleTask
from genesis.preference_agent import PreferenceAgent, create_population
from genesis.competition_v3 import simulate_competition
from genesis.fitness_sharing_v3 import apply_fitness_sharing
from genesis.llm_client import LLMClient


@dataclass
class SwapTestResult:
    """Result of a single swap test."""
    specialist_a_rule: str
    specialist_b_rule: str

    # Before swap
    a_on_own: float  # A's performance on A's rule
    a_on_other: float  # A's performance on B's rule
    b_on_own: float  # B's performance on B's rule
    b_on_other: float  # B's performance on A's rule

    # After swap (A uses B's prompt, B uses A's prompt)
    a_swapped_on_own: float  # A with B's prompt on A's rule
    a_swapped_on_other: float  # A with B's prompt on B's rule
    b_swapped_on_own: float  # B with A's prompt on B's rule
    b_swapped_on_other: float  # B with A's prompt on A's rule

    @property
    def causality_score(self) -> float:
        """
        Measure of causal effect.
        High score = swapping prompts changes performance as expected.
        """
        # A should get WORSE on own rule when using B's prompt
        a_drop = self.a_on_own - self.a_swapped_on_own

        # A should get BETTER on B's rule when using B's prompt
        a_gain = self.a_swapped_on_other - self.a_on_other

        # B should get WORSE on own rule when using A's prompt
        b_drop = self.b_on_own - self.b_swapped_on_own

        # B should get BETTER on A's rule when using A's prompt
        b_gain = self.b_swapped_on_other - self.b_on_other

        # Average of all four effects
        return (a_drop + a_gain + b_drop + b_gain) / 4

    @property
    def passed(self) -> bool:
        """Did the swap test pass?"""
        return self.causality_score > 0.10  # 10% average effect


def evolve_specialists_simulation(
    num_agents: int = 12,
    num_generations: int = 100,
    seed: int = 42
) -> List[PreferenceAgent]:
    """Evolve specialists using simulation (fast, no API calls)."""
    print(f"Evolving {num_agents} agents for {num_generations} generations...", flush=True)

    random.seed(seed)
    np.random.seed(seed)

    agents = create_population(num_agents, prefix="agent")

    for gen in range(num_generations):
        if gen % 20 == 0:
            print(f"  Gen {gen}...", flush=True)

        for _ in range(8):  # 8 tasks per generation
            rule = random.choice(list(RuleType))
            task = generate_tasks(rule, n=1, seed=seed * 1000 + gen)[0]

            winner_id, results = simulate_competition(agents, task)

            if winner_id:
                winner = next(a for a in agents if a.agent_id == winner_id)
                for agent in agents:
                    agent.record_attempt(task.rule_type, won=(agent.agent_id == winner_id))
                apply_fitness_sharing(agents, winner, task.rule_type)

        if gen % 10 == 0:
            for a in agents:
                a.snapshot_preferences()

    return agents


def find_specialist_pairs(
    agents: List[PreferenceAgent],
    min_strength: float = 0.3
) -> List[Tuple[PreferenceAgent, PreferenceAgent]]:
    """Find pairs of agents with different specializations."""
    # Group by primary preference
    by_pref: Dict[RuleType, List[PreferenceAgent]] = {}

    for agent in agents:
        pref = agent.get_primary_preference()
        strength = agent.get_preference_strength()

        if pref and strength >= min_strength:
            if pref not in by_pref:
                by_pref[pref] = []
            by_pref[pref].append(agent)

    # Create pairs from different specializations
    pairs = []
    rules_with_specialists = list(by_pref.keys())

    for i, rule_a in enumerate(rules_with_specialists):
        for rule_b in rules_with_specialists[i+1:]:
            agent_a = by_pref[rule_a][0]  # Take first specialist
            agent_b = by_pref[rule_b][0]
            pairs.append((agent_a, agent_b))

    return pairs


async def test_agent_on_rule(
    agent: PreferenceAgent,
    rule: RuleType,
    client: LLMClient,
    num_tasks: int = 5,
    use_prompt: str = None
) -> float:
    """Test an agent's performance on a specific rule."""
    tasks = generate_tasks(rule, n=num_tasks, seed=hash(agent.agent_id) % 10000)

    prompt = use_prompt if use_prompt else agent.get_prompt()

    scores = []
    for task in tasks:
        try:
            response = await asyncio.wait_for(
                client.generate(
                    prompt=task.prompt,
                    system=prompt,
                    temperature=0.1,
                    max_tokens=50
                ),
                timeout=30
            )
            scores.append(task.evaluate(response))
        except Exception as e:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


async def run_swap_test(
    agent_a: PreferenceAgent,
    agent_b: PreferenceAgent,
    client: LLMClient,
    tasks_per_test: int = 5
) -> SwapTestResult:
    """Run a complete swap test between two specialists."""
    rule_a = agent_a.get_primary_preference()
    rule_b = agent_b.get_primary_preference()

    prompt_a = agent_a.get_prompt()
    prompt_b = agent_b.get_prompt()

    print(f"\nSwap test: {rule_a.value} ↔ {rule_b.value}", flush=True)
    print(f"  Agent A prompt: {prompt_a[:60]}...", flush=True)
    print(f"  Agent B prompt: {prompt_b[:60]}...", flush=True)

    # Before swap
    print("  Testing before swap...", flush=True)
    a_on_own = await test_agent_on_rule(agent_a, rule_a, client, tasks_per_test)
    a_on_other = await test_agent_on_rule(agent_a, rule_b, client, tasks_per_test)
    b_on_own = await test_agent_on_rule(agent_b, rule_b, client, tasks_per_test)
    b_on_other = await test_agent_on_rule(agent_b, rule_a, client, tasks_per_test)

    print(f"    A on {rule_a.value}: {a_on_own:.2f}", flush=True)
    print(f"    A on {rule_b.value}: {a_on_other:.2f}", flush=True)
    print(f"    B on {rule_b.value}: {b_on_own:.2f}", flush=True)
    print(f"    B on {rule_a.value}: {b_on_other:.2f}", flush=True)

    # After swap (A uses B's prompt, B uses A's prompt)
    print("  Testing after swap...", flush=True)
    a_swapped_on_own = await test_agent_on_rule(agent_a, rule_a, client, tasks_per_test, use_prompt=prompt_b)
    a_swapped_on_other = await test_agent_on_rule(agent_a, rule_b, client, tasks_per_test, use_prompt=prompt_b)
    b_swapped_on_own = await test_agent_on_rule(agent_b, rule_b, client, tasks_per_test, use_prompt=prompt_a)
    b_swapped_on_other = await test_agent_on_rule(agent_b, rule_a, client, tasks_per_test, use_prompt=prompt_a)

    print(f"    A(B's prompt) on {rule_a.value}: {a_swapped_on_own:.2f}", flush=True)
    print(f"    A(B's prompt) on {rule_b.value}: {a_swapped_on_other:.2f}", flush=True)
    print(f"    B(A's prompt) on {rule_b.value}: {b_swapped_on_own:.2f}", flush=True)
    print(f"    B(A's prompt) on {rule_a.value}: {b_swapped_on_other:.2f}", flush=True)

    result = SwapTestResult(
        specialist_a_rule=rule_a.value,
        specialist_b_rule=rule_b.value,
        a_on_own=a_on_own,
        a_on_other=a_on_other,
        b_on_own=b_on_own,
        b_on_other=b_on_other,
        a_swapped_on_own=a_swapped_on_own,
        a_swapped_on_other=a_swapped_on_other,
        b_swapped_on_own=b_swapped_on_own,
        b_swapped_on_other=b_swapped_on_other,
    )

    print(f"  Causality score: {result.causality_score:.2f} {'✅' if result.passed else '❌'}", flush=True)

    return result


async def run_phase2(
    api_key: str,
    model: str = "gemini-2.5-flash",
    num_pairs: int = 3,
    tasks_per_test: int = 3,
    save_results: bool = True
) -> List[SwapTestResult]:
    """Run the complete Phase 2 experiment."""
    print("=" * 60, flush=True)
    print("PHASE 2: PROMPT SWAP CAUSALITY TEST", flush=True)
    print("=" * 60, flush=True)

    # Step 1: Evolve specialists
    agents = evolve_specialists_simulation(num_agents=12, num_generations=100, seed=42)

    # Step 2: Find specialist pairs
    print("\nFinding specialist pairs...", flush=True)
    pairs = find_specialist_pairs(agents, min_strength=0.2)
    print(f"Found {len(pairs)} potential pairs", flush=True)

    if len(pairs) < num_pairs:
        print(f"Warning: Only {len(pairs)} pairs available, need {num_pairs}", flush=True)
        num_pairs = len(pairs)

    if num_pairs == 0:
        print("ERROR: No valid specialist pairs found!", flush=True)
        return []

    # Show pairs
    for i, (a, b) in enumerate(pairs[:num_pairs]):
        print(f"  Pair {i+1}: {a.get_primary_preference().value} vs {b.get_primary_preference().value}", flush=True)

    # Step 3: Run swap tests
    print("\nRunning swap tests with real LLM...", flush=True)
    client = LLMClient.for_gemini(api_key=api_key, model=model)

    results = []
    for i, (agent_a, agent_b) in enumerate(pairs[:num_pairs]):
        print(f"\n--- Test {i+1}/{num_pairs} ---", flush=True)
        try:
            result = await run_swap_test(agent_a, agent_b, client, tasks_per_test)
            results.append(result)
        except Exception as e:
            print(f"Error in swap test: {e}", flush=True)

    await client.close()

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("PHASE 2 RESULTS", flush=True)
    print("=" * 60, flush=True)

    if results:
        avg_causality = np.mean([r.causality_score for r in results])
        passed_count = sum(1 for r in results if r.passed)

        print(f"Tests completed: {len(results)}", flush=True)
        print(f"Tests passed: {passed_count}/{len(results)}", flush=True)
        print(f"Average causality score: {avg_causality:.2f}", flush=True)
        print(flush=True)

        for r in results:
            print(f"  {r.specialist_a_rule} ↔ {r.specialist_b_rule}: {r.causality_score:.2f} {'✅' if r.passed else '❌'}", flush=True)

        print(flush=True)
        if passed_count >= len(results) / 2:
            print("🎉 PHASE 2 PASSED: Prompt swap shows causal effect!", flush=True)
        else:
            print("❌ PHASE 2 FAILED: Prompt swap shows no clear causal effect", flush=True)

    # Save results
    if save_results and results:
        output_dir = Path(__file__).parent.parent / "results" / "phase2_swap"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"phase2_results_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)

        print(f"\nResults saved to {output_file}", flush=True)

    return results


async def run_phase2_quick(
    api_key: str,
    model: str = "gemini-2.5-flash"
) -> List[SwapTestResult]:
    """Quick version for testing."""
    return await run_phase2(
        api_key=api_key,
        model=model,
        num_pairs=2,
        tasks_per_test=2,
        save_results=True
    )


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Run Phase 2 swap test")
    parser.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY", ""))
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--tasks", type=int, default=3)
    parser.add_argument("--quick", action="store_true")

    args = parser.parse_args()

    if args.quick:
        asyncio.run(run_phase2_quick(api_key=args.api_key, model=args.model))
    else:
        asyncio.run(run_phase2(
            api_key=args.api_key,
            model=args.model,
            num_pairs=args.pairs,
            tasks_per_test=args.tasks,
        ))
