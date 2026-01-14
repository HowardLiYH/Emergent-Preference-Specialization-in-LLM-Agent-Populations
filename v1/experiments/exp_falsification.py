"""
Preference Falsification Experiment

Panel Modification #4: Explicit test to distinguish preference from capability.

Protocol:
1. Take evolved L3 specialist (e.g., VOWEL_START specialist with 100% accuracy)
2. Remove their accumulated strategy (reset to L0)
3. Test on their specialized rule (VOWEL_START tasks)
4. Measure performance

Interpretation:
- If performance drops significantly (e.g., to 30%): Confirmed as PREFERENCE
- If performance persists (e.g., stays at 90%): Actually CAPABILITY

This is the strongest evidence that specialization is preference-based.

Ground Rules Applied:
- Checkpoints for long experiments
- API keys from .env only
- Unified model across tests
"""

import os
import sys
import json
import asyncio
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genesis.llm_client import LLMClient
from src.genesis.synthetic_rules import RuleType, RuleTaskGenerator
from src.genesis.preference_agent import PreferenceAgent, create_handcrafted_specialist
from src.genesis.rule_strategies import build_prompt_from_levels
from src.genesis.analysis import (
    compute_mean_std, compute_confidence_interval, bootstrap_ci,
    cohens_d, paired_t_test
)


# =============================================================================
# CONFIGURATION
# =============================================================================

CHECKPOINT_DIR = Path("results/falsification")
CHECKPOINT_FILE = CHECKPOINT_DIR / "checkpoint.json"
RESULTS_FILE = CHECKPOINT_DIR / "falsification_results.json"
LOG_FILE = CHECKPOINT_DIR / "falsification_log.txt"

# Experiment parameters
N_TASKS_PER_RULE = 20  # Tasks per rule
CHECKPOINT_INTERVAL = 5


@dataclass
class FalsificationResult:
    """Result for one rule's falsification test."""
    rule: str
    n_tasks: int

    # With strategy (L3 specialist)
    with_strategy_correct: int
    with_strategy_accuracy: float

    # Without strategy (reset to L0)
    without_strategy_correct: int
    without_strategy_accuracy: float

    # Analysis
    accuracy_drop: float
    is_preference: bool  # True if performance drops significantly
    cohens_d: float
    p_value: float
    interpretation: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExperimentCheckpoint:
    """Checkpoint for resuming."""
    started_at: str
    rules_completed: List[str]
    current_rule: Optional[str]
    current_task_idx: int
    results: Dict[str, Dict]

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ExperimentCheckpoint":
        return cls(**data)


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def save_checkpoint(checkpoint: ExperimentCheckpoint):
    """Save checkpoint to disk."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint.to_dict(), f, indent=2)


def load_checkpoint() -> Optional[ExperimentCheckpoint]:
    """Load checkpoint if exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
        return ExperimentCheckpoint.from_dict(data)
    return None


def clear_checkpoint():
    """Clear checkpoint."""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


def log(message: str):
    """Log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(log_line + "\n")


# =============================================================================
# CORE EXPERIMENT
# =============================================================================

async def test_agent_on_tasks(
    client: LLMClient,
    agent: PreferenceAgent,
    tasks: List,
    condition: str
) -> Tuple[int, List[float]]:
    """
    Test an agent on a set of tasks.

    Returns (n_correct, list of 0/1 scores)
    """
    system_prompt = build_prompt_from_levels(agent.strategy_levels)
    correct = 0
    scores = []

    for task in tasks:
        prompt = f"""Answer this question. Only provide the answer letter (A, B, C, or D).

Question: {task.prompt}"""

        try:
            response = await client.generate(
                prompt,
                system=system_prompt if agent.strategy_levels else None,
                temperature=0.3,
                max_tokens=20
            )

            # Parse answer
            answer = response.strip().upper()[:1]
            score = task.evaluate(answer)
            is_correct = score >= 0.8

            if is_correct:
                correct += 1
            scores.append(1.0 if is_correct else 0.0)

        except Exception as e:
            scores.append(0.0)

    return correct, scores


async def run_falsification_for_rule(
    client: LLMClient,
    rule: RuleType,
    n_tasks: int,
    seed: int
) -> FalsificationResult:
    """
    Run falsification test for one rule.

    1. Create L3 specialist for this rule
    2. Test with strategy
    3. Reset to L0
    4. Test without strategy
    5. Compare
    """
    log(f"  Testing rule: {rule.value}")

    # Generate tasks for this specific rule
    generator = RuleTaskGenerator(seed=seed)
    tasks = []
    for _ in range(n_tasks * 3):  # Generate extra to ensure enough of this rule
        task = generator.generate()
        if task.rule_type == rule:
            tasks.append(task)
        if len(tasks) >= n_tasks:
            break

    if len(tasks) < n_tasks:
        # Generate more if needed
        for _ in range(n_tasks - len(tasks)):
            task = generator.generate(rule_type=rule)
            tasks.append(task)

    tasks = tasks[:n_tasks]

    # Create L3 specialist
    specialist = create_handcrafted_specialist(rule, agent_id=f"specialist_{rule.value}")

    # Test WITH strategy
    with_correct, with_scores = await test_agent_on_tasks(
        client, specialist, tasks, "with_strategy"
    )
    with_accuracy = with_correct / len(tasks)

    # Reset to L0 (remove strategy)
    generalist = PreferenceAgent(
        agent_id=f"reset_{rule.value}",
        strategy_levels={r: 0 for r in RuleType}  # All zeros
    )

    # Test WITHOUT strategy
    without_correct, without_scores = await test_agent_on_tasks(
        client, generalist, tasks, "without_strategy"
    )
    without_accuracy = without_correct / len(tasks)

    # Statistical analysis
    accuracy_drop = with_accuracy - without_accuracy

    # Effect size
    d = cohens_d(with_scores, without_scores)

    # Paired t-test (same tasks, different conditions)
    try:
        test_result = paired_t_test(without_scores, with_scores)
        p_value = test_result.p_value
    except:
        p_value = 1.0

    # Interpretation
    # Preference = performance drops significantly when strategy removed
    is_preference = accuracy_drop > 0.20 and p_value < 0.05

    if is_preference:
        interpretation = f"PREFERENCE: {accuracy_drop:.0%} drop when strategy removed (d={d:.2f}, p={p_value:.4f})"
    elif accuracy_drop > 0.10:
        interpretation = f"LIKELY PREFERENCE: {accuracy_drop:.0%} drop, but not statistically significant (p={p_value:.4f})"
    elif accuracy_drop > 0.05:
        interpretation = f"WEAK PREFERENCE: Small {accuracy_drop:.0%} drop (p={p_value:.4f})"
    else:
        interpretation = f"CAPABILITY: Performance persists without strategy ({without_accuracy:.0%} vs {with_accuracy:.0%})"

    log(f"    With strategy: {with_accuracy:.0%}")
    log(f"    Without strategy: {without_accuracy:.0%}")
    log(f"    {interpretation}")

    return FalsificationResult(
        rule=rule.value,
        n_tasks=len(tasks),
        with_strategy_correct=with_correct,
        with_strategy_accuracy=with_accuracy,
        without_strategy_correct=without_correct,
        without_strategy_accuracy=without_accuracy,
        accuracy_drop=accuracy_drop,
        is_preference=is_preference,
        cohens_d=d,
        p_value=p_value,
        interpretation=interpretation
    )


async def run_falsification_experiment(
    n_tasks_per_rule: int = N_TASKS_PER_RULE,
    seed: int = 42,
    resume: bool = True
) -> Dict:
    """
    Run the full falsification experiment across all rules.
    """
    log("=" * 60)
    log("PREFERENCE FALSIFICATION EXPERIMENT")
    log("=" * 60)

    # Check checkpoint
    checkpoint = None
    if resume:
        checkpoint = load_checkpoint()

    if checkpoint is None:
        checkpoint = ExperimentCheckpoint(
            started_at=datetime.now().isoformat(),
            rules_completed=[],
            current_rule=None,
            current_task_idx=0,
            results={}
        )

    # Initialize
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    client = LLMClient()
    log(f"Using model: {client.config.model}")

    # Test each rule
    all_rules = list(RuleType)

    for rule in all_rules:
        if rule.value in checkpoint.rules_completed:
            log(f"[{rule.value}] Already completed, skipping")
            continue

        log(f"\n[{rule.value}] Testing...")
        checkpoint.current_rule = rule.value

        try:
            result = await run_falsification_for_rule(
                client, rule, n_tasks_per_rule, seed
            )

            checkpoint.results[rule.value] = result.to_dict()
            checkpoint.rules_completed.append(rule.value)
            save_checkpoint(checkpoint)

        except Exception as e:
            log(f"[{rule.value}] ERROR: {e}")
            save_checkpoint(checkpoint)
            raise

    # Summary
    log("\n" + "=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)

    results = checkpoint.results

    # Count preferences vs capabilities
    n_preference = sum(1 for r in results.values() if r.get('is_preference', False))
    n_total = len(results)

    log(f"\nPreference confirmed: {n_preference}/{n_total} rules ({n_preference/n_total:.0%})")

    # Table
    log("\n| Rule | With Strategy | Without | Drop | Verdict |")
    log("|------|---------------|---------|------|---------|")
    for rule_name, r in results.items():
        verdict = "PREFERENCE" if r['is_preference'] else "CAPABILITY"
        log(f"| {rule_name:12s} | {r['with_strategy_accuracy']:.0%} | {r['without_strategy_accuracy']:.0%} | {r['accuracy_drop']:+.0%} | {verdict} |")

    # Aggregate statistics
    drops = [r['accuracy_drop'] for r in results.values()]
    mean_drop, std_drop = compute_mean_std(drops)
    ci_low, ci_high = bootstrap_ci(drops, n_bootstrap=10000)

    log(f"\nAggregate accuracy drop: {mean_drop:.1%} ± {std_drop:.1%}")
    log(f"Bootstrap 95% CI: [{ci_low:.1%}, {ci_high:.1%}]")

    # Success criterion
    success = n_preference >= n_total * 0.75  # At least 75% of rules show preference
    log(f"\nSuccess criterion (>=75% preference): {'MET' if success else 'NOT MET'}")

    # Thesis support
    if success:
        thesis_statement = (
            "STRONG SUPPORT: Evolved specialization is preference-based, not capability-based. "
            "Removing accumulated strategies causes significant performance drops, demonstrating "
            "that the prompts encode task-specific preferences rather than general capabilities."
        )
    else:
        thesis_statement = (
            "PARTIAL SUPPORT: Some rules show preference-based specialization, but results are mixed. "
            "Further investigation needed to understand which rule types are more amenable to preference learning."
        )

    log(f"\nThesis support: {thesis_statement}")

    # Save final results
    final_results = {
        "experiment": "preference_falsification",
        "timestamp": datetime.now().isoformat(),
        "n_tasks_per_rule": n_tasks_per_rule,
        "seed": seed,
        "model": client.config.model,
        "results": results,
        "summary": {
            "n_preference": n_preference,
            "n_total": n_total,
            "preference_rate": n_preference / n_total,
            "mean_accuracy_drop": mean_drop,
            "std_accuracy_drop": std_drop,
            "bootstrap_ci_95": [ci_low, ci_high],
            "success": success
        },
        "thesis_statement": thesis_statement,
        "cost_estimate": client.usage.to_dict() if hasattr(client, 'usage') else {}
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(final_results, f, indent=2)

    log(f"\nResults saved to {RESULTS_FILE}")
    clear_checkpoint()

    return final_results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run preference falsification experiment")
    parser.add_argument("--n-tasks", type=int, default=20, help="Tasks per rule")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh")

    args = parser.parse_args()

    asyncio.run(run_falsification_experiment(
        n_tasks_per_rule=args.n_tasks,
        seed=args.seed,
        resume=not args.no_resume
    ))
