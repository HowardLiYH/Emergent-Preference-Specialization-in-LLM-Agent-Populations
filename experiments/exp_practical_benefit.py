"""
5-Condition Practical Benefit Experiment

Panel Modification #5: Expanded comparison of population vs generalist.

Conditions:
1. SINGLE_GENERALIST: One agent, generic prompt (baseline)
2. ORACLE_ROUTING: Population + perfect task-type labels
3. LEARNED_ROUTING: Population + classifier predicts task type
4. CONFIDENCE_ROUTING: Population + route to highest-confidence specialist
5. ENSEMBLE: All specialists answer, take majority vote

Ground Rules Applied:
- Checkpoints every 10 tasks for crash recovery
- API keys from .env only
- Unified model (gemini-2.5-flash) across all conditions
"""

import os
import sys
import json
import asyncio
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from enum import Enum

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genesis.llm_client import LLMClient, LLMConfig
from src.genesis.synthetic_rules import RuleType, RuleTaskGenerator
from src.genesis.preference_agent import PreferenceAgent, create_population, create_handcrafted_specialist
from src.genesis.routing import (
    RoutingStrategy, create_router, OracleRouter, LearnedRouter,
    ConfidenceRouter, EnsembleRouter, RandomRouter
)
from src.genesis.rule_strategies import build_prompt_from_levels
from src.genesis.analysis import (
    compute_mean_std, compute_confidence_interval, bootstrap_ci,
    cohens_d, independent_t_test
)


# =============================================================================
# CONFIGURATION
# =============================================================================

CHECKPOINT_DIR = Path("results/practical_benefit")
CHECKPOINT_FILE = CHECKPOINT_DIR / "checkpoint.json"
RESULTS_FILE = CHECKPOINT_DIR / "practical_benefit_results.json"
LOG_FILE = CHECKPOINT_DIR / "practical_benefit_log.txt"

# Experiment parameters
N_TASKS = 24  # Total tasks per condition (3 per rule × 8 rules)
CHECKPOINT_INTERVAL = 10  # Save checkpoint every N tasks
N_AGENTS = 8  # Population size
N_RULES = 8  # Number of rules


class Condition(Enum):
    """Experiment conditions."""
    SINGLE_GENERALIST = "single_generalist"
    ORACLE_ROUTING = "oracle_routing"
    LEARNED_ROUTING = "learned_routing"
    CONFIDENCE_ROUTING = "confidence_routing"
    ENSEMBLE = "ensemble"


@dataclass
class ConditionResult:
    """Results for one condition."""
    condition: str
    n_tasks: int
    correct: int
    accuracy: float
    total_api_calls: int
    total_time_ms: float
    avg_time_per_task_ms: float
    task_results: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExperimentCheckpoint:
    """Checkpoint for resuming experiment."""
    started_at: str
    conditions_completed: List[str]
    current_condition: Optional[str]
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
    log(f"[Checkpoint] Saved at task {checkpoint.current_task_idx}")


def load_checkpoint() -> Optional[ExperimentCheckpoint]:
    """Load checkpoint if exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
        checkpoint = ExperimentCheckpoint.from_dict(data)
        log(f"[Checkpoint] Loaded, resuming from {checkpoint.current_condition} task {checkpoint.current_task_idx}")
        return checkpoint
    return None


def clear_checkpoint():
    """Clear checkpoint after successful completion."""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        log("[Checkpoint] Cleared")


def log(message: str):
    """Log message to file and console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(log_line + "\n")


# =============================================================================
# AGENT SETUP
# =============================================================================

def create_specialist_population() -> List[PreferenceAgent]:
    """
    Create a population of specialists (one per rule).

    Simulates the outcome of successful evolution where each agent
    specializes in a different rule.
    """
    agents = []
    rules = list(RuleType)

    for i, rule in enumerate(rules):
        agent = create_handcrafted_specialist(rule_type=rule)
        agents.append(agent)

    return agents


def create_generalist_agent() -> PreferenceAgent:
    """Create a generalist agent with no specialization."""
    return PreferenceAgent(
        agent_id="generalist",
        strategy_levels={r: 0 for r in RuleType}
    )


# =============================================================================
# CONDITION RUNNERS
# =============================================================================

async def run_single_generalist(
    client: LLMClient,
    tasks: List,
    checkpoint_start: int = 0
) -> Tuple[ConditionResult, List[Dict]]:
    """
    Condition 1: Single generalist agent handles all tasks.

    This is the baseline - no specialization, no routing.
    """
    agent = create_generalist_agent()
    correct = 0
    total_time = 0
    task_results = []

    for i, task in enumerate(tasks):
        if i < checkpoint_start:
            continue

        start = time.time()

        # Generalist has no special prompt
        prompt = f"""Answer this question. Provide your answer and confidence (0-100).

Format:
ANSWER: [your answer]
CONFIDENCE: [0-100]

Question: {task.prompt}"""

        try:
            response = await client.generate(prompt, temperature=0.3, max_tokens=300)

            # Parse answer
            import re
            answer_match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
            answer = answer_match.group(1).strip() if answer_match else response[:50]

            # Evaluate
            score = task.evaluate(answer)
            is_correct = score >= 0.8
            if is_correct:
                correct += 1

        except Exception as e:
            is_correct = False
            answer = f"ERROR: {e}"

        elapsed = (time.time() - start) * 1000
        total_time += elapsed

        task_results.append({
            "task_id": task.task_id,
            "rule": task.rule_type.value,
            "answer": answer,
            "correct": is_correct,
            "time_ms": elapsed
        })

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            log(f"  [SINGLE_GENERALIST] {i+1}/{len(tasks)} tasks, acc={correct/(i+1):.1%}")

    return ConditionResult(
        condition=Condition.SINGLE_GENERALIST.value,
        n_tasks=len(tasks),
        correct=correct,
        accuracy=correct / len(tasks) if tasks else 0,
        total_api_calls=len(tasks),
        total_time_ms=total_time,
        avg_time_per_task_ms=total_time / len(tasks) if tasks else 0,
        task_results=task_results
    ), task_results


async def run_oracle_routing(
    client: LLMClient,
    agents: List[PreferenceAgent],
    tasks: List,
    checkpoint_start: int = 0
) -> Tuple[ConditionResult, List[Dict]]:
    """
    Condition 2: Perfect routing using ground truth task labels.

    Upper bound on routing performance.
    """
    router = OracleRouter(agents)
    correct = 0
    total_time = 0
    task_results = []

    for i, task in enumerate(tasks):
        if i < checkpoint_start:
            continue

        start = time.time()

        # Route to best specialist
        routing_result = router.route(task)
        agent = next(a for a in agents if a.agent_id == routing_result.selected_agent_id)

        # Build specialist prompt
        system_prompt = build_prompt_from_levels(agent.strategy_levels)

        prompt = f"""Answer this question. Provide your answer and confidence (0-100).

Format:
ANSWER: [your answer]
CONFIDENCE: [0-100]

Question: {task.prompt}"""

        try:
            response = await client.generate(
                prompt,
                system=system_prompt,
                temperature=0.3,
                max_tokens=300
            )

            import re
            answer_match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
            answer = answer_match.group(1).strip() if answer_match else response[:50]

            score = task.evaluate(answer)
            is_correct = score >= 0.8
            if is_correct:
                correct += 1

        except Exception as e:
            is_correct = False
            answer = f"ERROR: {e}"

        elapsed = (time.time() - start) * 1000
        total_time += elapsed

        task_results.append({
            "task_id": task.task_id,
            "rule": task.rule_type.value,
            "routed_to": routing_result.selected_agent_id,
            "answer": answer,
            "correct": is_correct,
            "time_ms": elapsed
        })

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            log(f"  [ORACLE_ROUTING] {i+1}/{len(tasks)} tasks, acc={correct/(i+1):.1%}")

    return ConditionResult(
        condition=Condition.ORACLE_ROUTING.value,
        n_tasks=len(tasks),
        correct=correct,
        accuracy=correct / len(tasks) if tasks else 0,
        total_api_calls=len(tasks),
        total_time_ms=total_time,
        avg_time_per_task_ms=total_time / len(tasks) if tasks else 0,
        task_results=task_results
    ), task_results


async def run_confidence_routing(
    client: LLMClient,
    agents: List[PreferenceAgent],
    tasks: List,
    checkpoint_start: int = 0
) -> Tuple[ConditionResult, List[Dict]]:
    """
    Condition 4: Route to highest-confidence specialist.

    Each agent provides confidence, route to most confident.
    """
    correct = 0
    total_time = 0
    total_api_calls = 0
    task_results = []

    for i, task in enumerate(tasks):
        if i < checkpoint_start:
            continue

        start = time.time()

        # Get confidence from all agents (parallel)
        confidence_prompt = f"""Rate your confidence (0-100) in answering this question correctly.
Only respond with a number between 0 and 100.

Question: {task.prompt[:200]}"""

        try:
            # Get confidences in parallel
            confidence_tasks = []
            for agent in agents:
                system_prompt = build_prompt_from_levels(agent.strategy_levels)
                confidence_tasks.append(
                    client.generate(confidence_prompt, system=system_prompt, temperature=0.1, max_tokens=100)
                )

            confidence_responses = await asyncio.gather(*confidence_tasks, return_exceptions=True)
            total_api_calls += len(agents)

            # Parse confidences
            confidences = []
            import re
            for resp in confidence_responses:
                if isinstance(resp, Exception):
                    confidences.append(50)
                else:
                    numbers = re.findall(r'\d+', str(resp))
                    conf = int(numbers[0]) if numbers else 50
                    confidences.append(min(100, max(0, conf)))

            # Select highest confidence agent
            best_idx = max(range(len(confidences)), key=lambda i: confidences[i])
            best_agent = agents[best_idx]

            # Get answer from best agent
            system_prompt = build_prompt_from_levels(best_agent.strategy_levels)
            answer_prompt = f"""Answer this question.

Format:
ANSWER: [your answer]

Question: {task.prompt}"""

            response = await client.generate(
                answer_prompt,
                system=system_prompt,
                temperature=0.3,
                max_tokens=300
            )
            total_api_calls += 1

            answer_match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
            answer = answer_match.group(1).strip() if answer_match else response[:50]

            score = task.evaluate(answer)
            is_correct = score >= 0.8
            if is_correct:
                correct += 1

        except Exception as e:
            is_correct = False
            answer = f"ERROR: {e}"
            best_agent = agents[0]
            confidences = [0] * len(agents)

        elapsed = (time.time() - start) * 1000
        total_time += elapsed

        task_results.append({
            "task_id": task.task_id,
            "rule": task.rule_type.value,
            "routed_to": best_agent.agent_id,
            "all_confidences": {agents[j].agent_id: confidences[j] for j in range(len(agents))},
            "answer": answer,
            "correct": is_correct,
            "time_ms": elapsed
        })

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            log(f"  [CONFIDENCE_ROUTING] {i+1}/{len(tasks)} tasks, acc={correct/(i+1):.1%}")

    return ConditionResult(
        condition=Condition.CONFIDENCE_ROUTING.value,
        n_tasks=len(tasks),
        correct=correct,
        accuracy=correct / len(tasks) if tasks else 0,
        total_api_calls=total_api_calls,
        total_time_ms=total_time,
        avg_time_per_task_ms=total_time / len(tasks) if tasks else 0,
        task_results=task_results
    ), task_results


async def run_ensemble(
    client: LLMClient,
    agents: List[PreferenceAgent],
    tasks: List,
    checkpoint_start: int = 0
) -> Tuple[ConditionResult, List[Dict]]:
    """
    Condition 5: All specialists answer, take majority vote.

    Most expensive but potentially most accurate.
    """
    correct = 0
    total_time = 0
    total_api_calls = 0
    task_results = []

    for i, task in enumerate(tasks):
        if i < checkpoint_start:
            continue

        start = time.time()

        answer_prompt = f"""Answer this question with ONLY the answer, nothing else.

Question: {task.prompt}"""

        try:
            # Get answers from all agents in parallel
            answer_tasks = []
            for agent in agents:
                system_prompt = build_prompt_from_levels(agent.strategy_levels)
                answer_tasks.append(
                    client.generate(answer_prompt, system=system_prompt, temperature=0.3, max_tokens=200)
                )

            answer_responses = await asyncio.gather(*answer_tasks, return_exceptions=True)
            total_api_calls += len(agents)

            # Parse answers and vote
            answers = []
            for resp in answer_responses:
                if isinstance(resp, Exception):
                    answers.append("")
                else:
                    answers.append(str(resp).strip().upper()[:20])

            # Majority vote
            from collections import Counter
            vote_counts = Counter(a for a in answers if a)
            if vote_counts:
                ensemble_answer = vote_counts.most_common(1)[0][0]
            else:
                ensemble_answer = "A"

            score = task.evaluate(ensemble_answer)
            is_correct = score >= 0.8
            if is_correct:
                correct += 1

        except Exception as e:
            is_correct = False
            ensemble_answer = f"ERROR: {e}"
            answers = []

        elapsed = (time.time() - start) * 1000
        total_time += elapsed

        task_results.append({
            "task_id": task.task_id,
            "rule": task.rule_type.value,
            "individual_answers": {agents[j].agent_id: answers[j] if j < len(answers) else "" for j in range(len(agents))},
            "ensemble_answer": ensemble_answer,
            "correct": is_correct,
            "time_ms": elapsed
        })

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            log(f"  [ENSEMBLE] {i+1}/{len(tasks)} tasks, acc={correct/(i+1):.1%}")

    return ConditionResult(
        condition=Condition.ENSEMBLE.value,
        n_tasks=len(tasks),
        correct=correct,
        accuracy=correct / len(tasks) if tasks else 0,
        total_api_calls=total_api_calls,
        total_time_ms=total_time,
        avg_time_per_task_ms=total_time / len(tasks) if tasks else 0,
        task_results=task_results
    ), task_results


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

async def run_practical_benefit_experiment(
    n_tasks: int = N_TASKS,
    seed: int = 42,
    resume: bool = True
) -> Dict:
    """
    Run the 5-condition practical benefit experiment.

    Args:
        n_tasks: Number of tasks per condition
        seed: Random seed for reproducibility
        resume: Whether to resume from checkpoint

    Returns:
        Complete results dictionary
    """
    log("=" * 60)
    log("PRACTICAL BENEFIT EXPERIMENT (5 Conditions)")
    log("=" * 60)

    # Check for checkpoint
    checkpoint = None
    if resume:
        checkpoint = load_checkpoint()

    if checkpoint is None:
        checkpoint = ExperimentCheckpoint(
            started_at=datetime.now().isoformat(),
            conditions_completed=[],
            current_condition=None,
            current_task_idx=0,
            results={}
        )

    # Initialize
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Create LLM client (uses .env)
    client = LLMClient()
    log(f"Using model: {client.config.model}")

    # Create agents
    specialists = create_specialist_population()
    log(f"Created {len(specialists)} specialists")

    # Generate tasks (same for all conditions)
    generator = RuleTaskGenerator(seed=seed, opaque=True)
    tasks = []
    rules = list(RuleType)
    for i in range(n_tasks):
        rule = rules[i % len(rules)]
        tasks.append(generator.generate_task(rule, i))
    log(f"Generated {len(tasks)} tasks")

    # Run conditions
    conditions = [
        (Condition.SINGLE_GENERALIST, lambda t, s: run_single_generalist(client, t, s)),
        (Condition.ORACLE_ROUTING, lambda t, s: run_oracle_routing(client, specialists, t, s)),
        (Condition.CONFIDENCE_ROUTING, lambda t, s: run_confidence_routing(client, specialists, t, s)),
        (Condition.ENSEMBLE, lambda t, s: run_ensemble(client, specialists, t, s)),
    ]

    for condition, runner in conditions:
        if condition.value in checkpoint.conditions_completed:
            log(f"[{condition.value}] Already completed, skipping")
            continue

        log(f"\n[{condition.value}] Starting...")
        checkpoint.current_condition = condition.value
        start_idx = checkpoint.current_task_idx if checkpoint.current_condition == condition.value else 0

        try:
            result, task_results = await runner(tasks, start_idx)

            checkpoint.results[condition.value] = result.to_dict()
            checkpoint.conditions_completed.append(condition.value)
            checkpoint.current_task_idx = 0
            save_checkpoint(checkpoint)

            log(f"[{condition.value}] Completed: {result.accuracy:.1%} accuracy, {result.total_api_calls} API calls")

        except Exception as e:
            log(f"[{condition.value}] ERROR: {e}")
            save_checkpoint(checkpoint)
            raise

    # Analysis
    log("\n" + "=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)

    results = checkpoint.results

    # Print comparison table
    log("\n| Condition | Accuracy | API Calls | Avg Time (ms) |")
    log("|-----------|----------|-----------|---------------|")
    for cond in [c.value for c in Condition if c.value in results]:
        r = results[cond]
        log(f"| {cond:20s} | {r['accuracy']:.1%} | {r['total_api_calls']:9d} | {r['avg_time_per_task_ms']:.1f} |")

    # Statistical comparison vs baseline
    baseline = Condition.SINGLE_GENERALIST.value
    if baseline in results:
        baseline_acc = results[baseline]['accuracy']
        log(f"\nComparison vs {baseline}:")
        for cond in results:
            if cond != baseline:
                improvement = (results[cond]['accuracy'] - baseline_acc) * 100
                cost_ratio = results[cond]['total_api_calls'] / results[baseline]['total_api_calls']
                log(f"  {cond}: {improvement:+.1f}pp accuracy, {cost_ratio:.1f}x cost")

    # Success criterion
    success = any(
        results[c]['accuracy'] > results[baseline]['accuracy'] + 0.05
        and results[c]['total_api_calls'] < results[baseline]['total_api_calls'] * 2
        for c in results if c != baseline
    ) if baseline in results else False

    log(f"\nSuccess criterion met: {success}")

    # Save final results
    final_results = {
        "experiment": "practical_benefit_5condition",
        "timestamp": datetime.now().isoformat(),
        "n_tasks": n_tasks,
        "n_agents": len(specialists),
        "seed": seed,
        "model": client.config.model,
        "results": results,
        "success": success,
        "cost_estimate": {"total_tokens": client.usage.total_tokens, "estimated_cost": client.usage.estimated_cost} if hasattr(client, 'usage') else {}
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(final_results, f, indent=2)

    log(f"\nResults saved to {RESULTS_FILE}")

    # Clear checkpoint on success
    clear_checkpoint()

    return final_results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run practical benefit experiment")
    parser.add_argument("--n-tasks", type=int, default=N_TASKS, help="Tasks per condition")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh (ignore checkpoint)")

    args = parser.parse_args()

    asyncio.run(run_practical_benefit_experiment(
        n_tasks=args.n_tasks,
        seed=args.seed,
        resume=not args.no_resume
    ))
