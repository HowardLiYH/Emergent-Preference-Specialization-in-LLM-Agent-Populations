"""
Temperature Sensitivity Analysis for All 8 Rules

Extends temperature sensitivity testing to all 8 rules.
Temperatures: 0.1, 0.3, 0.5, 0.7, 0.9, 1.0

Ground Rules Applied:
- Checkpoints for long experiments
- API keys from .env only
- Unified model across tests
"""

import os
import sys
import json
import asyncio
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genesis.llm_client import LLMClient
from src.genesis.synthetic_rules import RuleType, RuleTaskGenerator
from src.genesis.preference_agent import create_handcrafted_specialist
from src.genesis.rule_strategies import build_prompt_from_levels


# =============================================================================
# CONFIGURATION
# =============================================================================

CHECKPOINT_DIR = Path("results/temperature_all")
CHECKPOINT_FILE = CHECKPOINT_DIR / "checkpoint.json"
RESULTS_FILE = CHECKPOINT_DIR / "temperature_results.json"
LOG_FILE = CHECKPOINT_DIR / "temperature_log.txt"

TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
N_TASKS_PER_CONDITION = 10
RULES = list(RuleType)


@dataclass
class TemperatureResult:
    """Result for one rule at one temperature."""
    rule: str
    temperature: float
    n_tasks: int
    correct: int
    accuracy: float

    def to_dict(self) -> Dict:
        return asdict(self)


def log(message: str):
    """Log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(log_line + "\n")


def save_checkpoint(completed: Dict, results: Dict):
    """Save checkpoint."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {"completed": completed, "results": results}
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint() -> Optional[Dict]:
    """Load checkpoint if exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None


async def test_rule_at_temperature(
    client: LLMClient,
    rule: RuleType,
    temperature: float,
    n_tasks: int = N_TASKS_PER_CONDITION,
    seed: int = 42
) -> TemperatureResult:
    """Test a specialist on their rule at given temperature."""

    # Create specialist
    specialist = create_handcrafted_specialist(rule, f"specialist_{rule.value}")
    system_prompt = build_prompt_from_levels(specialist.strategy_levels)

    # Generate tasks for this rule
    generator = RuleTaskGenerator(seed=seed)
    tasks = []
    for _ in range(n_tasks * 3):
        task = generator.generate()
        if task.rule_type == rule:
            tasks.append(task)
        if len(tasks) >= n_tasks:
            break

    # Fill if needed
    while len(tasks) < n_tasks:
        task = generator.generate(rule_type=rule)
        tasks.append(task)

    tasks = tasks[:n_tasks]

    # Test
    correct = 0
    for task in tasks:
        prompt = f"Answer with only the letter (A, B, C, or D).\n\nQuestion: {task.prompt}"
        try:
            response = await client.generate(
                prompt,
                system=system_prompt,
                temperature=temperature,
                max_tokens=20
            )
            answer = response.strip().upper()[:1]
            if task.evaluate(answer) >= 0.8:
                correct += 1
        except Exception as e:
            pass

    return TemperatureResult(
        rule=rule.value,
        temperature=temperature,
        n_tasks=n_tasks,
        correct=correct,
        accuracy=correct / n_tasks
    )


async def run_temperature_experiment(
    n_tasks: int = N_TASKS_PER_CONDITION,
    resume: bool = True
) -> Dict:
    """Run full temperature sensitivity experiment."""

    log("=" * 60)
    log("TEMPERATURE SENSITIVITY ANALYSIS (ALL 8 RULES)")
    log("=" * 60)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint = load_checkpoint() if resume else None
    completed = checkpoint.get("completed", {}) if checkpoint else {}
    results = checkpoint.get("results", {}) if checkpoint else {}

    client = LLMClient()
    log(f"Using model: {client.config.model}")

    # Test each rule at each temperature
    for rule in RULES:
        for temp in TEMPERATURES:
            key = f"{rule.value}_T{temp}"

            if key in completed:
                log(f"[{key}] Already completed, skipping")
                continue

            log(f"[{rule.value}] Temperature {temp}...")

            try:
                result = await test_rule_at_temperature(
                    client, rule, temp, n_tasks
                )

                results[key] = result.to_dict()
                completed[key] = True
                save_checkpoint(completed, results)

                log(f"  Accuracy: {result.accuracy:.1%}")

            except Exception as e:
                log(f"  ERROR: {e}")
                save_checkpoint(completed, results)

    # Summary
    log("\n" + "=" * 60)
    log("RESULTS SUMMARY (Heatmap)")
    log("=" * 60)

    # Create heatmap data
    log("\n| Rule | T=0.1 | T=0.3 | T=0.5 | T=0.7 | T=0.9 | T=1.0 |")
    log("|------|-------|-------|-------|-------|-------|-------|")

    for rule in RULES:
        row = f"| {rule.value:12s} |"
        for temp in TEMPERATURES:
            key = f"{rule.value}_T{temp}"
            if key in results:
                acc = results[key]["accuracy"]
                row += f" {acc:.0%} |"
            else:
                row += " N/A |"
        log(row)

    # Statistical summary
    log("\n## Statistical Summary")
    for temp in TEMPERATURES:
        accs = [results[f"{r.value}_T{temp}"]["accuracy"]
                for r in RULES if f"{r.value}_T{temp}" in results]
        if accs:
            mean_acc = sum(accs) / len(accs)
            log(f"T={temp}: Mean accuracy = {mean_acc:.1%}")

    # Temperature effect
    low_temp_accs = [results[f"{r.value}_T0.1"]["accuracy"]
                     for r in RULES if f"{r.value}_T0.1" in results]
    high_temp_accs = [results[f"{r.value}_T1.0"]["accuracy"]
                      for r in RULES if f"{r.value}_T1.0" in results]

    if low_temp_accs and high_temp_accs:
        diff = sum(low_temp_accs)/len(low_temp_accs) - sum(high_temp_accs)/len(high_temp_accs)
        log(f"\nTemperature effect (T=0.1 vs T=1.0): {diff:+.1%}")

        if abs(diff) < 0.05:
            conclusion = "ROBUST: Specialization is robust across temperatures"
        else:
            conclusion = f"SENSITIVE: Performance varies by {abs(diff):.1%} across temperatures"
        log(f"Conclusion: {conclusion}")

    # Save final results
    final_results = {
        "experiment": "temperature_all_rules",
        "timestamp": datetime.now().isoformat(),
        "temperatures": TEMPERATURES,
        "rules": [r.value for r in RULES],
        "n_tasks_per_condition": n_tasks,
        "model": client.config.model,
        "results": results
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(final_results, f, indent=2)

    log(f"\nResults saved to {RESULTS_FILE}")

    return final_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tasks", type=int, default=10)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    asyncio.run(run_temperature_experiment(
        n_tasks=args.n_tasks,
        resume=not args.no_resume
    ))
