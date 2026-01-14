"""
Cost-Benefit Analysis Experiment

Panel Modification #6: Calculate break-even point for training investment.

Analysis:
- Training cost: X API calls over 100 generations
- Inference benefit: Y% accuracy improvement
- Retries saved: Z fewer corrections needed
- Break-even: After N tasks, specialized population is net positive

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

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genesis.llm_client import LLMClient, TokenUsage


# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = Path("results/cost_benefit")
RESULTS_FILE = RESULTS_DIR / "cost_benefit_analysis.json"


@dataclass
class CostBreakdown:
    """Cost breakdown for a condition."""
    api_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CostBenefitAnalysis:
    """Complete cost-benefit analysis."""
    # Training costs (evolution phase)
    training_generations: int
    training_api_calls: int
    training_tokens: int
    training_cost_usd: float
    training_time_hours: float

    # Inference metrics
    generalist_accuracy: float
    specialist_accuracy: float
    accuracy_improvement: float

    # Per-task costs
    generalist_cost_per_task: float
    specialist_cost_per_task: float

    # Break-even analysis
    accuracy_improvement_per_task: float
    cost_of_retry: float  # Cost of one retry due to error
    expected_retries_generalist: float  # Per 100 tasks
    expected_retries_specialist: float
    retries_saved_per_100_tasks: float
    value_per_retry_saved: float  # In USD (user time, downstream costs)

    # Break-even point
    break_even_tasks: int
    break_even_days: float  # At typical usage

    # ROI metrics
    roi_after_1000_tasks: float  # Return on investment
    payback_period_days: float

    interpretation: str

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# PRICING DATA
# =============================================================================

# Gemini 2.5 Flash pricing (as of 2025, free tier has limits)
PRICING = {
    "gemini-2.5-flash": {
        "input_per_1k": 0.0,  # Free tier
        "output_per_1k": 0.0,
        "paid_input_per_1k": 0.00015,  # Paid tier
        "paid_output_per_1k": 0.0006,
    },
    "gpt-4o-mini": {
        "input_per_1k": 0.00015,
        "output_per_1k": 0.0006,
    },
    "claude-3-haiku": {
        "input_per_1k": 0.00025,
        "output_per_1k": 0.00125,
    }
}

# Estimated token counts per operation
TOKEN_ESTIMATES = {
    "task_prompt": 150,      # Average input tokens per task
    "task_response": 50,     # Average output tokens per task
    "confidence_query": 80,  # Tokens for confidence routing
    "training_per_gen": 2000,  # Tokens per generation during training
}

# Business assumptions
BUSINESS_ASSUMPTIONS = {
    "value_per_correct_answer": 0.10,  # USD value of correct answer
    "cost_per_retry": 0.05,            # USD cost when need to retry
    "tasks_per_day_typical": 100,      # Typical daily usage
    "training_generations": 100,
    "population_size": 12,
}


# =============================================================================
# COST CALCULATIONS
# =============================================================================

def estimate_training_cost(
    n_generations: int = 100,
    population_size: int = 12,
    tasks_per_generation: int = 10,
    model: str = "gemini-2.5-flash"
) -> CostBreakdown:
    """
    Estimate cost of training (evolution) phase.

    During training:
    - Each generation: N agents compete on M tasks
    - Each competition: 1 task prompt + N agent responses
    - Plus: Fitness evaluation, strategy updates (minimal tokens)
    """
    # Tokens per generation
    input_per_gen = tasks_per_generation * TOKEN_ESTIMATES["task_prompt"]
    output_per_gen = tasks_per_generation * population_size * TOKEN_ESTIMATES["task_response"]

    # Total across training
    total_input = n_generations * input_per_gen
    total_output = n_generations * output_per_gen
    total_tokens = total_input + total_output

    # API calls: each agent responds to each task
    total_calls = n_generations * tasks_per_generation * population_size

    # Cost calculation
    pricing = PRICING.get(model, PRICING["gemini-2.5-flash"])
    cost = (
        (total_input / 1000) * pricing.get("paid_input_per_1k", 0) +
        (total_output / 1000) * pricing.get("paid_output_per_1k", 0)
    )

    return CostBreakdown(
        api_calls=total_calls,
        input_tokens=total_input,
        output_tokens=total_output,
        total_tokens=total_tokens,
        estimated_cost_usd=cost
    )


def estimate_inference_cost(
    n_tasks: int,
    method: str = "single",  # "single", "oracle", "confidence", "ensemble"
    population_size: int = 12,
    model: str = "gemini-2.5-flash"
) -> CostBreakdown:
    """
    Estimate inference cost for different methods.

    Methods:
    - single: 1 call per task (generalist)
    - oracle: 1 call per task (perfect routing)
    - confidence: N+1 calls per task (confidence probe + answer)
    - ensemble: N calls per task (all agents answer)
    """
    if method == "single" or method == "oracle":
        calls_per_task = 1
    elif method == "confidence":
        calls_per_task = population_size + 1
    elif method == "ensemble":
        calls_per_task = population_size
    else:
        calls_per_task = 1

    total_calls = n_tasks * calls_per_task

    # Token estimates
    input_per_task = TOKEN_ESTIMATES["task_prompt"] * calls_per_task
    output_per_task = TOKEN_ESTIMATES["task_response"] * calls_per_task

    total_input = n_tasks * input_per_task
    total_output = n_tasks * output_per_task

    pricing = PRICING.get(model, PRICING["gemini-2.5-flash"])
    cost = (
        (total_input / 1000) * pricing.get("paid_input_per_1k", 0) +
        (total_output / 1000) * pricing.get("paid_output_per_1k", 0)
    )

    return CostBreakdown(
        api_calls=total_calls,
        input_tokens=total_input,
        output_tokens=total_output,
        total_tokens=total_input + total_output,
        estimated_cost_usd=cost
    )


def calculate_break_even(
    training_cost: float,
    generalist_accuracy: float,
    specialist_accuracy: float,
    generalist_cost_per_task: float,
    specialist_cost_per_task: float,
    value_per_correct: float,
    cost_per_retry: float
) -> Dict:
    """
    Calculate break-even point.

    Break-even occurs when:
    cumulative_benefit > training_cost

    Where benefit per task = (accuracy improvement * value) + (retry savings)
    """
    # Accuracy improvement
    accuracy_improvement = specialist_accuracy - generalist_accuracy

    # Expected retries (assuming each error needs ~1.5 retries on average)
    retry_rate = 1.5
    generalist_retries_per_task = (1 - generalist_accuracy) * retry_rate
    specialist_retries_per_task = (1 - specialist_accuracy) * retry_rate
    retries_saved_per_task = generalist_retries_per_task - specialist_retries_per_task

    # Value per task
    accuracy_value = accuracy_improvement * value_per_correct
    retry_value = retries_saved_per_task * cost_per_retry

    # Extra inference cost for specialists (if using confidence routing)
    extra_inference_cost = specialist_cost_per_task - generalist_cost_per_task

    # Net benefit per task
    net_benefit_per_task = accuracy_value + retry_value - extra_inference_cost

    # Break-even point
    if net_benefit_per_task <= 0:
        break_even_tasks = float('inf')
    else:
        break_even_tasks = training_cost / net_benefit_per_task

    return {
        "accuracy_improvement": accuracy_improvement,
        "retries_saved_per_task": retries_saved_per_task,
        "accuracy_value_per_task": accuracy_value,
        "retry_value_per_task": retry_value,
        "extra_inference_cost_per_task": extra_inference_cost,
        "net_benefit_per_task": net_benefit_per_task,
        "break_even_tasks": int(break_even_tasks) if break_even_tasks != float('inf') else -1
    }


def run_cost_benefit_analysis(
    generalist_accuracy: float = 0.50,
    specialist_accuracy: float = 0.85,
    model: str = "gemini-2.5-flash",
    routing_method: str = "oracle"
) -> CostBenefitAnalysis:
    """
    Run complete cost-benefit analysis.

    Args:
        generalist_accuracy: Baseline accuracy without specialization
        specialist_accuracy: Accuracy with specialized population
        model: LLM model being used
        routing_method: How tasks are routed to specialists
    """
    # Training costs
    training = estimate_training_cost(
        n_generations=BUSINESS_ASSUMPTIONS["training_generations"],
        population_size=BUSINESS_ASSUMPTIONS["population_size"],
        model=model
    )

    # Inference costs (per 100 tasks)
    generalist_inference = estimate_inference_cost(100, "single", model=model)
    specialist_inference = estimate_inference_cost(
        100, routing_method,
        population_size=BUSINESS_ASSUMPTIONS["population_size"],
        model=model
    )

    # Break-even calculation
    break_even = calculate_break_even(
        training_cost=training.estimated_cost_usd,
        generalist_accuracy=generalist_accuracy,
        specialist_accuracy=specialist_accuracy,
        generalist_cost_per_task=generalist_inference.estimated_cost_usd / 100,
        specialist_cost_per_task=specialist_inference.estimated_cost_usd / 100,
        value_per_correct=BUSINESS_ASSUMPTIONS["value_per_correct_answer"],
        cost_per_retry=BUSINESS_ASSUMPTIONS["cost_per_retry"]
    )

    # Time estimates
    typical_tasks_per_day = BUSINESS_ASSUMPTIONS["tasks_per_day_typical"]
    break_even_days = break_even["break_even_tasks"] / typical_tasks_per_day if break_even["break_even_tasks"] > 0 else -1

    # ROI after 1000 tasks
    if break_even["net_benefit_per_task"] > 0:
        total_benefit_1000 = 1000 * break_even["net_benefit_per_task"]
        roi_1000 = (total_benefit_1000 - training.estimated_cost_usd) / training.estimated_cost_usd
    else:
        roi_1000 = -1.0

    # Interpretation
    if break_even["break_even_tasks"] <= 0:
        interpretation = (
            "NEGATIVE ROI: Specialist population costs more than it saves. "
            "Consider reducing routing overhead or increasing training efficiency."
        )
    elif break_even_days <= 7:
        interpretation = (
            f"EXCELLENT ROI: Break-even in ~{break_even_days:.1f} days ({break_even['break_even_tasks']} tasks). "
            "Training investment pays off quickly."
        )
    elif break_even_days <= 30:
        interpretation = (
            f"GOOD ROI: Break-even in ~{break_even_days:.1f} days ({break_even['break_even_tasks']} tasks). "
            "Worthwhile for sustained usage."
        )
    elif break_even_days <= 90:
        interpretation = (
            f"MODERATE ROI: Break-even in ~{break_even_days:.1f} days ({break_even['break_even_tasks']} tasks). "
            "Consider for high-value applications."
        )
    else:
        interpretation = (
            f"LOW ROI: Break-even in ~{break_even_days:.1f} days ({break_even['break_even_tasks']} tasks). "
            "May not be cost-effective for most use cases."
        )

    # Training time estimate (assuming 1 second per API call)
    training_time_hours = training.api_calls / 3600

    return CostBenefitAnalysis(
        training_generations=BUSINESS_ASSUMPTIONS["training_generations"],
        training_api_calls=training.api_calls,
        training_tokens=training.total_tokens,
        training_cost_usd=training.estimated_cost_usd,
        training_time_hours=training_time_hours,

        generalist_accuracy=generalist_accuracy,
        specialist_accuracy=specialist_accuracy,
        accuracy_improvement=specialist_accuracy - generalist_accuracy,

        generalist_cost_per_task=generalist_inference.estimated_cost_usd / 100,
        specialist_cost_per_task=specialist_inference.estimated_cost_usd / 100,

        accuracy_improvement_per_task=break_even["accuracy_value_per_task"],
        cost_of_retry=BUSINESS_ASSUMPTIONS["cost_per_retry"],
        expected_retries_generalist=(1 - generalist_accuracy) * 1.5 * 100,
        expected_retries_specialist=(1 - specialist_accuracy) * 1.5 * 100,
        retries_saved_per_100_tasks=break_even["retries_saved_per_task"] * 100,
        value_per_retry_saved=BUSINESS_ASSUMPTIONS["cost_per_retry"],

        break_even_tasks=break_even["break_even_tasks"],
        break_even_days=break_even_days,

        roi_after_1000_tasks=roi_1000,
        payback_period_days=break_even_days,

        interpretation=interpretation
    )


def generate_cost_benefit_report(
    analysis: CostBenefitAnalysis,
    model: str = "gemini-2.5-flash"
) -> str:
    """Generate markdown report of cost-benefit analysis."""
    report = f"""# Cost-Benefit Analysis Report

## Executive Summary

{analysis.interpretation}

## Training Investment

| Metric | Value |
|--------|-------|
| Generations | {analysis.training_generations} |
| API Calls | {analysis.training_api_calls:,} |
| Total Tokens | {analysis.training_tokens:,} |
| Estimated Cost | ${analysis.training_cost_usd:.2f} |
| Training Time | {analysis.training_time_hours:.1f} hours |

## Performance Improvement

| Metric | Generalist | Specialist | Improvement |
|--------|------------|------------|-------------|
| Accuracy | {analysis.generalist_accuracy:.0%} | {analysis.specialist_accuracy:.0%} | +{analysis.accuracy_improvement:.0%} |
| Expected Retries (per 100 tasks) | {analysis.expected_retries_generalist:.1f} | {analysis.expected_retries_specialist:.1f} | -{analysis.retries_saved_per_100_tasks:.1f} |

## Cost Per Task

| Method | Cost/Task |
|--------|-----------|
| Generalist | ${analysis.generalist_cost_per_task:.4f} |
| Specialist | ${analysis.specialist_cost_per_task:.4f} |

## Break-Even Analysis

| Metric | Value |
|--------|-------|
| Break-Even Tasks | {analysis.break_even_tasks:,} |
| Break-Even Days | {analysis.break_even_days:.1f} |
| ROI after 1000 tasks | {analysis.roi_after_1000_tasks:.1%} |

## Recommendation

{analysis.interpretation}

## Assumptions

- Model: {model}
- Value per correct answer: ${BUSINESS_ASSUMPTIONS['value_per_correct_answer']:.2f}
- Cost per retry: ${BUSINESS_ASSUMPTIONS['cost_per_retry']:.2f}
- Tasks per day (typical): {BUSINESS_ASSUMPTIONS['tasks_per_day_typical']}

---
*Generated: {datetime.now().isoformat()}*
"""
    return report


# =============================================================================
# MAIN
# =============================================================================

def run_analysis_suite():
    """Run cost-benefit analysis for different scenarios."""
    print("=" * 60)
    print("COST-BENEFIT ANALYSIS")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Scenario 1: Oracle routing (best case)
    print("\n## Scenario 1: Oracle Routing (Best Case)")
    oracle_analysis = run_cost_benefit_analysis(
        generalist_accuracy=0.50,
        specialist_accuracy=0.85,
        routing_method="oracle"
    )
    print(f"Break-even: {oracle_analysis.break_even_tasks} tasks ({oracle_analysis.break_even_days:.1f} days)")
    print(f"Interpretation: {oracle_analysis.interpretation}")

    # Scenario 2: Confidence routing (realistic)
    print("\n## Scenario 2: Confidence Routing (Realistic)")
    confidence_analysis = run_cost_benefit_analysis(
        generalist_accuracy=0.50,
        specialist_accuracy=0.80,  # Slightly lower due to routing errors
        routing_method="confidence"
    )
    print(f"Break-even: {confidence_analysis.break_even_tasks} tasks ({confidence_analysis.break_even_days:.1f} days)")
    print(f"Interpretation: {confidence_analysis.interpretation}")

    # Scenario 3: Ensemble (expensive but accurate)
    print("\n## Scenario 3: Ensemble (Expensive)")
    ensemble_analysis = run_cost_benefit_analysis(
        generalist_accuracy=0.50,
        specialist_accuracy=0.90,  # Higher due to voting
        routing_method="ensemble"
    )
    print(f"Break-even: {ensemble_analysis.break_even_tasks} tasks ({ensemble_analysis.break_even_days:.1f} days)")
    print(f"Interpretation: {ensemble_analysis.interpretation}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "scenarios": {
            "oracle_routing": oracle_analysis.to_dict(),
            "confidence_routing": confidence_analysis.to_dict(),
            "ensemble": ensemble_analysis.to_dict()
        },
        "assumptions": BUSINESS_ASSUMPTIONS,
        "pricing": PRICING
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate report
    report = generate_cost_benefit_report(oracle_analysis)
    report_path = RESULTS_DIR / "cost_benefit_report.md"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Report saved to {report_path}")

    return results


if __name__ == "__main__":
    run_analysis_suite()
