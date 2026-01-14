"""
Counterfactual validation for causal claims.

The key test: Prompt Swap
If prompts cause specialization, performance should follow the prompt,
not the agent identity.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import asyncio

from .agent import GenesisAgent
from .tasks import Task, TaskType, TaskPool


@dataclass
class SwapTestResult:
    """Result of a prompt swap test."""
    # Baseline performance (original prompts)
    baseline: Dict[str, Dict[str, float]]

    # Swapped performance (after exchanging prompts)
    swapped: Dict[str, Dict[str, float]]

    # Transfer coefficient: how much does performance follow the prompt?
    transfer_coefficient: float

    # Detailed analysis
    analysis: str


async def run_prompt_swap_test(
    agents: List[GenesisAgent],
    task_pool: TaskPool,
    llm_client,
    n_eval_tasks: int = 10
) -> SwapTestResult:
    """
    Run the prompt swap test to validate causal claims.

    Protocol:
    1. Identify the strongest specialist for each task type
    2. Evaluate their performance on all task types (baseline)
    3. Swap prompts between two specialists
    4. Re-evaluate performance (swapped)
    5. Compute transfer coefficient

    If prompts cause performance, swapped performance should follow
    the prompt, not the original agent.

    Args:
        agents: Population of trained agents
        task_pool: Pool of evaluation tasks
        llm_client: LLM client for evaluation
        n_eval_tasks: Number of tasks per type for evaluation

    Returns:
        SwapTestResult with baseline, swapped, and transfer coefficient
    """
    from .competition import CompetitionEngine

    # 1. Identify specialists
    specialists = {}
    for task_type in TaskType:
        best_agent = max(
            agents,
            key=lambda a: a.get_performance_by_type().get(task_type.value, 0)
        )
        specialists[task_type.value] = best_agent

    # Pick two specialists to swap (math and language for contrast)
    if "math" in specialists and "language" in specialists:
        agent_a = specialists["math"]
        agent_b = specialists["language"]
        type_a = "math"
        type_b = "language"
    else:
        # Fallback: use first two specialists
        types = list(specialists.keys())[:2]
        agent_a = specialists[types[0]]
        agent_b = specialists[types[1]]
        type_a = types[0]
        type_b = types[1]

    # 2. Get evaluation tasks
    tasks_a = task_pool.sample(n_eval_tasks, TaskType(type_a))
    tasks_b = task_pool.sample(n_eval_tasks, TaskType(type_b))

    # 3. Baseline evaluation (original prompts)
    engine = CompetitionEngine(llm_client, evolution_enabled=False)

    baseline = {
        "agent_a": {
            "on_type_a": await _evaluate_agent(agent_a, tasks_a, engine),
            "on_type_b": await _evaluate_agent(agent_a, tasks_b, engine),
        },
        "agent_b": {
            "on_type_a": await _evaluate_agent(agent_b, tasks_a, engine),
            "on_type_b": await _evaluate_agent(agent_b, tasks_b, engine),
        }
    }

    # 4. Swap prompts
    prompt_a = agent_a.system_prompt
    prompt_b = agent_b.system_prompt
    agent_a.system_prompt = prompt_b
    agent_b.system_prompt = prompt_a

    # 5. Swapped evaluation
    swapped = {
        "agent_a_with_prompt_b": {
            "on_type_a": await _evaluate_agent(agent_a, tasks_a, engine),
            "on_type_b": await _evaluate_agent(agent_a, tasks_b, engine),
        },
        "agent_b_with_prompt_a": {
            "on_type_a": await _evaluate_agent(agent_b, tasks_a, engine),
            "on_type_b": await _evaluate_agent(agent_b, tasks_b, engine),
        }
    }

    # 6. Restore original prompts
    agent_a.system_prompt = prompt_a
    agent_b.system_prompt = prompt_b

    # 7. Compute transfer coefficient
    transfer_coef = _compute_transfer_coefficient(baseline, swapped, type_a, type_b)

    # 8. Generate analysis
    analysis = _generate_analysis(baseline, swapped, transfer_coef, type_a, type_b)

    return SwapTestResult(
        baseline=baseline,
        swapped=swapped,
        transfer_coefficient=transfer_coef,
        analysis=analysis
    )


async def _evaluate_agent(
    agent: GenesisAgent,
    tasks: List[Task],
    engine
) -> float:
    """Evaluate an agent on a set of tasks, return mean score."""
    scores = []
    for task in tasks:
        response = await engine.get_response(agent, task)
        score = task.evaluate(response)
        scores.append(score)
    return float(np.mean(scores))


def _compute_transfer_coefficient(
    baseline: Dict,
    swapped: Dict,
    type_a: str,
    type_b: str
) -> float:
    """
    Compute how much performance follows the prompt vs agent identity.

    Transfer coefficient interpretation:
    - 0: Performance follows agent identity (prompts don't matter)
    - 1: Performance follows prompt completely
    - 0.5: Equal influence of prompt and agent identity

    Returns:
        float: Transfer coefficient between 0 and 1
    """
    # Baseline: Agent A is good at type_a, Agent B is good at type_b
    # After swap: Agent A has prompt_b, Agent B has prompt_a

    # If prompts matter:
    # - Agent A (with prompt_b) should now be good at type_b
    # - Agent B (with prompt_a) should now be good at type_a

    # Change in Agent A's type_a performance
    a_type_a_baseline = baseline["agent_a"]["on_type_a"]
    a_type_a_swapped = swapped["agent_a_with_prompt_b"]["on_type_a"]

    # Change in Agent A's type_b performance
    a_type_b_baseline = baseline["agent_a"]["on_type_b"]
    a_type_b_swapped = swapped["agent_a_with_prompt_b"]["on_type_b"]

    # Change in Agent B's type_a performance
    b_type_a_baseline = baseline["agent_b"]["on_type_a"]
    b_type_a_swapped = swapped["agent_b_with_prompt_a"]["on_type_a"]

    # Change in Agent B's type_b performance
    b_type_b_baseline = baseline["agent_b"]["on_type_b"]
    b_type_b_swapped = swapped["agent_b_with_prompt_a"]["on_type_b"]

    # If prompts transfer:
    # - Agent A should get worse at type_a (lost specialized prompt)
    # - Agent A should get better at type_b (gained specialized prompt)
    # - Agent B should get better at type_a (gained specialized prompt)
    # - Agent B should get worse at type_b (lost specialized prompt)

    expected_changes = [
        a_type_a_swapped < a_type_a_baseline,  # A worse at A's specialty
        a_type_b_swapped > a_type_b_baseline,  # A better at B's specialty
        b_type_a_swapped > b_type_a_baseline,  # B better at A's specialty
        b_type_b_swapped < b_type_b_baseline,  # B worse at B's specialty
    ]

    # Simple transfer coefficient: fraction of expected changes that occurred
    transfer_coef = sum(expected_changes) / len(expected_changes)

    return transfer_coef


def _generate_analysis(
    baseline: Dict,
    swapped: Dict,
    transfer_coef: float,
    type_a: str,
    type_b: str
) -> str:
    """Generate human-readable analysis of swap test results."""

    analysis = f"""
PROMPT SWAP TEST ANALYSIS
=========================

Specialists: {type_a} specialist (Agent A) vs {type_b} specialist (Agent B)

BASELINE PERFORMANCE (Original Prompts):
- Agent A on {type_a}: {baseline['agent_a']['on_type_a']:.3f}
- Agent A on {type_b}: {baseline['agent_a']['on_type_b']:.3f}
- Agent B on {type_a}: {baseline['agent_b']['on_type_a']:.3f}
- Agent B on {type_b}: {baseline['agent_b']['on_type_b']:.3f}

SWAPPED PERFORMANCE (Exchanged Prompts):
- Agent A (with {type_b} prompt) on {type_a}: {swapped['agent_a_with_prompt_b']['on_type_a']:.3f}
- Agent A (with {type_b} prompt) on {type_b}: {swapped['agent_a_with_prompt_b']['on_type_b']:.3f}
- Agent B (with {type_a} prompt) on {type_a}: {swapped['agent_b_with_prompt_a']['on_type_a']:.3f}
- Agent B (with {type_a} prompt) on {type_b}: {swapped['agent_b_with_prompt_a']['on_type_b']:.3f}

TRANSFER COEFFICIENT: {transfer_coef:.2f}

INTERPRETATION:
"""

    if transfer_coef >= 0.75:
        analysis += "Strong evidence that prompts CAUSE specialization. Performance follows prompts, not agent identity."
    elif transfer_coef >= 0.5:
        analysis += "Moderate evidence that prompts influence specialization. Both prompts and agent identity contribute."
    else:
        analysis += "Weak evidence for prompt causality. Agent identity may matter more than prompt content."

    return analysis
