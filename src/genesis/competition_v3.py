"""
Confidence-Based Competition for Synthetic Rules

Agents output both answer AND confidence.
Winner = highest confidence among correct answers.

This differentiates agents even when multiple get correct answers.
"""

import asyncio
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random

from .synthetic_rules import RuleTask, RuleType
from .preference_agent import PreferenceAgent
from .llm_client import LLMClient


@dataclass
class CompetitionResult:
    """Result of a single agent's attempt in a competition."""
    agent_id: str
    answer: str
    confidence: float  # 0.0 to 1.0
    is_correct: bool
    raw_response: str


def parse_answer_confidence(response: str) -> Tuple[str, float]:
    """
    Parse answer and confidence from LLM response.

    Expected format: "ANSWER: [answer] | CONFIDENCE: [0-100]%"
    Fallback parsing for various formats.
    """
    response = response.strip()

    # Try standard format
    answer_match = re.search(r'ANSWER:\s*([A-Da-d]|\w+)', response, re.IGNORECASE)
    confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', response, re.IGNORECASE)

    # Extract answer
    if answer_match:
        answer = answer_match.group(1).strip().upper()
    else:
        # Fallback: look for A, B, C, D at start
        first_letter = re.match(r'^([A-Da-d])\b', response)
        if first_letter:
            answer = first_letter.group(1).upper()
        else:
            # Last resort: first word
            answer = response.split()[0] if response.split() else ""

    # Extract confidence
    if confidence_match:
        confidence = min(100, max(0, int(confidence_match.group(1)))) / 100.0
    else:
        # Fallback: look for any percentage
        pct_match = re.search(r'(\d+)%', response)
        if pct_match:
            confidence = min(100, max(0, int(pct_match.group(1)))) / 100.0
        else:
            # Default to medium confidence
            confidence = 0.5

    return answer, confidence


async def run_competition(
    agents: List[PreferenceAgent],
    task: RuleTask,
    client: LLMClient,
    temperature: float = 0.3
) -> Tuple[Optional[str], List[CompetitionResult]]:
    """
    All agents compete on the same task.
    Winner = highest confidence among correct answers.

    Args:
        agents: List of competing agents
        task: The task to compete on
        client: LLM client for generation
        temperature: Generation temperature

    Returns:
        Tuple of (winner_agent_id or None, list of all results)
    """
    results = []

    # Create competition prompt
    competition_prompt = f"""{task.prompt}

Provide your answer AND your confidence level (0-100%).
Format: ANSWER: [your answer] | CONFIDENCE: [0-100]%"""

    # Run all agents
    for agent in agents:
        try:
            response = await client.generate(
                prompt=competition_prompt,
                system=agent.get_prompt(),
                temperature=temperature,
                max_tokens=100
            )

            answer, confidence = parse_answer_confidence(response)
            is_correct = task.evaluate(answer) > 0.5

            results.append(CompetitionResult(
                agent_id=agent.agent_id,
                answer=answer,
                confidence=confidence,
                is_correct=is_correct,
                raw_response=response
            ))

        except Exception as e:
            # On error, record as incorrect with 0 confidence
            results.append(CompetitionResult(
                agent_id=agent.agent_id,
                answer="ERROR",
                confidence=0.0,
                is_correct=False,
                raw_response=str(e)
            ))

    # Determine winner: highest confidence among correct answers
    correct_results = [r for r in results if r.is_correct]

    if correct_results:
        # Winner = highest confidence among correct
        winner = max(correct_results, key=lambda r: r.confidence)

        # Tiebreaker: if multiple have same confidence, random selection
        tied_winners = [r for r in correct_results if r.confidence == winner.confidence]
        if len(tied_winners) > 1:
            winner = random.choice(tied_winners)

        return winner.agent_id, results
    else:
        # No correct answers - no winner this round
        return None, results


async def run_competition_batch(
    agents: List[PreferenceAgent],
    tasks: List[RuleTask],
    client: LLMClient
) -> List[Tuple[Optional[str], List[CompetitionResult]]]:
    """Run multiple competitions in sequence."""
    results = []
    for task in tasks:
        result = await run_competition(agents, task, client)
        results.append(result)
    return results


def simulate_competition(
    agents: List[PreferenceAgent],
    task: RuleTask,
    base_correct_prob: float = 0.5
) -> Tuple[Optional[str], List[CompetitionResult]]:
    """
    Simulate a competition without LLM calls (for testing/baselines).

    Agents with higher strategy levels for this rule have higher success probability.
    """
    results = []

    for agent in agents:
        # Higher strategy level = higher success probability
        strategy_level = agent.get_strategy_level(task.rule_type)
        correct_prob = base_correct_prob + (strategy_level * 0.15)  # +15% per level
        correct_prob = min(0.95, correct_prob)

        is_correct = random.random() < correct_prob

        # Confidence correlates with strategy level and correctness
        if is_correct:
            confidence = 0.5 + (strategy_level * 0.1) + random.uniform(0, 0.2)
        else:
            confidence = 0.3 + random.uniform(0, 0.3)
        confidence = min(1.0, max(0.0, confidence))

        results.append(CompetitionResult(
            agent_id=agent.agent_id,
            answer="SIMULATED",
            confidence=confidence,
            is_correct=is_correct,
            raw_response="SIMULATED"
        ))

    # Determine winner
    correct_results = [r for r in results if r.is_correct]
    if correct_results:
        winner = max(correct_results, key=lambda r: r.confidence)
        return winner.agent_id, results
    else:
        return None, results
