"""
Competition engine for LLM agent tournaments.

Implements winner-take-all dynamics where agents compete on tasks
and only the winner evolves their prompt.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import asyncio
import numpy as np

from .agent import GenesisAgent
from .tasks import Task, TaskType


@dataclass
class CompetitionResult:
    """Result of a single competition round."""
    task: Task
    winner: GenesisAgent
    winner_score: float
    all_scores: Dict[str, float]  # agent_id -> score
    all_responses: Dict[str, str]  # agent_id -> response


class CompetitionEngine:
    """
    Engine for running agent competitions.

    Implements configurable competition modes:
    - evolution_enabled: Whether winners evolve prompts
    - winner_takes_all: Only winner evolves vs all agents evolve
    """

    def __init__(
        self,
        llm_client,
        evolution_enabled: bool = True,
        winner_takes_all: bool = True,
        temperature: float = 0.7
    ):
        """
        Initialize competition engine.

        Args:
            llm_client: Client for LLM API calls
            evolution_enabled: Whether to evolve prompts after wins
            winner_takes_all: If True, only winner evolves; else all evolve
            temperature: LLM temperature for responses
        """
        self.llm = llm_client
        self.evolution_enabled = evolution_enabled
        self.winner_takes_all = winner_takes_all
        self.temperature = temperature

    async def get_response(self, agent: GenesisAgent, task: Task) -> str:
        """Get agent's response to a task."""
        messages = [
            agent.to_openai_message(),
            {"role": "user", "content": task.prompt}
        ]

        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=self.temperature,
            max_tokens=500
        )

        return response.choices[0].message.content

    async def run_round(
        self,
        agents: List[GenesisAgent],
        task: Task,
        evolve_func=None
    ) -> CompetitionResult:
        """
        Run a single competition round.

        Args:
            agents: List of competing agents
            task: The task to compete on
            evolve_func: Optional evolution function (async)

        Returns:
            CompetitionResult with winner and scores
        """
        # 1. All agents attempt the task in parallel
        responses = await asyncio.gather(*[
            self.get_response(agent, task) for agent in agents
        ])

        # 2. Score each response
        scores = {}
        response_map = {}
        for agent, response in zip(agents, responses):
            score = task.evaluate(response)
            scores[agent.id] = score
            response_map[agent.id] = response

            # Record performance for all agents
            agent.record_performance(task.task_type.value, score)

        # 3. Determine winner (highest score, random tiebreak)
        max_score = max(scores.values())
        winners = [aid for aid, s in scores.items() if s == max_score]
        winner_id = np.random.choice(winners)
        winner = next(a for a in agents if a.id == winner_id)

        # 4. Evolve winner's prompt (if enabled)
        if self.evolution_enabled and evolve_func:
            if self.winner_takes_all:
                # Only winner evolves
                old_prompt = winner.system_prompt
                new_prompt = await evolve_func(winner, task, max_score)
                winner.system_prompt = new_prompt
                winner.record_evolution(old_prompt, new_prompt,
                                        task.task_type.value, max_score)
            else:
                # All agents evolve (ablation condition)
                for agent in agents:
                    old_prompt = agent.system_prompt
                    agent_score = scores[agent.id]
                    new_prompt = await evolve_func(agent, task, agent_score)
                    agent.system_prompt = new_prompt
                    agent.record_evolution(old_prompt, new_prompt,
                                          task.task_type.value, agent_score)

        return CompetitionResult(
            task=task,
            winner=winner,
            winner_score=max_score,
            all_scores=scores,
            all_responses=response_map
        )

    async def run_generation(
        self,
        agents: List[GenesisAgent],
        tasks: List[Task],
        evolve_func=None
    ) -> List[CompetitionResult]:
        """
        Run a full generation (multiple competition rounds).

        Args:
            agents: List of competing agents
            tasks: List of tasks for this generation
            evolve_func: Evolution function to use

        Returns:
            List of CompetitionResults
        """
        results = []
        for task in tasks:
            result = await self.run_round(agents, task, evolve_func)
            results.append(result)

        # Increment generation counter for all agents
        for agent in agents:
            agent.generation += 1

        return results


async def evaluate_language_task(
    llm_client,
    task: Task,
    response: str
) -> float:
    """
    Use LLM-as-judge to evaluate a language task response.

    Returns:
        float: Score between 0 and 1
    """
    judge_prompt = f"""You are evaluating an AI response to a task.

Task: {task.prompt}

Response: {response}

Rate the response on a scale of 1-10 based on:
- Relevance to the task
- Quality of writing
- Creativity and insight

Output only a single number from 1 to 10."""

    judge_response = await llm_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,
        max_tokens=10
    )

    try:
        score = int(judge_response.choices[0].message.content.strip())
        return score / 10.0
    except ValueError:
        return 0.5  # Default if parsing fails
