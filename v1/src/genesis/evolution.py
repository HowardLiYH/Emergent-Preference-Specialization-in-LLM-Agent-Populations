"""
Prompt evolution mechanisms for LLM agents.

Two evolution types:
1. Directed: LLM updates prompt based on specific success
2. Random: LLM makes arbitrary changes (baseline for ablation)
"""

from .agent import GenesisAgent
from .tasks import Task


async def evolve_prompt_directed(
    agent: GenesisAgent,
    task: Task,
    score: float,
    llm_client,
    temperature: float = 0.7
) -> str:
    """
    Directed evolution: Update prompt based on success.

    The LLM analyzes what made the agent successful and reinforces
    those skills in the updated prompt.

    Args:
        agent: The agent to evolve
        task: The task that triggered evolution
        score: The score achieved on the task
        llm_client: LLM API client
        temperature: Evolution temperature

    Returns:
        Updated system prompt
    """
    evolution_prompt = f"""You are an AI agent evolution system.

This agent just SUCCEEDED at a {task.task_type.value} task with score {score:.2f}.

The task was: {task.prompt[:200]}...

Current agent role description:
{agent.system_prompt}

Your job is to update the agent's role description to:
1. Reinforce the skills that led to this success
2. Become more specialized in {task.task_type.value} tasks
3. Add specific strategies or approaches for this task type
4. Maintain coherence and professionalism

Keep the new prompt under 300 words.
Output ONLY the new system prompt, nothing else."""

    response = await llm_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": evolution_prompt}],
        temperature=temperature,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()


async def evolve_prompt_random(
    agent: GenesisAgent,
    llm_client,
    temperature: float = 1.0
) -> str:
    """
    Random evolution: Make arbitrary changes to prompt.

    This is a baseline for ablation studies. The LLM modifies
    the prompt in random ways without considering task success.

    Args:
        agent: The agent to evolve
        llm_client: LLM API client
        temperature: Higher temperature for more randomness

    Returns:
        Randomly modified system prompt
    """
    evolution_prompt = f"""Modify this AI agent's role description in a RANDOM way.

Make arbitrary changes:
- Add random skills or expertise
- Remove some existing capabilities
- Change the focus or approach
- Be creative and unpredictable

Current role description:
{agent.system_prompt}

The changes should be significant but keep the prompt coherent.
Keep the new prompt under 300 words.
Output ONLY the new system prompt, nothing else."""

    response = await llm_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": evolution_prompt}],
        temperature=temperature,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()


async def evolve_prompt_minimal(
    agent: GenesisAgent,
    task: Task,
    score: float,
    llm_client
) -> str:
    """
    Minimal evolution: Add a single sentence about success.

    Used for analysis of how much evolution is needed.

    Returns:
        Minimally modified prompt
    """
    addition = f" I am particularly skilled at {task.task_type.value} tasks."

    if addition not in agent.system_prompt:
        return agent.system_prompt + addition
    return agent.system_prompt


def create_evolution_function(
    llm_client,
    evolution_type: str = "directed",
    temperature: float = 0.7
):
    """
    Factory function to create evolution functions.

    Args:
        llm_client: LLM API client
        evolution_type: "directed", "random", or "minimal"
        temperature: LLM temperature

    Returns:
        Async evolution function
    """
    async def evolve(agent: GenesisAgent, task: Task, score: float) -> str:
        if evolution_type == "directed":
            return await evolve_prompt_directed(
                agent, task, score, llm_client, temperature
            )
        elif evolution_type == "random":
            return await evolve_prompt_random(agent, llm_client, temperature)
        elif evolution_type == "minimal":
            return await evolve_prompt_minimal(agent, task, score, llm_client)
        else:
            raise ValueError(f"Unknown evolution type: {evolution_type}")

    return evolve
