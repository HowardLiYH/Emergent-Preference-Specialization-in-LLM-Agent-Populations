"""
Specialization explanation generator.

Generates human-readable explanations for why agents specialized.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class SpecializationExplanation:
    """Explanation of an agent's specialization."""
    agent_id: str
    specialty: str
    tool_level: int
    win_rate: float
    total_wins: int
    first_win_generation: Optional[int]
    key_memory: Optional[str]
    explanation_text: str


class ExplanationGenerator:
    """
    Generates human-readable explanations for agent specialization.

    Explains WHY an agent specialized in a particular regime,
    including their journey and key learnings.
    """

    TEMPLATE = """Agent {agent_id} specialized in {specialty} because:

1. **Performance**: Achieved {win_rate:.1%} win rate with {total_wins} total wins
2. **Tool Level**: Operates at L{tool_level} ({tool_description})
3. **First Success**: Won first competition in generation {first_win}
4. **Key Strategy**: {key_memory}

The specialization emerged through competitive selection - this agent
consistently outperformed others on {specialty} tasks, accumulating
expertise while other agents found different niches."""

    TOOL_DESCRIPTIONS = {
        0: "Base LLM only",
        1: "Python execution",
        2: "Vision/multi-modal",
        3: "RAG/retrieval",
        4: "Web access",
    }

    def __init__(self, llm_client=None):
        """
        Initialize explanation generator.

        Args:
            llm_client: Optional LLM client for enhanced explanations
        """
        self.llm_client = llm_client

    def generate(
        self,
        agent_id: str,
        specialty: str,
        tool_level: int,
        win_rate: float,
        total_wins: int,
        first_win_generation: Optional[int] = None,
        key_memory: Optional[str] = None,
        enhanced: bool = False
    ) -> SpecializationExplanation:
        """
        Generate an explanation for agent specialization.

        Args:
            agent_id: Agent identifier
            specialty: Regime the agent specializes in
            tool_level: Agent's tool level
            win_rate: Agent's win rate
            total_wins: Total number of wins
            first_win_generation: When the agent first won
            key_memory: Most important memory entry
            enhanced: Whether to use LLM for richer explanation

        Returns:
            SpecializationExplanation object
        """
        tool_description = self.TOOL_DESCRIPTIONS.get(tool_level, f"L{tool_level}")

        explanation_text = self.TEMPLATE.format(
            agent_id=agent_id,
            specialty=specialty,
            win_rate=win_rate,
            total_wins=total_wins,
            tool_level=tool_level,
            tool_description=tool_description,
            first_win=first_win_generation or "early",
            key_memory=key_memory or "Accumulated through competitive experience"
        )

        if enhanced and self.llm_client:
            explanation_text = self._enhance_explanation(
                agent_id, specialty, tool_level, win_rate,
                total_wins, first_win_generation, key_memory
            )

        return SpecializationExplanation(
            agent_id=agent_id,
            specialty=specialty,
            tool_level=tool_level,
            win_rate=win_rate,
            total_wins=total_wins,
            first_win_generation=first_win_generation,
            key_memory=key_memory,
            explanation_text=explanation_text
        )

    def _enhance_explanation(
        self,
        agent_id: str,
        specialty: str,
        tool_level: int,
        win_rate: float,
        total_wins: int,
        first_win_generation: Optional[int],
        key_memory: Optional[str]
    ) -> str:
        """Generate an enhanced explanation using LLM."""
        prompt = f"""Generate a detailed explanation of how an AI agent specialized:

Agent ID: {agent_id}
Specialty: {specialty}
Tool Level: L{tool_level}
Win Rate: {win_rate:.1%}
Total Wins: {total_wins}
First Win: Generation {first_win_generation or 'unknown'}
Key Memory: {key_memory or 'None recorded'}

Write a 2-3 paragraph explanation that:
1. Describes the agent's journey to specialization
2. Explains why this specialty emerged (competitive dynamics)
3. Summarizes the agent's key learned strategies

Keep it factual and grounded in the provided data."""

        try:
            response = self.llm_client.generate(prompt)
            return response.text
        except Exception:
            # Fall back to template
            return self.TEMPLATE.format(
                agent_id=agent_id,
                specialty=specialty,
                win_rate=win_rate,
                total_wins=total_wins,
                tool_level=tool_level,
                tool_description=self.TOOL_DESCRIPTIONS.get(tool_level, ""),
                first_win=first_win_generation or "early",
                key_memory=key_memory or "Accumulated through experience"
            )

    def explain_population(
        self,
        agents: List[Dict[str, Any]]
    ) -> List[SpecializationExplanation]:
        """
        Generate explanations for all agents in a population.

        Args:
            agents: List of agent data dicts

        Returns:
            List of SpecializationExplanation objects
        """
        explanations = []

        for agent in agents:
            explanation = self.generate(
                agent_id=agent.get('id', 'unknown'),
                specialty=agent.get('specialty', 'unknown'),
                tool_level=agent.get('tool_level', 0),
                win_rate=agent.get('win_rate', 0),
                total_wins=agent.get('total_wins', 0),
                first_win_generation=agent.get('first_win'),
                key_memory=agent.get('key_memory')
            )
            explanations.append(explanation)

        return explanations
