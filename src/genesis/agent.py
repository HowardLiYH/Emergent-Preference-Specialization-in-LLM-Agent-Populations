"""
GenesisAgent: LLM agent with evolvable system prompt.

The system_prompt is the agent's "DNA" - it defines the agent's role,
expertise, and approach to tasks. Through competition and evolution,
this prompt becomes increasingly specialized.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import uuid


@dataclass
class PromptEvolutionEvent:
    """Record of a single prompt evolution."""
    generation: int
    old_prompt: str
    new_prompt: str
    trigger_task_type: str
    trigger_score: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GenesisAgent:
    """
    An LLM agent with an evolvable system prompt.

    Attributes:
        id: Unique identifier
        system_prompt: The agent's role definition (evolvable)
        generation: Current generation number
        performance_history: Task type -> list of scores
        prompt_history: Record of all prompt evolutions
    """
    id: str
    system_prompt: str
    generation: int = 0
    performance_history: Dict[str, List[float]] = field(default_factory=dict)
    prompt_history: List[PromptEvolutionEvent] = field(default_factory=list)

    @classmethod
    def create(cls, initial_prompt: str = None) -> 'GenesisAgent':
        """Create a new agent with optional initial prompt."""
        if initial_prompt is None:
            initial_prompt = (
                "I am a general-purpose AI assistant. "
                "I can help with various tasks including math, coding, logic, and language."
            )
        return cls(
            id=str(uuid.uuid4())[:8],
            system_prompt=initial_prompt
        )

    def get_performance_by_type(self) -> Dict[str, float]:
        """Get mean performance per task type."""
        return {
            task_type: np.mean(scores) if scores else 0.0
            for task_type, scores in self.performance_history.items()
        }

    def get_best_task_type(self) -> Optional[str]:
        """Get the task type with highest mean performance."""
        perf = self.get_performance_by_type()
        if not perf:
            return None
        return max(perf, key=perf.get)

    def record_performance(self, task_type: str, score: float) -> None:
        """Record performance on a task."""
        if task_type not in self.performance_history:
            self.performance_history[task_type] = []
        self.performance_history[task_type].append(score)

    def record_evolution(self, old_prompt: str, new_prompt: str,
                         task_type: str, score: float) -> None:
        """Record a prompt evolution event."""
        event = PromptEvolutionEvent(
            generation=self.generation,
            old_prompt=old_prompt,
            new_prompt=new_prompt,
            trigger_task_type=task_type,
            trigger_score=score
        )
        self.prompt_history.append(event)

    def to_openai_message(self) -> dict:
        """Convert to OpenAI chat message format."""
        return {"role": "system", "content": self.system_prompt}

    def __repr__(self) -> str:
        perf = self.get_performance_by_type()
        best = self.get_best_task_type()
        return (
            f"GenesisAgent(id={self.id}, gen={self.generation}, "
            f"best_type={best}, prompt_len={len(self.system_prompt)})"
        )


def create_population(n_agents: int, initial_prompt: str = None) -> List[GenesisAgent]:
    """Create a population of agents with identical initial prompts."""
    return [GenesisAgent.create(initial_prompt) for _ in range(n_agents)]
