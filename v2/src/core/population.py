"""
Population management for competitive specialization.

Manages a population of agents competing for wins.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import random
import logging
import math
import json
from pathlib import Path

from ..tools.agent import Agent
from ..regimes.config import RegimeConfig, REGIMES
from ..regimes.sampler import RegimeSampler
from ..safety.collusion import CollusionDetector

logger = logging.getLogger(__name__)


@dataclass
class CompetitionResult:
    """Result of a single competition round."""
    generation: int
    regime: str
    winner_id: Optional[str]
    winner_tool: Optional[str]
    participants: List[str]
    responses: Dict[str, str]
    confidences: Dict[str, float]
    correct: Dict[str, bool]


class Population:
    """
    Population of agents for competitive specialization.

    Implements:
    - Winner-take-all dynamics
    - Fitness sharing (1/sqrt(n) penalty)
    - Competition rounds
    - State persistence
    """

    def __init__(
        self,
        n_agents: int = 12,
        initial_tool_level: int = 0,
        fitness_sharing_gamma: float = 0.5,
        seed: Optional[int] = None,
        llm_client=None
    ):
        """
        Initialize population.

        Args:
            n_agents: Number of agents
            initial_tool_level: Starting tool level for all agents
            fitness_sharing_gamma: Fitness sharing exponent
            seed: Random seed for reproducibility
            llm_client: LLM client for agent responses
        """
        self.n_agents = n_agents
        self.fitness_sharing_gamma = fitness_sharing_gamma
        self.rng = random.Random(seed)
        self.llm_client = llm_client

        # Create agents
        self.agents: List[Agent] = [
            Agent(
                tool_level=initial_tool_level,
                agent_id=f"agent_{i}",
                llm_client=llm_client
            )
            for i in range(n_agents)
        ]

        # Regime sampler
        self.regime_sampler = RegimeSampler(seed=seed)

        # Collusion detector
        self.collusion_detector = CollusionDetector()

        # Competition history
        self.generation = 0
        self.history: List[CompetitionResult] = []

        # Specialist counts per regime (for fitness sharing)
        self._specialist_counts: Dict[str, int] = {}
        self._update_specialist_counts()

    def _update_specialist_counts(self) -> None:
        """Update specialist counts for fitness sharing."""
        self._specialist_counts = {}

        for agent in self.agents:
            specialty = agent.get_specialty()
            if specialty:
                self._specialist_counts[specialty] = \
                    self._specialist_counts.get(specialty, 0) + 1

    def get_fitness_penalty(self, regime: str) -> float:
        """
        Get fitness sharing penalty for a regime.

        Uses 1/n^gamma penalty where n is number of specialists.

        Args:
            regime: Regime name

        Returns:
            Penalty multiplier (0-1)
        """
        n = self._specialist_counts.get(regime, 1)
        return 1.0 / (n ** self.fitness_sharing_gamma)

    def run_competition(
        self,
        task: str,
        regime: str,
        evaluator: Optional[callable] = None
    ) -> CompetitionResult:
        """
        Run a single competition round.

        Args:
            task: Task description
            regime: Task regime
            evaluator: Function to evaluate responses (response, task) -> (correct, confidence)

        Returns:
            CompetitionResult with winner and details
        """
        self.generation += 1

        # Collect responses from all agents
        responses: Dict[str, str] = {}
        confidences: Dict[str, float] = {}
        correct: Dict[str, bool] = {}
        tools_used: Dict[str, str] = {}

        for agent in self.agents:
            try:
                # Agent selects tool and generates response
                tool = agent.select_tool(regime)
                result = agent.execute_with_tool(task, regime, tool)

                tools_used[agent.id] = tool

                if result.success:
                    responses[agent.id] = result.output
                    # Extract confidence from response or default
                    confidences[agent.id] = agent.get_confidence(regime)
                else:
                    responses[agent.id] = ""
                    confidences[agent.id] = 0.0

            except Exception as e:
                logger.warning(f"Agent {agent.id} failed: {e}")
                responses[agent.id] = ""
                confidences[agent.id] = 0.0

        # Evaluate responses
        if evaluator:
            for agent_id, response in responses.items():
                is_correct, conf = evaluator(response, task)
                correct[agent_id] = is_correct
                # Override confidence with evaluator's assessment if provided
                if conf is not None:
                    confidences[agent_id] = conf
        else:
            # Default: mark all non-empty responses as correct
            for agent_id, response in responses.items():
                correct[agent_id] = bool(response.strip())

        # Determine winner (highest confidence among correct responders)
        winner_id = None
        winner_tool = None
        best_score = -1

        for agent_id in responses:
            if correct.get(agent_id, False):
                # Apply fitness sharing penalty
                agent = self._get_agent(agent_id)
                agent_specialty = agent.get_specialty() if agent else None

                penalty = self.get_fitness_penalty(regime)
                score = confidences[agent_id] * penalty

                if score > best_score:
                    best_score = score
                    winner_id = agent_id
                    winner_tool = tools_used.get(agent_id)

        # Update agents
        for agent in self.agents:
            tool = tools_used.get(agent.id, 'L0')
            won = agent.id == winner_id
            response = responses.get(agent.id, "")

            agent.update_from_competition(regime, tool, won, response)

        # Update specialist counts
        self._update_specialist_counts()

        # Check for collusion
        if winner_id:
            self.collusion_detector.record_win(regime, winner_id)

        # Record result
        result = CompetitionResult(
            generation=self.generation,
            regime=regime,
            winner_id=winner_id,
            winner_tool=winner_tool,
            participants=[a.id for a in self.agents],
            responses=responses,
            confidences=confidences,
            correct=correct
        )

        self.history.append(result)

        logger.debug(
            f"Gen {self.generation}: regime={regime}, "
            f"winner={winner_id} with {winner_tool}"
        )

        return result

    def _get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID."""
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None

    def run_generation(
        self,
        task_generator: Optional[callable] = None,
        evaluator: Optional[callable] = None
    ) -> CompetitionResult:
        """
        Run one generation (sample regime, generate task, compete).

        Args:
            task_generator: Function (regime) -> task_string
            evaluator: Function (response, task) -> (correct, confidence)

        Returns:
            CompetitionResult
        """
        # Sample regime
        regime = self.regime_sampler.sample()

        # Generate task
        if task_generator:
            task = task_generator(regime)
        else:
            task = f"Complete this {regime} task."

        return self.run_competition(task, regime, evaluator)

    def run_multiple_generations(
        self,
        n_generations: int,
        task_generator: Optional[callable] = None,
        evaluator: Optional[callable] = None,
        callback: Optional[callable] = None
    ) -> List[CompetitionResult]:
        """
        Run multiple generations.

        Args:
            n_generations: Number of generations to run
            task_generator: Function (regime) -> task_string
            evaluator: Function (response, task) -> (correct, confidence)
            callback: Function called after each generation with result

        Returns:
            List of CompetitionResult
        """
        results = []

        for _ in range(n_generations):
            result = self.run_generation(task_generator, evaluator)
            results.append(result)

            if callback:
                callback(result)

        return results

    def get_specialization_index(self) -> float:
        """
        Calculate Specialization Concentration Index (SCI).

        Higher values indicate more concentrated specialization.

        Returns:
            SCI value between 0 and 1
        """
        if not self._specialist_counts:
            return 0.0

        counts = list(self._specialist_counts.values())
        total = sum(counts)

        if total == 0:
            return 0.0

        # Herfindahl-Hirschman Index
        hhi = sum((c / total) ** 2 for c in counts)

        # Normalize to [0, 1]
        n_regimes = len(REGIMES)
        return (hhi - 1/n_regimes) / (1 - 1/n_regimes) if n_regimes > 1 else hhi

    def get_coverage(self) -> float:
        """
        Calculate regime coverage.

        Returns fraction of regimes with at least one specialist.
        """
        covered = len(self._specialist_counts)
        total = len(REGIMES)
        return covered / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get population statistics."""
        return {
            'n_agents': self.n_agents,
            'generation': self.generation,
            'sci': self.get_specialization_index(),
            'coverage': self.get_coverage(),
            'specialist_distribution': self._specialist_counts.copy(),
            'collusion_alerts': len(self.collusion_detector.alerts),
            'regime_samples': self.regime_sampler.get_empirical_distribution(),
        }

    def save(self, path: Path) -> None:
        """Save population state to file."""
        state = {
            'n_agents': self.n_agents,
            'generation': self.generation,
            'fitness_sharing_gamma': self.fitness_sharing_gamma,
            'agents': [a.to_dict() for a in self.agents],
            'history': [
                {
                    'generation': r.generation,
                    'regime': r.regime,
                    'winner_id': r.winner_id,
                    'winner_tool': r.winner_tool,
                }
                for r in self.history[-100:]  # Keep last 100
            ]
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: Path, llm_client=None) -> 'Population':
        """Load population state from file."""
        with open(path, 'r') as f:
            state = json.load(f)

        pop = cls(
            n_agents=state['n_agents'],
            fitness_sharing_gamma=state['fitness_sharing_gamma'],
            llm_client=llm_client
        )

        pop.generation = state['generation']
        pop.agents = [
            Agent.from_dict(a, llm_client)
            for a in state['agents']
        ]

        pop._update_specialist_counts()

        return pop
