"""
Main simulation orchestrator for Genesis experiments.

Coordinates agents, tasks, competition, evolution, and metrics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import asyncio
import json
from datetime import datetime
from pathlib import Path

from .agent import GenesisAgent, create_population
from .tasks import TaskPool, TaskType
from .competition import CompetitionEngine
from .evolution import create_evolution_function
from .metrics import compute_all_metrics, compute_population_lsi


@dataclass
class SimulationConfig:
    """Configuration for a Genesis simulation."""
    n_agents: int = 16
    n_generations: int = 100
    tasks_per_generation: int = 10

    # Evolution settings
    evolution_type: str = "directed"  # "directed", "random", "minimal", "none"
    evolution_temperature: float = 0.7

    # Competition settings
    winner_takes_all: bool = True

    # LLM settings
    model: str = "gpt-4"
    response_temperature: float = 0.7

    # Logging
    log_every: int = 10
    save_prompts: bool = True

    # Random seed
    seed: Optional[int] = None


@dataclass
class GenerationResult:
    """Result of a single generation."""
    generation: int
    n_rounds: int
    winners: List[str]  # agent IDs
    mean_score: float
    metrics: Dict[str, Any]


@dataclass
class SimulationResult:
    """Result of a complete simulation."""
    config: SimulationConfig
    generations: List[GenerationResult]
    final_agents: List[GenesisAgent]
    final_metrics: Dict[str, Any]
    duration_seconds: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "config": self.config.__dict__,
            "n_generations": len(self.generations),
            "final_metrics": self.final_metrics,
            "duration_seconds": self.duration_seconds,
            "generation_metrics": [
                {
                    "generation": g.generation,
                    "mean_score": g.mean_score,
                    "lsi_mean": g.metrics.get("lsi", {}).get("mean", 0)
                }
                for g in self.generations
            ]
        }

    def save(self, path: str):
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class GenesisSimulation:
    """
    Main simulation class for Genesis experiments.

    Usage:
        sim = GenesisSimulation(config, llm_client)
        results = await sim.run()
    """

    def __init__(
        self,
        config: SimulationConfig,
        llm_client,
        initial_prompt: str = None
    ):
        """
        Initialize simulation.

        Args:
            config: Simulation configuration
            llm_client: OpenAI-compatible client
            initial_prompt: Starting prompt for all agents (optional)
        """
        self.config = config
        self.llm = llm_client
        self.initial_prompt = initial_prompt

        # Initialize components
        self.agents: List[GenesisAgent] = []
        self.task_pool = TaskPool()
        self.engine: Optional[CompetitionEngine] = None
        self.evolve_func = None

        # Results tracking
        self.generation_results: List[GenerationResult] = []

    def _setup(self):
        """Initialize agents and components."""
        # Create agent population
        self.agents = create_population(
            self.config.n_agents,
            self.initial_prompt
        )

        # Create competition engine
        evolution_enabled = self.config.evolution_type != "none"
        self.engine = CompetitionEngine(
            llm_client=self.llm,
            evolution_enabled=evolution_enabled,
            winner_takes_all=self.config.winner_takes_all,
            temperature=self.config.response_temperature
        )

        # Create evolution function
        if evolution_enabled:
            self.evolve_func = create_evolution_function(
                self.llm,
                self.config.evolution_type,
                self.config.evolution_temperature
            )
        else:
            self.evolve_func = None

    async def run(self) -> SimulationResult:
        """
        Run the full simulation.

        Returns:
            SimulationResult with all data
        """
        import time
        start_time = time.time()

        # Setup
        self._setup()

        # Run generations
        for gen in range(self.config.n_generations):
            gen_result = await self._run_generation(gen)
            self.generation_results.append(gen_result)

            # Logging
            if gen % self.config.log_every == 0:
                lsi_mean = gen_result.metrics.get("lsi", {}).get("mean", 0)
                print(f"Generation {gen}: LSI={lsi_mean:.3f}, "
                      f"Score={gen_result.mean_score:.3f}")

        # Final metrics
        final_metrics = compute_all_metrics(self.agents)

        # Create result
        duration = time.time() - start_time
        result = SimulationResult(
            config=self.config,
            generations=self.generation_results,
            final_agents=self.agents,
            final_metrics=final_metrics,
            duration_seconds=duration
        )

        return result

    async def _run_generation(self, gen_num: int) -> GenerationResult:
        """Run a single generation."""
        # Sample tasks for this generation
        tasks = self.task_pool.sample(self.config.tasks_per_generation)

        # Run competition rounds
        results = await self.engine.run_generation(
            self.agents, tasks, self.evolve_func
        )

        # Compute metrics
        metrics = compute_all_metrics(self.agents)

        # Aggregate results
        winners = [r.winner.id for r in results]
        scores = [r.winner_score for r in results]

        return GenerationResult(
            generation=gen_num,
            n_rounds=len(results),
            winners=winners,
            mean_score=float(sum(scores) / len(scores)),
            metrics=metrics
        )

    def get_lsi_trajectory(self) -> List[float]:
        """Get LSI values over time."""
        return [
            g.metrics.get("lsi", {}).get("mean", 0)
            for g in self.generation_results
        ]

    def get_specialists(self) -> Dict[str, List[GenesisAgent]]:
        """Get current specialists by task type."""
        from .metrics import identify_specialists
        return identify_specialists(self.agents)


async def run_experiment(
    config: SimulationConfig,
    llm_client,
    n_runs: int = 10,
    output_dir: str = "results"
) -> Dict[str, Any]:
    """
    Run multiple simulation runs for statistical analysis.

    Args:
        config: Simulation configuration
        llm_client: LLM client
        n_runs: Number of independent runs
        output_dir: Directory to save results

    Returns:
        Aggregated results across all runs
    """
    import numpy as np

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = []

    for run in range(n_runs):
        print(f"\n=== Run {run + 1}/{n_runs} ===")

        # Create simulation with different seed
        run_config = SimulationConfig(**config.__dict__)
        run_config.seed = run

        sim = GenesisSimulation(run_config, llm_client)
        result = await sim.run()

        # Save individual run
        result.save(str(output_path / f"run_{run}.json"))
        all_results.append(result)

    # Aggregate statistics
    final_lsi_values = [r.final_metrics["lsi"]["mean"] for r in all_results]

    aggregated = {
        "n_runs": n_runs,
        "config": config.__dict__,
        "final_lsi": {
            "mean": float(np.mean(final_lsi_values)),
            "std": float(np.std(final_lsi_values)),
            "min": float(np.min(final_lsi_values)),
            "max": float(np.max(final_lsi_values)),
        },
        "runs": [r.to_dict() for r in all_results]
    }

    # Save aggregated results
    with open(output_path / "aggregated.json", 'w') as f:
        json.dump(aggregated, f, indent=2)

    return aggregated
