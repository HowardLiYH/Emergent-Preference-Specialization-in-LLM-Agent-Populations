"""
Context manager for experiment execution.

Handles setup, teardown, and resource management.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json
import logging
import os

from .llm_client import LLMClient
from .population import Population
from .curriculum import TaskCurriculum
from ..memory.store import MemoryStore

logger = logging.getLogger(__name__)


class ExperimentContext:
    """
    Context for running experiments.

    Manages:
    - LLM client
    - Population
    - Memory store
    - Curriculum
    - Logging
    - Checkpointing
    """

    def __init__(
        self,
        experiment_name: str,
        base_path: Optional[Path] = None,
        seed: int = 0,
        n_agents: int = 12,
        n_generations: int = 100
    ):
        """
        Initialize experiment context.

        Args:
            experiment_name: Unique name for this experiment
            base_path: Base directory for outputs
            seed: Random seed
            n_agents: Number of agents
            n_generations: Number of generations to run
        """
        self.experiment_name = experiment_name
        self.base_path = Path(base_path) if base_path else Path('v2/results')
        self.seed = seed
        self.n_agents = n_agents
        self.n_generations = n_generations

        # Create output directory
        self.output_path = self.base_path / experiment_name
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Timestamp
        self.started_at = datetime.now()

        # Components (initialized in __enter__)
        self.llm_client: Optional[LLMClient] = None
        self.population: Optional[Population] = None
        self.curriculum: Optional[TaskCurriculum] = None
        self.memory_store: Optional[MemoryStore] = None

        # Metrics
        self.metrics: Dict[str, Any] = {}

    def __enter__(self):
        """Set up experiment context."""
        logger.info(f"Starting experiment: {self.experiment_name}")

        # Initialize LLM client
        self.llm_client = LLMClient()

        # Initialize memory store
        self.memory_store = MemoryStore(
            base_path=self.output_path / 'memories'
        )

        # Initialize population
        self.population = Population(
            n_agents=self.n_agents,
            seed=self.seed,
            llm_client=self.llm_client
        )

        # Initialize curriculum
        self.curriculum = TaskCurriculum(seed=self.seed)

        # Save initial config
        self._save_config()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up and save results."""
        self.ended_at = datetime.now()
        duration = (self.ended_at - self.started_at).total_seconds()

        logger.info(
            f"Experiment {self.experiment_name} completed in {duration:.1f}s"
        )

        # Save final results
        self._save_results()

        # Don't suppress exceptions
        return False

    def _save_config(self) -> None:
        """Save experiment configuration."""
        config = {
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'n_agents': self.n_agents,
            'n_generations': self.n_generations,
            'started_at': self.started_at.isoformat(),
            'llm_model': self.llm_client.model if self.llm_client else None,
        }

        config_path = self.output_path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def _save_results(self) -> None:
        """Save experiment results."""
        results = {
            'experiment_name': self.experiment_name,
            'started_at': self.started_at.isoformat(),
            'ended_at': self.ended_at.isoformat() if hasattr(self, 'ended_at') else None,
            'metrics': self.metrics,
            'population_stats': self.population.get_stats() if self.population else {},
            'llm_stats': self.llm_client.get_stats() if self.llm_client else {},
        }

        results_path = self.output_path / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Save population state
        if self.population:
            self.population.save(self.output_path / 'population.json')

    def checkpoint(self, name: str = 'checkpoint') -> None:
        """Save a checkpoint."""
        checkpoint_path = self.output_path / f'{name}.json'

        state = {
            'generation': self.population.generation if self.population else 0,
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics.copy(),
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(state, f, indent=2)

        if self.population:
            self.population.save(self.output_path / f'{name}_population.json')

    def record_metric(self, name: str, value: Any) -> None:
        """Record a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append({
            'generation': self.population.generation if self.population else 0,
            'value': value,
            'timestamp': datetime.now().isoformat(),
        })

    def get_output_path(self, filename: str) -> Path:
        """Get path for an output file."""
        return self.output_path / filename


class ContextManager:
    """
    Manager for multiple experiment contexts.
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize context manager.

        Args:
            base_path: Base directory for all experiments
        """
        self.base_path = Path(base_path) if base_path else Path('v2/results')
        self.base_path.mkdir(parents=True, exist_ok=True)

    def create_experiment(
        self,
        name: str,
        seed: int = 0,
        **kwargs
    ) -> ExperimentContext:
        """
        Create a new experiment context.

        Args:
            name: Experiment name
            seed: Random seed
            **kwargs: Additional arguments for ExperimentContext

        Returns:
            ExperimentContext
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        full_name = f"{name}_{timestamp}_seed{seed}"

        return ExperimentContext(
            experiment_name=full_name,
            base_path=self.base_path,
            seed=seed,
            **kwargs
        )

    def list_experiments(self) -> list:
        """List all experiments in base path."""
        experiments = []

        for path in self.base_path.iterdir():
            if path.is_dir():
                config_path = path / 'config.json'
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    experiments.append({
                        'name': path.name,
                        'config': config,
                        'path': path,
                    })

        return experiments
