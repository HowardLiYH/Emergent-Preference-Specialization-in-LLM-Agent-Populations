"""
Genesis: LLM Agent Evolution Framework

Paper 2: Competitive Selection Drives Role Specialization in LLM Agent Populations

This package implements:
- 10 orthogonal cognitive domains (based on Gardner, Bloom, Pearl)
- 280 problem-solving strategies from established benchmarks
- 150 worked examples
- Cumulative evolution mechanism
- Fitness sharing for diversity preservation
- Specialization metrics (LSI, semantic, behavioral)
- Counterfactual validation (prompt swap test)
"""

# Core agent and tasks
from .agent import GenesisAgent
from .tasks import Task, TaskPool, TaskType
from .competition import CompetitionEngine, CompetitionResult

# Legacy evolution
from .evolution import evolve_prompt_directed, evolve_prompt_random

# v7: New domain system
from .domains import DomainType, DomainSpec, get_domain_spec, get_all_domains

# v7: Strategy and example libraries
from .strategy_library import Strategy, StrategyLibrary, STRATEGY_LIBRARY
from .example_library import WorkedExample, ExampleLibrary, EXAMPLE_LIBRARY

# v7: Cumulative evolution
from .evolution_v2 import (
    SpecialistPrompt,
    EvolutionType,
    evolve_cumulative,
    evolve_random_strategy,
    evolve_no_change,
    create_initial_prompt,
    create_specialist_prompt,
    create_all_specialists
)

# v7: Fitness sharing
from .fitness_sharing import (
    FitnessSharingConfig,
    apply_fitness_sharing,
    calculate_crowding_penalty,
    get_niche_statistics
)

# v7: Cost tracking
from .cost_tracker import CostTracker, estimate_experiment_cost

# Metrics
from .metrics import compute_lsi, compute_semantic_specialization, compute_behavioral_fingerprint

# Counterfactual
from .counterfactual import run_prompt_swap_test

# Simulation
from .simulation import GenesisSimulation

# LLM client
from .llm_client import LLMClient, create_client
from .config import LLMConfig, get_config

__all__ = [
    # Core
    'GenesisAgent',
    'Task',
    'TaskPool',
    'TaskType',
    'CompetitionEngine',
    'CompetitionResult',

    # v7: Domains
    'DomainType',
    'DomainSpec',
    'get_domain_spec',
    'get_all_domains',

    # v7: Libraries
    'Strategy',
    'StrategyLibrary',
    'STRATEGY_LIBRARY',
    'WorkedExample',
    'ExampleLibrary',
    'EXAMPLE_LIBRARY',

    # v7: Evolution
    'SpecialistPrompt',
    'EvolutionType',
    'evolve_cumulative',
    'evolve_random_strategy',
    'evolve_no_change',
    'create_initial_prompt',
    'create_specialist_prompt',
    'create_all_specialists',

    # v7: Fitness sharing
    'FitnessSharingConfig',
    'apply_fitness_sharing',
    'calculate_crowding_penalty',
    'get_niche_statistics',

    # v7: Cost tracking
    'CostTracker',
    'estimate_experiment_cost',

    # Legacy evolution
    'evolve_prompt_directed',
    'evolve_prompt_random',

    # Metrics
    'compute_lsi',
    'compute_semantic_specialization',
    'compute_behavioral_fingerprint',

    # Counterfactual
    'run_prompt_swap_test',

    # Simulation
    'GenesisSimulation',

    # LLM
    'LLMClient',
    'create_client',
    'LLMConfig',
    'get_config',
]

__version__ = '0.3.0'
