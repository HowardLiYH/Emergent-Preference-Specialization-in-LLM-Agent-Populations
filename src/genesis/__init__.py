"""
Genesis: LLM Agent Evolution Framework

Paper 2: Emergent Prompt Specialization in LLM Agent Populations

This package implements:
- GenesisAgent: LLM agents with evolvable system prompts
- Competition engine with winner-take-all dynamics
- Directed and random prompt evolution
- Specialization metrics (LSI, semantic, behavioral)
- Counterfactual validation (prompt swap test)
"""

from .agent import GenesisAgent
from .tasks import Task, TaskPool, TaskType
from .competition import CompetitionEngine, CompetitionResult
from .evolution import evolve_prompt_directed, evolve_prompt_random
from .metrics import compute_lsi, compute_semantic_specialization, compute_behavioral_fingerprint
from .counterfactual import run_prompt_swap_test
from .simulation import GenesisSimulation

__all__ = [
    'GenesisAgent',
    'Task',
    'TaskPool',
    'TaskType',
    'CompetitionEngine',
    'CompetitionResult',
    'evolve_prompt_directed',
    'evolve_prompt_random',
    'compute_lsi',
    'compute_semantic_specialization',
    'compute_behavioral_fingerprint',
    'run_prompt_swap_test',
    'GenesisSimulation',
]

__version__ = '0.1.0'
