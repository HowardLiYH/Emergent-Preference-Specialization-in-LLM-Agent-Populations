"""
Metrics for measuring LLM agent specialization.

Multi-layer measurement approach:
1. Task Performance (objective)
2a. LSI - LLM Specialization Index (performance-based)
2b. Semantic Specialization (embedding-based)
2c. Behavioral Fingerprint (self-reported)
2d. Performance Variance Ratio
"""

from typing import List, Dict, Tuple
import numpy as np
from collections import Counter

from .agent import GenesisAgent


def compute_lsi(agent: GenesisAgent) -> float:
    """
    Compute LLM Specialization Index (LSI) for an agent.

    LSI is based on the entropy of performance distribution across task types.

    LSI = 1 - (entropy / max_entropy)

    - LSI = 0: Equal performance across all types (generalist)
    - LSI = 1: Perfect in one type, zero in others (specialist)

    Args:
        agent: The agent to evaluate

    Returns:
        float: LSI value between 0 and 1
    """
    perf = agent.get_performance_by_type()

    if not perf or len(perf) < 2:
        return 0.0

    # Get performance values
    values = np.array(list(perf.values()))

    # Handle edge cases
    if values.sum() == 0:
        return 0.0

    # Normalize to probability distribution
    p = values / values.sum()

    # Compute entropy
    # Add small epsilon to avoid log(0)
    entropy = -np.sum(p * np.log(p + 1e-10))
    max_entropy = np.log(len(p))

    # LSI = 1 - normalized entropy
    lsi = 1 - entropy / max_entropy

    return float(np.clip(lsi, 0, 1))


def compute_population_lsi(agents: List[GenesisAgent]) -> Dict[str, float]:
    """
    Compute LSI statistics for a population.

    Returns:
        Dict with mean, std, min, max LSI values
    """
    lsi_values = [compute_lsi(agent) for agent in agents]

    return {
        "mean": float(np.mean(lsi_values)),
        "std": float(np.std(lsi_values)),
        "min": float(np.min(lsi_values)),
        "max": float(np.max(lsi_values)),
        "values": lsi_values
    }


def compute_semantic_specialization(
    agents: List[GenesisAgent],
    model_name: str = 'all-MiniLM-L6-v2'
) -> List[float]:
    """
    Compute semantic specialization based on prompt embeddings.

    Measures how semantically distinct each agent's prompt is from others.
    Higher values = more unique/specialized prompts.

    Args:
        agents: List of agents
        model_name: Sentence transformer model to use

    Returns:
        List of semantic specialization scores (one per agent)
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics import pairwise_distances
    except ImportError:
        print("Warning: sentence-transformers or sklearn not installed")
        return [0.0] * len(agents)

    if len(agents) < 2:
        return [0.0] * len(agents)

    # Get prompt embeddings
    model = SentenceTransformer(model_name)
    prompts = [agent.system_prompt for agent in agents]
    embeddings = model.encode(prompts)

    # Compute pairwise cosine distances
    distances = pairwise_distances(embeddings, metric='cosine')

    # Semantic specialization = mean distance from other agents
    semantic_spec = distances.mean(axis=1)

    return semantic_spec.tolist()


def compute_prompt_diversity(agents: List[GenesisAgent]) -> float:
    """
    Compute overall prompt diversity in the population.

    Returns mean pairwise distance between all prompts.
    Higher = more diverse population.
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics import pairwise_distances
    except ImportError:
        return 0.0

    if len(agents) < 2:
        return 0.0

    model = SentenceTransformer('all-MiniLM-L6-v2')
    prompts = [agent.system_prompt for agent in agents]
    embeddings = model.encode(prompts)

    distances = pairwise_distances(embeddings, metric='cosine')

    # Mean of upper triangle (excluding diagonal)
    n = len(agents)
    upper_tri = distances[np.triu_indices(n, k=1)]

    return float(np.mean(upper_tri))


DIAGNOSTIC_QUESTIONS = [
    "What type of tasks are you best at? Answer in one word.",
    "Rate your expertise in math, coding, logic, and language from 1-10. Format: math:X, coding:X, logic:X, language:X",
    "Describe your specialty in 10 words or less.",
]


async def compute_behavioral_fingerprint(
    agent: GenesisAgent,
    llm_client
) -> Dict[str, str]:
    """
    Compute behavioral fingerprint via self-reported specialization.

    Ask the agent diagnostic questions to understand how it perceives itself.

    Args:
        agent: The agent to fingerprint
        llm_client: LLM API client

    Returns:
        Dict mapping questions to agent responses
    """
    fingerprint = {}

    for question in DIAGNOSTIC_QUESTIONS:
        messages = [
            agent.to_openai_message(),
            {"role": "user", "content": question}
        ]

        response = await llm_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0,
            max_tokens=50
        )

        fingerprint[question] = response.choices[0].message.content.strip()

    return fingerprint


def compute_behavioral_diversity(fingerprints: List[Dict[str, str]]) -> float:
    """
    Compute diversity of behavioral fingerprints.

    Higher values = agents report more diverse specializations.
    """
    if len(fingerprints) < 2:
        return 0.0

    # Focus on first question ("What type of tasks are you best at?")
    first_q = DIAGNOSTIC_QUESTIONS[0]
    responses = [fp.get(first_q, "").lower() for fp in fingerprints]

    # Count unique responses
    unique = len(set(responses))

    # Diversity = unique responses / total agents
    return unique / len(fingerprints)


def compute_performance_variance_ratio(agent: GenesisAgent) -> float:
    """
    Compute Performance Variance Ratio (PVR).

    PVR = best_performance / worst_performance

    - PVR = 1: Equal performance (generalist)
    - PVR > 2: Strong specialization

    Args:
        agent: The agent to evaluate

    Returns:
        float: PVR value (>= 1)
    """
    perf = agent.get_performance_by_type()

    if not perf:
        return 1.0

    values = list(perf.values())

    # Avoid division by zero
    min_val = min(values)
    max_val = max(values)

    if min_val <= 0:
        return max_val / 0.01  # Use small epsilon

    return max_val / min_val


def identify_specialists(
    agents: List[GenesisAgent],
    lsi_threshold: float = 0.5
) -> Dict[str, List[GenesisAgent]]:
    """
    Identify specialists by task type.

    Args:
        agents: List of agents
        lsi_threshold: Minimum LSI to be considered specialist

    Returns:
        Dict mapping task type to list of specialists
    """
    specialists = {}

    for agent in agents:
        lsi = compute_lsi(agent)
        if lsi >= lsi_threshold:
            best_type = agent.get_best_task_type()
            if best_type:
                if best_type not in specialists:
                    specialists[best_type] = []
                specialists[best_type].append(agent)

    return specialists


def compute_all_metrics(
    agents: List[GenesisAgent]
) -> Dict[str, any]:
    """
    Compute all specialization metrics for a population.

    Returns comprehensive metrics dictionary.
    """
    lsi_stats = compute_population_lsi(agents)
    semantic_spec = compute_semantic_specialization(agents)
    prompt_diversity = compute_prompt_diversity(agents)
    pvr_values = [compute_performance_variance_ratio(a) for a in agents]
    specialists = identify_specialists(agents)

    return {
        "lsi": lsi_stats,
        "semantic_specialization": {
            "mean": float(np.mean(semantic_spec)),
            "std": float(np.std(semantic_spec)),
            "values": semantic_spec
        },
        "prompt_diversity": prompt_diversity,
        "performance_variance_ratio": {
            "mean": float(np.mean(pvr_values)),
            "std": float(np.std(pvr_values)),
            "values": pvr_values
        },
        "specialists": {
            task_type: len(agents)
            for task_type, agents in specialists.items()
        },
        "n_specialists": sum(len(a) for a in specialists.values()),
        "n_generalists": len(agents) - sum(len(a) for a in specialists.values())
    }
