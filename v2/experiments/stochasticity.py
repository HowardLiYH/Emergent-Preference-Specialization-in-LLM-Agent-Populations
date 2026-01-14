"""
Stochasticity analysis for emergent behavior validation.

Measures how much specialization patterns vary across runs.
High entropy = truly emergent; Low entropy = deterministic.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from collections import Counter
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def get_specialization_pattern(population) -> Dict[str, str]:
    """
    Extract specialization pattern from population.

    Args:
        population: Population object

    Returns:
        Dict mapping agent_id to specialty regime
    """
    pattern = {}
    for agent in population.agents:
        specialty = agent.get_specialty()
        if specialty:
            pattern[agent.id] = specialty
    return pattern


def pattern_to_string(pattern: Dict[str, str]) -> str:
    """Convert pattern to comparable string."""
    return str(sorted(pattern.items()))


def analyze_stochasticity(
    train_population: callable,
    n_runs: int = 10,
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Analyze stochasticity of specialization.

    Args:
        train_population: Function (seed) -> trained_population
        n_runs: Number of runs
        output_path: Path to save results

    Returns:
        Stochasticity analysis results
    """
    logger.info(f"Running stochasticity analysis with {n_runs} runs...")

    patterns = []
    pattern_strings = []

    for seed in range(n_runs):
        logger.info(f"Run {seed + 1}/{n_runs}")

        try:
            population = train_population(seed)
            pattern = get_specialization_pattern(population)
            patterns.append(pattern)
            pattern_strings.append(pattern_to_string(pattern))

        except Exception as e:
            logger.error(f"Run {seed} failed: {e}")

    # Analyze patterns
    unique_patterns = len(set(pattern_strings))
    pattern_entropy = np.log2(unique_patterns) if unique_patterns > 1 else 0

    # Count pattern occurrences
    pattern_counts = Counter(pattern_strings)
    most_common = pattern_counts.most_common(3)

    # Determine interpretation
    if unique_patterns >= 8:
        interpretation = "Highly stochastic (truly emergent)"
    elif unique_patterns >= 4:
        interpretation = "Moderately stochastic"
    elif unique_patterns >= 2:
        interpretation = "Low stochasticity (somewhat deterministic)"
    else:
        interpretation = "Deterministic (not emergent)"

    results = {
        'n_runs': n_runs,
        'unique_patterns': unique_patterns,
        'pattern_entropy': float(pattern_entropy),
        'interpretation': interpretation,
        'is_emergent': unique_patterns >= 4,
        'pattern_distribution': dict(pattern_counts),
        'most_common_patterns': [
            {'pattern': p, 'count': c}
            for p, c in most_common
        ],
    }

    # Save results
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / 'stochasticity_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    logger.info(f"Stochasticity analysis complete: {interpretation}")
    return results


def generate_stochasticity_report(results: Dict[str, Any]) -> str:
    """Generate stochasticity report."""
    lines = [
        "# Stochasticity Analysis Report",
        "",
        f"**Runs:** {results['n_runs']}",
        f"**Unique Patterns:** {results['unique_patterns']}",
        f"**Pattern Entropy:** {results['pattern_entropy']:.2f} bits",
        f"**Interpretation:** {results['interpretation']}",
        "",
        "## Conclusion",
        "",
    ]

    if results['is_emergent']:
        lines.append(
            "✅ **Specialization is EMERGENT** - different patterns emerge across runs."
        )
    else:
        lines.append(
            "⚠️ **Specialization may be DETERMINISTIC** - similar patterns across runs."
        )

    return "\n".join(lines)
