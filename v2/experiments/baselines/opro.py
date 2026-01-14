"""
OPRO Baseline Implementation.

OPRO (Optimization by PROmpting): Uses LLM to iteratively improve prompts.
For N specialists, we run OPRO N times (once per regime).

Reference: Yang et al., "Large Language Models as Optimizers" (2023)
"""

import random
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class OPROOptimizer:
    """
    OPRO: Prompt optimization via LLM meta-prompting.

    For each regime, maintains a population of prompts and uses
    LLM to generate improved versions based on performance feedback.
    """
    regime: str
    population_size: int = 10
    max_iterations: int = 20
    prompts: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    total_tokens: int = 0

    def __post_init__(self):
        # Initialize with basic prompts
        self.prompts = [
            f"You are a specialist for {self.regime} tasks.",
            f"Focus on solving {self.regime} problems efficiently.",
            f"Apply {self.regime}-specific strategies.",
        ]
        self.scores = [0.0] * len(self.prompts)

    def optimize_step(self, evaluate_fn, meta_generate_fn, rng: random.Random) -> Dict:
        """
        One OPRO optimization step:
        1. Evaluate current prompts
        2. Generate improved prompts via meta-prompting
        3. Update population
        """
        # Evaluate all prompts
        for i, prompt in enumerate(self.prompts):
            score, tokens = evaluate_fn(prompt)
            self.scores[i] = score
            self.total_tokens += tokens

        # Sort by score
        sorted_pairs = sorted(zip(self.prompts, self.scores), key=lambda x: -x[1])
        top_k = sorted_pairs[:3]  # Keep top 3

        # Meta-prompt to generate new candidates
        meta_prompt = self._build_meta_prompt(top_k)
        new_prompts, tokens = meta_generate_fn(meta_prompt)
        self.total_tokens += tokens

        # Update population
        self.prompts = [p for p, _ in top_k] + new_prompts[:self.population_size - 3]
        self.scores = [s for _, s in top_k] + [0.0] * (len(self.prompts) - 3)

        return {
            'best_score': top_k[0][1] if top_k else 0,
            'best_prompt': top_k[0][0] if top_k else '',
            'tokens': self.total_tokens,
        }

    def _build_meta_prompt(self, top_k: List[Tuple[str, float]]) -> str:
        """Build meta-prompt for generating improved prompts."""
        examples = "\n".join([
            f"Prompt: {p}\nScore: {s:.2f}"
            for p, s in top_k
        ])

        return f"""You are optimizing prompts for {self.regime} tasks.

Here are the best prompts so far:
{examples}

Generate 3 improved prompts that might score higher.
Focus on clarity, specificity, and task relevance.
Return each prompt on a new line."""


def run_opro_system(n_specialists: int, regimes: List[str],
                    evaluate_fn, meta_generate_fn, n_iterations: int = 20) -> Dict:
    """
    Run OPRO N times to produce N specialists.

    Args:
        n_specialists: Number of specialists to create
        regimes: List of regime names
        evaluate_fn: Function to evaluate a prompt
        meta_generate_fn: Function to generate new prompts
        n_iterations: Optimization iterations per specialist

    Returns:
        Dict with specialists and total token cost
    """
    specialists = {}
    total_tokens = 0

    for i, regime in enumerate(regimes[:n_specialists]):
        optimizer = OPROOptimizer(regime=regime)

        for _ in range(n_iterations):
            result = optimizer.optimize_step(
                evaluate_fn=lambda p: evaluate_fn(p, regime),
                meta_generate_fn=meta_generate_fn,
                rng=random.Random(i)
            )

        specialists[regime] = {
            'prompt': result['best_prompt'],
            'score': result['best_score'],
        }
        total_tokens += optimizer.total_tokens

    return {
        'specialists': specialists,
        'total_tokens': total_tokens,
        'n_specialists': len(specialists),
    }
