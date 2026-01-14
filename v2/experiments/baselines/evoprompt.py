"""
EvoPrompt Baseline Implementation.

EvoPrompt: Evolutionary prompt optimization using DE/GA.
Uses differential evolution or genetic algorithms to evolve prompts.

Reference: Guo et al., "Connecting Large Language Models with Evolutionary Algorithms" (2023)
"""

import random
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass, field


@dataclass
class Prompt:
    """A prompt individual in the evolutionary population."""
    text: str
    fitness: float = 0.0
    
    def mutate(self, rng: random.Random, mutation_rate: float = 0.3) -> 'Prompt':
        """Simple mutation: swap/add/delete words."""
        words = self.text.split()
        if not words:
            return Prompt(text=self.text)
        
        if rng.random() < mutation_rate:
            # Mutation operations
            op = rng.choice(['swap', 'add', 'delete', 'replace'])
            
            if op == 'swap' and len(words) >= 2:
                i, j = rng.sample(range(len(words)), 2)
                words[i], words[j] = words[j], words[i]
            elif op == 'add':
                modifiers = ['carefully', 'precisely', 'step-by-step', 'thoroughly']
                pos = rng.randint(0, len(words))
                words.insert(pos, rng.choice(modifiers))
            elif op == 'delete' and len(words) > 3:
                del words[rng.randint(0, len(words) - 1)]
            elif op == 'replace':
                synonyms = {
                    'solve': ['answer', 'compute', 'determine'],
                    'task': ['problem', 'question', 'challenge'],
                    'accurate': ['precise', 'correct', 'exact'],
                }
                for i, word in enumerate(words):
                    if word.lower() in synonyms:
                        words[i] = rng.choice(synonyms[word.lower()])
                        break
        
        return Prompt(text=' '.join(words))


@dataclass
class EvoPromptOptimizer:
    """
    Evolutionary prompt optimizer using DE/GA.
    
    Maintains a population of prompts and evolves them
    using crossover and mutation operations.
    """
    regime: str
    population_size: int = 10
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    population: List[Prompt] = field(default_factory=list)
    total_tokens: int = 0
    generation: int = 0
    
    def __post_init__(self):
        # Initialize population with diverse prompts
        templates = [
            f"Solve the {self.regime} task accurately.",
            f"You are an expert at {self.regime}. Answer precisely.",
            f"For this {self.regime} problem, think step-by-step.",
            f"Apply {self.regime} knowledge to solve this.",
            f"Focus on {self.regime} principles and give the correct answer.",
        ]
        
        self.population = [Prompt(text=t) for t in templates]
        while len(self.population) < self.population_size:
            base = random.choice(templates)
            self.population.append(Prompt(text=base).mutate(random.Random()))
    
    def evolve_step(self, evaluate_fn: Callable, rng: random.Random) -> Dict:
        """
        One evolutionary step:
        1. Evaluate population
        2. Selection
        3. Crossover
        4. Mutation
        """
        # Evaluate all individuals
        for prompt in self.population:
            prompt.fitness, tokens = evaluate_fn(prompt.text)
            self.total_tokens += tokens
        
        # Sort by fitness
        self.population.sort(key=lambda p: -p.fitness)
        
        # Selection: keep top half
        survivors = self.population[:self.population_size // 2]
        
        # Generate offspring
        offspring = []
        while len(offspring) < self.population_size - len(survivors):
            # Tournament selection
            parent1 = rng.choice(survivors)
            parent2 = rng.choice(survivors)
            
            # Crossover
            if rng.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2, rng)
            else:
                child = Prompt(text=parent1.text)
            
            # Mutation
            child = child.mutate(rng, self.mutation_rate)
            offspring.append(child)
        
        self.population = survivors + offspring
        self.generation += 1
        
        best = max(self.population, key=lambda p: p.fitness)
        return {
            'best_fitness': best.fitness,
            'best_prompt': best.text,
            'generation': self.generation,
            'tokens': self.total_tokens,
        }
    
    def _crossover(self, p1: Prompt, p2: Prompt, rng: random.Random) -> Prompt:
        """Single-point crossover of two prompts."""
        words1 = p1.text.split()
        words2 = p2.text.split()
        
        if not words1 or not words2:
            return Prompt(text=p1.text)
        
        # Crossover point
        point1 = rng.randint(0, len(words1))
        point2 = rng.randint(0, len(words2))
        
        child_words = words1[:point1] + words2[point2:]
        return Prompt(text=' '.join(child_words) if child_words else p1.text)


def run_evoprompt_system(n_specialists: int, regimes: List[str],
                         evaluate_fn: Callable,
                         n_generations: int = 20) -> Dict:
    """
    Run EvoPrompt to produce N specialists using evolutionary optimization.
    
    Args:
        n_specialists: Number of specialists to create
        regimes: List of regime names
        evaluate_fn: Function to evaluate a prompt
        n_generations: Number of evolutionary generations
    
    Returns:
        Dict with evolved specialists and total token cost
    """
    specialists = {}
    total_tokens = 0
    
    for i, regime in enumerate(regimes[:n_specialists]):
        optimizer = EvoPromptOptimizer(regime=regime)
        rng = random.Random(i)
        
        for _ in range(n_generations):
            result = optimizer.evolve_step(
                evaluate_fn=lambda p: evaluate_fn(p, regime),
                rng=rng
            )
        
        specialists[regime] = {
            'prompt': result['best_prompt'],
            'fitness': result['best_fitness'],
        }
        total_tokens += optimizer.total_tokens
    
    return {
        'specialists': specialists,
        'total_tokens': total_tokens,
        'n_specialists': len(specialists),
    }
