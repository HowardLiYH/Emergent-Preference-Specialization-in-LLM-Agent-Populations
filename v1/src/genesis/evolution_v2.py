"""
Cumulative Evolution Mechanism (v2)

Key improvements over v1:
1. Cumulative: Strategies are ADDED to the prompt, not replaced
2. Fixed library: Strategies come from a fixed library, not LLM generation
3. Structured prompts: Agent prompts follow a structured template
4. History tracking: Full history of acquired strategies and performance
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum
import random

from .domains import DomainType, get_domain_spec
from .strategy_library import Strategy, STRATEGY_LIBRARY, get_random_strategy
from .example_library import WorkedExample, EXAMPLE_LIBRARY, get_random_example


@dataclass
class SpecialistPrompt:
    """Structured agent prompt with accumulated knowledge."""

    # Identity
    agent_id: str
    generation: int = 0

    # Specialization
    primary_domain: Optional[DomainType] = None
    acquired_strategies: List[Strategy] = field(default_factory=list)
    reference_examples: List[WorkedExample] = field(default_factory=list)

    # Performance tracking
    domain_wins: Dict[str, int] = field(default_factory=dict)
    domain_attempts: Dict[str, int] = field(default_factory=dict)
    total_wins: int = 0
    total_attempts: int = 0

    # Evolution history
    evolution_history: List[Dict] = field(default_factory=list)

    # Configuration
    max_strategies: int = 12  # Cap to prevent prompt explosion
    max_examples: int = 5
    max_prompt_words: int = 800

    def get_strategy_ids(self) -> Set[str]:
        """Get IDs of all acquired strategies."""
        return {s.id for s in self.acquired_strategies}

    def get_example_ids(self) -> Set[str]:
        """Get IDs of all reference examples."""
        return {e.id for e in self.reference_examples}

    def add_strategy(self, strategy: Strategy) -> bool:
        """Add a strategy if not already present and under limit."""
        if strategy.id in self.get_strategy_ids():
            return False  # Already have this strategy

        if len(self.acquired_strategies) >= self.max_strategies:
            # Remove least relevant strategy (oldest or lowest domain match)
            self._prune_strategies()

        self.acquired_strategies.append(strategy)
        return True

    def add_example(self, example: WorkedExample) -> bool:
        """Add an example if not already present and under limit."""
        if example.id in self.get_example_ids():
            return False

        if len(self.reference_examples) >= self.max_examples:
            # Remove oldest example
            self.reference_examples.pop(0)

        self.reference_examples.append(example)
        return True

    def _prune_strategies(self):
        """Remove least relevant strategy when at capacity."""
        if not self.acquired_strategies:
            return

        # If we have a primary domain, keep strategies from that domain
        if self.primary_domain:
            # Sort by relevance to primary domain
            def relevance(s: Strategy) -> int:
                if s.domain == self.primary_domain:
                    return 1  # Keep domain-relevant strategies
                return 0

            self.acquired_strategies.sort(key=relevance, reverse=True)

        # Remove the last (least relevant) strategy
        self.acquired_strategies.pop()

    def record_attempt(self, domain: DomainType, won: bool):
        """Record a task attempt result."""
        domain_key = domain.value
        self.domain_attempts[domain_key] = self.domain_attempts.get(domain_key, 0) + 1
        self.total_attempts += 1

        if won:
            self.domain_wins[domain_key] = self.domain_wins.get(domain_key, 0) + 1
            self.total_wins += 1

            # Update primary domain based on wins
            self._update_primary_domain()

    def _update_primary_domain(self):
        """Update primary domain based on win distribution."""
        if not self.domain_wins:
            return

        # Find domain with most wins
        best_domain = max(self.domain_wins.keys(), key=lambda d: self.domain_wins[d])
        self.primary_domain = DomainType(best_domain)

    def get_win_rate(self, domain: DomainType) -> float:
        """Get win rate for a specific domain."""
        domain_key = domain.value
        attempts = self.domain_attempts.get(domain_key, 0)
        if attempts == 0:
            return 0.0
        wins = self.domain_wins.get(domain_key, 0)
        return wins / attempts

    def get_overall_win_rate(self) -> float:
        """Get overall win rate across all domains."""
        if self.total_attempts == 0:
            return 0.0
        return self.total_wins / self.total_attempts

    def to_prompt_text(self) -> str:
        """Generate the full prompt text for the LLM."""
        sections = []

        # Header
        sections.append(self._generate_header())

        # Specialization summary
        if self.primary_domain:
            sections.append(self._generate_specialization())

        # Acquired strategies
        if self.acquired_strategies:
            sections.append(self._generate_strategies())

        # Reference examples (abbreviated)
        if self.reference_examples:
            sections.append(self._generate_examples())

        # Performance summary
        if self.total_attempts > 0:
            sections.append(self._generate_performance())

        return "\n\n".join(sections)

    def _generate_header(self) -> str:
        """Generate the role header."""
        if self.primary_domain:
            spec = get_domain_spec(self.primary_domain)
            return f"""# Agent {self.agent_id} | Generation {self.generation}
Role: {spec.name} Specialist
Core Operation: {spec.core_operation}"""
        else:
            return f"""# Agent {self.agent_id} | Generation {self.generation}
Role: General AI Assistant (developing specialization)"""

    def _generate_specialization(self) -> str:
        """Generate specialization description."""
        if not self.primary_domain:
            return ""

        spec = get_domain_spec(self.primary_domain)
        win_rate = self.get_win_rate(self.primary_domain)

        return f"""## Specialization: {spec.name}
- Description: {spec.description}
- Input type: {spec.input_type}
- Output type: {spec.output_type}
- Win rate: {win_rate:.0%}"""

    def _generate_strategies(self) -> str:
        """Generate strategies section."""
        lines = ["## Acquired Strategies"]

        for strategy in self.acquired_strategies:
            lines.append(strategy.to_short_text())

        return "\n".join(lines)

    def _generate_examples(self) -> str:
        """Generate examples section (abbreviated)."""
        lines = ["## Reference Examples"]

        for example in self.reference_examples[:3]:  # Show max 3
            lines.append(f"• [{example.domain.value}] {example.problem[:80]}...")

        return "\n".join(lines)

    def _generate_performance(self) -> str:
        """Generate performance summary."""
        lines = ["## Performance History"]
        lines.append(f"Total: {self.total_wins}/{self.total_attempts} wins ({self.get_overall_win_rate():.0%})")

        # Top 3 domains by wins
        if self.domain_wins:
            sorted_domains = sorted(self.domain_wins.items(), key=lambda x: x[1], reverse=True)[:3]
            for domain_key, wins in sorted_domains:
                attempts = self.domain_attempts.get(domain_key, 0)
                lines.append(f"  - {domain_key}: {wins}/{attempts}")

        return "\n".join(lines)

    def to_openai_message(self) -> Dict:
        """Convert to OpenAI message format."""
        return {
            "role": "system",
            "content": self.to_prompt_text()
        }


class EvolutionType(Enum):
    """Types of evolution mechanisms."""
    CUMULATIVE = "cumulative"  # Add strategies from library
    RANDOM_STRATEGY = "random_strategy"  # Random strategy from library (baseline)
    NO_EVOLUTION = "no_evolution"  # No changes
    DIRECTED_LLM = "directed_llm"  # LLM generates changes (v1 style)


@dataclass
class EvolutionEvent:
    """Record of a single evolution event."""
    generation: int
    domain: DomainType
    task_score: float
    strategy_added: Optional[Strategy] = None
    example_added: Optional[WorkedExample] = None
    evolution_type: EvolutionType = EvolutionType.CUMULATIVE


def evolve_cumulative(
    prompt: SpecialistPrompt,
    domain: DomainType,
    score: float,
    add_example: bool = True
) -> EvolutionEvent:
    """
    Cumulative evolution: Add a relevant strategy from the fixed library.

    This is the main evolution mechanism. When an agent wins a task:
    1. Select a strategy from that domain's library
    2. Add it to the agent's accumulated strategies
    3. Optionally add a reference example
    """
    event = EvolutionEvent(
        generation=prompt.generation,
        domain=domain,
        task_score=score,
        evolution_type=EvolutionType.CUMULATIVE
    )

    # Get strategies for this domain that the agent doesn't have
    domain_strategies = STRATEGY_LIBRARY.get_by_domain(domain)
    available = [s for s in domain_strategies if s.id not in prompt.get_strategy_ids()]

    if available:
        # Select a random strategy from available ones
        strategy = random.choice(available)
        prompt.add_strategy(strategy)
        event.strategy_added = strategy

    # Optionally add an example
    if add_example:
        domain_examples = EXAMPLE_LIBRARY.get_by_domain(domain)
        available_examples = [e for e in domain_examples if e.id not in prompt.get_example_ids()]

        if available_examples:
            example = random.choice(available_examples)
            prompt.add_example(example)
            event.example_added = example

    # Update generation
    prompt.generation += 1
    prompt.evolution_history.append({
        "generation": prompt.generation,
        "domain": domain.value,
        "score": score,
        "strategy_added": event.strategy_added.id if event.strategy_added else None,
        "example_added": event.example_added.id if event.example_added else None
    })

    return event


def evolve_random_strategy(
    prompt: SpecialistPrompt,
    domain: DomainType,
    score: float
) -> EvolutionEvent:
    """
    Random strategy baseline: Add a random strategy from ANY domain.

    This baseline tests whether domain-specific selection matters.
    """
    event = EvolutionEvent(
        generation=prompt.generation,
        domain=domain,
        task_score=score,
        evolution_type=EvolutionType.RANDOM_STRATEGY
    )

    # Get all strategies the agent doesn't have
    all_strategies = STRATEGY_LIBRARY.get_all()
    available = [s for s in all_strategies if s.id not in prompt.get_strategy_ids()]

    if available:
        strategy = random.choice(available)
        prompt.add_strategy(strategy)
        event.strategy_added = strategy

    prompt.generation += 1
    return event


def evolve_no_change(
    prompt: SpecialistPrompt,
    domain: DomainType,
    score: float
) -> EvolutionEvent:
    """
    No evolution baseline: Record attempt but don't modify prompt.
    """
    event = EvolutionEvent(
        generation=prompt.generation,
        domain=domain,
        task_score=score,
        evolution_type=EvolutionType.NO_EVOLUTION
    )

    # Don't add strategies, just track
    prompt.generation += 1
    return event


def create_evolution_function(evolution_type: EvolutionType):
    """Factory to create the appropriate evolution function."""
    if evolution_type == EvolutionType.CUMULATIVE:
        return evolve_cumulative
    elif evolution_type == EvolutionType.RANDOM_STRATEGY:
        return evolve_random_strategy
    elif evolution_type == EvolutionType.NO_EVOLUTION:
        return evolve_no_change
    else:
        raise ValueError(f"Unknown evolution type: {evolution_type}")


def create_initial_prompt(agent_id: str) -> SpecialistPrompt:
    """Create an initial prompt for a new agent."""
    return SpecialistPrompt(
        agent_id=agent_id,
        generation=0,
        primary_domain=None,
        acquired_strategies=[],
        reference_examples=[],
        domain_wins={},
        domain_attempts={},
        total_wins=0,
        total_attempts=0
    )


def create_specialist_prompt(
    agent_id: str,
    domain: DomainType,
    num_strategies: int = 5,
    num_examples: int = 2
) -> SpecialistPrompt:
    """
    Create a hand-crafted specialist prompt for validation testing.

    This creates an agent that is pre-specialized in a domain.
    """
    prompt = SpecialistPrompt(
        agent_id=agent_id,
        generation=0,
        primary_domain=domain
    )

    # Add strategies from this domain
    domain_strategies = STRATEGY_LIBRARY.get_by_domain(domain)
    for strategy in domain_strategies[:num_strategies]:
        prompt.add_strategy(strategy)

    # Add examples from this domain
    domain_examples = EXAMPLE_LIBRARY.get_by_domain(domain)
    for example in domain_examples[:num_examples]:
        prompt.add_example(example)

    return prompt


def create_all_specialists() -> Dict[DomainType, SpecialistPrompt]:
    """Create one specialist for each domain (for validation testing)."""
    specialists = {}

    for i, domain in enumerate(DomainType):
        agent_id = f"specialist_{domain.value}"
        prompt = create_specialist_prompt(agent_id, domain)
        specialists[domain] = prompt

    return specialists


# ============================================================================
# PORTFOLIO AGENT - Paper 1 Aligned Implementation
# ============================================================================

@dataclass
class PortfolioAgent:
    """
    Agent with fixed-size strategy portfolio (Paper 1 aligned).

    Key differences from SpecialistPrompt:
    1. Fixed portfolio size (5 slots) - matches Paper 1's finite method set
    2. Thompson Sampling for strategy selection
    3. Paper 1 niche affinity formula for specialization tracking
    4. Winner-take-all update dynamics

    This implementation directly mirrors NicheAgent from Paper 1.
    """

    agent_id: str
    generation: int = 0

    # Fixed portfolio (5 slots max) - matches Paper 1
    portfolio: List[Strategy] = field(default_factory=list)
    MAX_PORTFOLIO_SIZE: int = 5

    # Niche affinity (continuous, Paper 1 formula)
    # Starts uniform, evolves toward specialization
    domain_affinity: Dict[DomainType, float] = field(default_factory=dict)

    # Performance tracking
    domain_wins: Dict[str, int] = field(default_factory=dict)
    domain_attempts: Dict[str, int] = field(default_factory=dict)
    total_wins: int = 0
    total_attempts: int = 0

    # Learning parameters (from Paper 1)
    learning_rate: float = 0.1  # η in Paper 1

    # Random state
    _rng: Optional[object] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize uniform affinity and RNG."""
        import numpy as np

        # Initialize uniform affinity across all domains
        if not self.domain_affinity:
            n_domains = len(DomainType)
            self.domain_affinity = {d: 1.0 / n_domains for d in DomainType}

        # Initialize RNG
        if self._rng is None:
            self._rng = np.random.default_rng()

    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        import numpy as np
        self._rng = np.random.default_rng(seed)

    def get_portfolio_ids(self) -> Set[str]:
        """Get IDs of strategies in portfolio."""
        return {s.id for s in self.portfolio}

    def select_strategy_to_add(self, domain: DomainType) -> Optional[Strategy]:
        """
        Use Thompson Sampling to select which strategy to add.

        Matches Paper 1's select_method() using Thompson Sampling.
        """
        from .strategy_beliefs import BELIEF_MANAGER

        return BELIEF_MANAGER.select_strategy_thompson(
            domain,
            exclude_ids=self.get_portfolio_ids()
        )

    def select_strategy_to_replace(self) -> Optional[Strategy]:
        """
        Use Thompson Sampling to select which strategy to REMOVE.

        Returns strategy with LOWEST sampled value (worst performing).
        """
        from .strategy_beliefs import BELIEF_MANAGER

        if not self.portfolio:
            return None

        portfolio_ids = [s.id for s in self.portfolio]
        replace_id = BELIEF_MANAGER.select_strategy_to_replace(portfolio_ids)

        if replace_id:
            return next((s for s in self.portfolio if s.id == replace_id), None)
        return None

    def add_strategy(self, strategy: Strategy) -> Optional[Strategy]:
        """
        Add strategy, potentially replacing one if at capacity.

        Returns the removed strategy (if any).
        """
        removed = None

        # Check if already in portfolio
        if strategy.id in self.get_portfolio_ids():
            return None

        if len(self.portfolio) >= self.MAX_PORTFOLIO_SIZE:
            removed = self.select_strategy_to_replace()
            if removed:
                self.portfolio.remove(removed)

        self.portfolio.append(strategy)
        return removed

    def update_affinity(self, domain: DomainType, won: bool):
        """
        Paper 1 niche affinity formula.

        From Paper 1:
            α_r ← α_r + η × (1 - α_r)  [for winning regime]
            α_r' ← α_r' - η/(R-1)       [for other regimes]
        Then normalize to maintain probability simplex.
        """
        if not won:
            return

        eta = self.learning_rate  # η = 0.1 as in Paper 1
        n_domains = len(DomainType)

        # Winner domain gets boost (Paper 1 formula)
        self.domain_affinity[domain] += eta * (1 - self.domain_affinity[domain])

        # Other domains get penalty
        for d in DomainType:
            if d != domain:
                self.domain_affinity[d] -= eta / (n_domains - 1)
                self.domain_affinity[d] = max(0.01, self.domain_affinity[d])

        # Normalize to maintain probability simplex
        total = sum(self.domain_affinity.values())
        self.domain_affinity = {d: a / total for d, a in self.domain_affinity.items()}

    def record_attempt(self, domain: DomainType, won: bool):
        """Record a task attempt and update affinity."""
        domain_key = domain.value
        self.domain_attempts[domain_key] = self.domain_attempts.get(domain_key, 0) + 1
        self.total_attempts += 1

        if won:
            self.domain_wins[domain_key] = self.domain_wins.get(domain_key, 0) + 1
            self.total_wins += 1

            # Update affinity using Paper 1 formula
            self.update_affinity(domain, won=True)

    def get_primary_domain(self) -> DomainType:
        """Get domain with highest affinity."""
        return max(self.domain_affinity, key=self.domain_affinity.get)

    def get_niche_strength(self, domain: DomainType) -> float:
        """
        How strong is this agent in the given domain?

        Combines affinity and success rate (matches Paper 1).
        """
        affinity = self.domain_affinity.get(domain, 0.1)

        domain_key = domain.value
        attempts = self.domain_attempts.get(domain_key, 0)
        if attempts == 0:
            return affinity

        success_rate = self.domain_wins.get(domain_key, 0) / attempts
        return 0.5 * affinity + 0.5 * success_rate

    def get_win_rate(self, domain: DomainType) -> float:
        """Get win rate for a specific domain."""
        domain_key = domain.value
        attempts = self.domain_attempts.get(domain_key, 0)
        if attempts == 0:
            return 0.0
        return self.domain_wins.get(domain_key, 0) / attempts

    def to_prompt_text(self) -> str:
        """Generate the full prompt text for the LLM."""
        sections = []

        # Header
        primary = self.get_primary_domain()
        primary_affinity = self.domain_affinity[primary]

        if primary_affinity > 0.2:  # Specialized
            spec = get_domain_spec(primary)
            sections.append(f"""# Agent {self.agent_id} | Generation {self.generation}
Role: {spec.name} Specialist (affinity: {primary_affinity:.0%})
Core Operation: {spec.core_operation}""")
        else:  # Generalist
            sections.append(f"""# Agent {self.agent_id} | Generation {self.generation}
Role: General AI Assistant (developing specialization)""")

        # Portfolio strategies
        if self.portfolio:
            sections.append("## Active Strategies")
            for strategy in self.portfolio:
                sections.append(f"• **{strategy.name}** ({strategy.domain.value})")
                sections.append(f"  {strategy.description}")

        # Performance summary
        if self.total_attempts > 0:
            sections.append(f"""## Performance
Total: {self.total_wins}/{self.total_attempts} wins ({self.total_wins/self.total_attempts:.0%})""")

            # Top 3 domains by affinity
            sorted_domains = sorted(
                self.domain_affinity.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            for domain, affinity in sorted_domains:
                wins = self.domain_wins.get(domain.value, 0)
                attempts = self.domain_attempts.get(domain.value, 0)
                if attempts > 0:
                    sections.append(f"  - {domain.value}: {wins}/{attempts} ({affinity:.0%} affinity)")

        return "\n\n".join(sections)

    def to_openai_message(self) -> Dict:
        """Convert to OpenAI message format."""
        return {
            "role": "system",
            "content": self.to_prompt_text()
        }

    def compute_lsi(self) -> float:
        """
        Compute LLM Specialization Index for this agent.

        LSI = 1 - (entropy / max_entropy)
        Higher LSI = more specialized
        """
        import numpy as np

        # Use affinity distribution
        values = np.array(list(self.domain_affinity.values()))

        if values.sum() == 0:
            return 0.0

        p = values / values.sum()
        entropy = -np.sum(p * np.log(p + 1e-10))
        max_entropy = np.log(len(p))

        lsi = 1 - entropy / max_entropy
        return float(np.clip(lsi, 0, 1))


def create_portfolio_agent(agent_id: str, seed: int = None) -> PortfolioAgent:
    """Create a new portfolio agent."""
    agent = PortfolioAgent(agent_id=agent_id)
    if seed is not None:
        agent.set_seed(seed)
    return agent


def create_portfolio_population(n_agents: int, base_seed: int = 42) -> List[PortfolioAgent]:
    """Create a population of portfolio agents with different seeds."""
    agents = []
    for i in range(n_agents):
        agent = create_portfolio_agent(f"agent_{i}", seed=base_seed + i * 100)
        agents.append(agent)
    return agents


if __name__ == "__main__":
    # Demo the evolution system
    print("=" * 60)
    print("CUMULATIVE EVOLUTION DEMO")
    print("=" * 60)

    # Create a new agent
    agent = create_initial_prompt("demo_agent")
    print(f"\nInitial prompt ({len(agent.to_prompt_text().split())} words):")
    print("-" * 40)
    print(agent.to_prompt_text())

    # Simulate winning some tasks
    print("\n" + "=" * 60)
    print("SIMULATING 5 WINS IN ARITHMETIC")
    print("=" * 60)

    for i in range(5):
        event = evolve_cumulative(agent, DomainType.ARITHMETIC, 0.9)
        agent.record_attempt(DomainType.ARITHMETIC, won=True)
        print(f"  Win {i+1}: Added strategy '{event.strategy_added.name if event.strategy_added else 'None'}'")

    print(f"\nEvolved prompt ({len(agent.to_prompt_text().split())} words):")
    print("-" * 40)
    print(agent.to_prompt_text())

    # Show specialists
    print("\n" + "=" * 60)
    print("CREATING ALL 10 SPECIALISTS")
    print("=" * 60)

    specialists = create_all_specialists()
    for domain, spec in specialists.items():
        word_count = len(spec.to_prompt_text().split())
        print(f"  {domain.value}: {len(spec.acquired_strategies)} strategies, {word_count} words")
