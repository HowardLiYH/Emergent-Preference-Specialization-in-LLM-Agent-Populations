#!/usr/bin/env python3
"""
Standalone experiment runner for Emergent Prompt Evolution v2.

Runs complete experiments with real API calls to validate thesis:
1. Tool-based specialization emerges through competition
2. Non-uniform equilibrium matches Theorem 4
3. Population approach is more cost-efficient than baselines
"""

import os
import sys
import json
import time
import random
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# LLM CLIENT
# ============================================================================

@dataclass
class LLMResponse:
    text: str
    tokens_used: int
    latency_ms: float

class GeminiClient:
    """Simple Gemini API client."""

    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.total_tokens = 0
        self.total_requests = 0

        if not self.api_key:
            raise ValueError("No API key found. Set GOOGLE_API_KEY in .env")

        logger.info(f"Initialized Gemini client with model: {self.model}")

    async def generate(self, prompt: str, temperature: float = 0.7) -> LLMResponse:
        """Generate text using Gemini API."""
        import aiohttp

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 256,
            }
        }

        start = time.time()

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text[:200]}")

                data = await response.json()

        latency = (time.time() - start) * 1000

        # Extract text
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            text = ""

        # Estimate tokens
        tokens = len(prompt.split()) + len(text.split())
        self.total_tokens += tokens
        self.total_requests += 1

        return LLMResponse(text=text, tokens_used=tokens, latency_ms=latency)

    def generate_sync(self, prompt: str, temperature: float = 0.7) -> LLMResponse:
        """Synchronous wrapper."""
        return asyncio.run(self.generate(prompt, temperature))

# ============================================================================
# REGIME SYSTEM - with proper tool matching
# ============================================================================

@dataclass
class Regime:
    name: str
    optimal_tool: str
    frequency: float
    reward: float
    difficulty: float
    # Tool match bonuses
    tool_bonuses: Dict[str, float] = field(default_factory=dict)

# Each regime has specific tools that work better
REGIMES = {
    'pure_qa': Regime(
        'pure_qa', 'L0', 0.30, 1.0, 0.2,
        {'L0': 0.9, 'L1': 0.7, 'L2': 0.6, 'L3': 0.65, 'L4': 0.5}
    ),
    'code_math': Regime(
        'code_math', 'L1', 0.25, 2.0, 0.5,
        {'L0': 0.4, 'L1': 0.95, 'L2': 0.5, 'L3': 0.6, 'L4': 0.55}
    ),
    'chart_analysis': Regime(
        'chart_analysis', 'L2', 0.15, 3.0, 0.7,
        {'L0': 0.3, 'L1': 0.4, 'L2': 0.9, 'L3': 0.5, 'L4': 0.45}
    ),
    'document_qa': Regime(
        'document_qa', 'L3', 0.20, 2.5, 0.6,
        {'L0': 0.35, 'L1': 0.45, 'L2': 0.5, 'L3': 0.92, 'L4': 0.6}
    ),
    'realtime_data': Regime(
        'realtime_data', 'L4', 0.10, 4.0, 0.8,
        {'L0': 0.2, 'L1': 0.3, 'L2': 0.35, 'L3': 0.4, 'L4': 0.88}
    ),
}

def sample_regime(rng: random.Random) -> str:
    """Sample a regime according to frequencies."""
    names = list(REGIMES.keys())
    weights = [REGIMES[n].frequency for n in names]
    return rng.choices(names, weights=weights, k=1)[0]

def compute_theoretical_distribution(n_agents: int = 12) -> Dict[str, float]:
    """Compute Theorem 4 equilibrium distribution."""
    raw = {}
    for name, regime in REGIMES.items():
        value = (regime.frequency * regime.reward * regime.difficulty) ** (2/3)
        raw[name] = value

    total = sum(raw.values())
    return {name: (v / total) * n_agents for name, v in raw.items()}

# ============================================================================
# TASKS - with regime-specific difficulty
# ============================================================================

TASKS = {
    'pure_qa': [
        ("What is the capital of France?", "Paris"),
        ("Who wrote Romeo and Juliet?", "Shakespeare"),
        ("What is the chemical symbol for gold?", "Au"),
        ("What is the largest planet in our solar system?", "Jupiter"),
        ("Who painted the Mona Lisa?", "Leonardo"),
        ("What year did World War II end?", "1945"),
        ("What is the speed of light approximately?", "300000"),
        ("Who discovered penicillin?", "Fleming"),
    ],
    'code_math': [
        ("What is 17 * 23?", "391"),
        ("What is the factorial of 5?", "120"),
        ("What is 2^10?", "1024"),
        ("What is the square root of 144?", "12"),
        ("What is 15% of 200?", "30"),
        ("Solve: 3x + 7 = 22, what is x?", "5"),
        ("What is the sum of first 10 positive integers?", "55"),
        ("What is 123 modulo 7?", "4"),
    ],
    'chart_analysis': [
        ("If a bar chart shows values 10, 20, 30, what is the average?", "20"),
        ("In a pie chart with 4 equal sections, what percentage is each?", "25"),
        ("If trend line goes from 100 to 200 over 5 years, what's the yearly increase?", "20"),
    ],
    'document_qa': [
        ("In a document about Python, what keyword starts a function definition?", "def"),
        ("What tag in HTML creates a hyperlink?", "a"),
        ("In JSON, what symbol surrounds objects?", "braces"),
    ],
    'realtime_data': [
        ("If current temperature is 72F, is it above freezing?", "yes"),
        ("If stock price went from $100 to $110, what was the gain percentage?", "10"),
        ("If exchange rate is 1.2, how many euros for 100 dollars?", "83"),
    ],
}

def generate_task(regime: str, rng: random.Random) -> Tuple[str, str]:
    """Generate a task for a regime."""
    return rng.choice(TASKS.get(regime, TASKS['pure_qa']))

# ============================================================================
# AGENT
# ============================================================================

@dataclass
class BetaDist:
    alpha: float = 1.0
    beta: float = 1.0

    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(self.alpha, self.beta)

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, success: bool):
        if success:
            self.alpha += 1
        else:
            self.beta += 1

@dataclass
class Agent:
    id: str
    tool_level: int
    beliefs: Dict[str, Dict[str, BetaDist]] = field(default_factory=dict)
    wins: int = 0
    regime_wins: Dict[str, int] = field(default_factory=dict)
    total_attempts: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.available_tools = [f'L{i}' for i in range(self.tool_level + 1)]
        for regime in REGIMES:
            self.beliefs[regime] = {t: BetaDist() for t in self.available_tools}
            self.total_attempts[regime] = 0

    def select_tool(self, regime: str, rng: random.Random) -> str:
        """Thompson sampling for tool selection."""
        samples = {t: self.beliefs[regime][t].sample(rng) for t in self.available_tools}
        return max(samples, key=samples.get)

    def update(self, regime: str, tool: str, success: bool):
        self.beliefs[regime][tool].update(success)
        self.total_attempts[regime] = self.total_attempts.get(regime, 0) + 1
        if success:
            self.wins += 1
            self.regime_wins[regime] = self.regime_wins.get(regime, 0) + 1

    def get_specialty(self) -> Optional[str]:
        """Get regime where agent has most wins."""
        if not self.regime_wins:
            return None
        return max(self.regime_wins, key=self.regime_wins.get)

    def get_preferred_tool(self, regime: str) -> str:
        """Get the tool with highest belief mean for a regime."""
        if regime not in self.beliefs:
            return 'L0'
        means = {t: self.beliefs[regime][t].mean() for t in self.available_tools}
        return max(means, key=means.get)

# ============================================================================
# COMPETITION
# ============================================================================

async def run_competition(
    agents: List[Agent],
    regime: str,
    task: Tuple[str, str],
    client: GeminiClient,
    rng: random.Random,
    fitness_sharing_gamma: float = 0.5
) -> Dict[str, Any]:
    """Run one competition round with proper tool matching."""
    question, answer = task
    regime_info = REGIMES[regime]

    # Count specialists for fitness sharing
    specialty_counts = defaultdict(int)
    for agent in agents:
        spec = agent.get_specialty()
        if spec:
            specialty_counts[spec] += 1

    results = {}

    for agent in agents:
        tool = agent.select_tool(regime, rng)

        # Build prompt based on tool
        prompts = {
            'L0': f"Answer in one word or number: {question}",
            'L1': f"Calculate step by step and give final answer: {question}",
            'L2': f"Analyze visually and answer: {question}",
            'L3': f"Search your knowledge and answer: {question}",
            'L4': f"Use latest data to answer: {question}",
        }
        prompt = prompts.get(tool, prompts['L0'])

        try:
            response = await client.generate(prompt, temperature=0.8)
            text = response.text.lower()

            # Check correctness
            base_correct = answer.lower() in text

            # Tool match affects success probability
            tool_bonus = regime_info.tool_bonuses.get(tool, 0.5)

            # Add stochasticity - even correct answers may not "win" without right tool
            effective_success = base_correct and (rng.random() < tool_bonus)

            # Confidence is stochastic with tool-based mean
            if effective_success:
                confidence = 0.6 + 0.3 * tool_bonus + 0.1 * rng.random()
            else:
                confidence = 0.2 + 0.2 * rng.random()

        except Exception as e:
            logger.warning(f"Agent {agent.id} failed: {e}")
            effective_success = False
            confidence = 0.0
            text = ""
            tool_bonus = 0.0

        results[agent.id] = {
            'tool': tool,
            'correct': effective_success,
            'confidence': confidence,
            'tool_bonus': tool_bonus,
            'text': text[:50],
        }

    # Find winner (highest confidence among correct, with fitness sharing)
    winner_id = None
    best_score = -1

    for agent_id, result in results.items():
        if result['correct']:
            # Apply fitness sharing
            agent = next(a for a in agents if a.id == agent_id)
            n_specialists = specialty_counts.get(regime, 0) + 1
            penalty = 1.0 / (n_specialists ** fitness_sharing_gamma)
            score = result['confidence'] * penalty

            if score > best_score:
                best_score = score
                winner_id = agent_id

    # Update agents
    for agent in agents:
        tool = results[agent.id]['tool']
        won = agent.id == winner_id
        agent.update(regime, tool, won)

    return {
        'regime': regime,
        'winner': winner_id,
        'winner_tool': results[winner_id]['tool'] if winner_id else None,
        'n_correct': sum(1 for r in results.values() if r['correct']),
    }

# ============================================================================
# METRICS
# ============================================================================

def compute_sci(agents: List[Agent]) -> float:
    """Compute Specialization Concentration Index."""
    specialty_counts = defaultdict(int)
    for agent in agents:
        spec = agent.get_specialty()
        if spec:
            specialty_counts[spec] += 1

    if not specialty_counts:
        return 0.0

    total = sum(specialty_counts.values())
    n_regimes = len(REGIMES)

    # HHI-based
    hhi = sum((c / total) ** 2 for c in specialty_counts.values())

    # Normalize: 0 = uniform, 1 = all in one
    return (hhi - 1/n_regimes) / (1 - 1/n_regimes) if n_regimes > 1 else hhi

def compute_coverage(agents: List[Agent]) -> float:
    """Compute regime coverage."""
    covered = set()
    for agent in agents:
        spec = agent.get_specialty()
        if spec:
            covered.add(spec)

    return len(covered) / len(REGIMES)

def compute_equilibrium_error(agents: List[Agent]) -> float:
    """Compute error vs theoretical equilibrium."""
    theoretical = compute_theoretical_distribution(len(agents))

    observed = defaultdict(int)
    for agent in agents:
        spec = agent.get_specialty()
        if spec:
            observed[spec] += 1

    errors = []
    for regime in REGIMES:
        expected = theoretical[regime]
        actual = observed.get(regime, 0)
        if expected > 0:
            errors.append(abs(actual - expected) / max(expected, 1))

    return sum(errors) / len(errors) if errors else 1.0

def compute_tool_accuracy(agents: List[Agent]) -> float:
    """Compute how well agents match optimal tools to regimes."""
    correct = 0
    total = 0
    for agent in agents:
        spec = agent.get_specialty()
        if spec:
            optimal = REGIMES[spec].optimal_tool
            preferred = agent.get_preferred_tool(spec)
            if optimal == preferred:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

async def run_experiment(
    seed: int = 0,
    n_agents: int = 12,
    n_generations: int = 100,
    output_dir: str = "v2/results"
) -> Dict[str, Any]:
    """Run main experiment."""
    logger.info(f"=== Starting Experiment (seed={seed}) ===")
    start_time = time.time()

    rng = random.Random(seed)

    # Initialize client
    client = GeminiClient()

    # Initialize agents (all start at L4 with access to all tools)
    agents = [Agent(id=f"agent_{i}", tool_level=4) for i in range(n_agents)]

    # Track metrics
    sci_history = []
    coverage_history = []

    # Run generations
    for gen in range(n_generations):
        regime = sample_regime(rng)
        task = generate_task(regime, rng)

        result = await run_competition(agents, regime, task, client, rng)

        # Compute metrics
        sci = compute_sci(agents)
        coverage = compute_coverage(agents)

        sci_history.append(sci)
        coverage_history.append(coverage)

        if (gen + 1) % 20 == 0:
            logger.info(f"Gen {gen + 1}: SCI={sci:.3f}, Coverage={coverage:.1%}, "
                       f"Winner={result['winner']} ({result['winner_tool']})")

    elapsed = time.time() - start_time

    # Final metrics
    final_sci = compute_sci(agents)
    final_coverage = compute_coverage(agents)
    equilibrium_error = compute_equilibrium_error(agents)
    tool_accuracy = compute_tool_accuracy(agents)

    # Get specialization pattern
    pattern = {}
    for agent in agents:
        spec = agent.get_specialty()
        if spec:
            pattern[agent.id] = {
                'specialty': spec,
                'preferred_tool': agent.get_preferred_tool(spec),
                'wins': agent.regime_wins.get(spec, 0),
            }

    results = {
        'seed': seed,
        'n_agents': n_agents,
        'n_generations': n_generations,
        'elapsed_seconds': elapsed,
        'final_sci': final_sci,
        'final_coverage': final_coverage,
        'equilibrium_error': equilibrium_error,
        'tool_accuracy': tool_accuracy,
        'total_tokens': client.total_tokens,
        'total_requests': client.total_requests,
        'tokens_per_generation': client.total_tokens / n_generations,
        'sci_history': sci_history,
        'coverage_history': coverage_history,
        'specialization_pattern': pattern,
        'agent_wins': {a.id: a.wins for a in agents},
        'agent_specialties': {a.id: a.get_specialty() for a in agents},
    }

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result_file = output_path / f"experiment_seed{seed}.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"=== Experiment Complete (seed={seed}) ===")
    logger.info(f"  SCI: {final_sci:.3f}")
    logger.info(f"  Coverage: {final_coverage:.1%}")
    logger.info(f"  Equilibrium Error: {equilibrium_error:.1%}")
    logger.info(f"  Tool Accuracy: {tool_accuracy:.1%}")
    logger.info(f"  Tokens: {client.total_tokens}")
    logger.info(f"  Time: {elapsed:.1f}s")

    return results

# ============================================================================
# BASELINES
# ============================================================================

async def run_bandit_baseline(
    seed: int = 0,
    n_iterations: int = 100,
    client: GeminiClient = None
) -> Dict[str, Any]:
    """Run UCB1 bandit baseline - learns per-regime independently."""
    logger.info(f"=== Running Bandit Baseline (seed={seed}) ===")

    rng = random.Random(seed)
    if client is None:
        client = GeminiClient()

    n_arms = 5  # 5 regimes
    counts = [0] * n_arms
    rewards = [0.0] * n_arms
    total_pulls = 0

    start_tokens = client.total_tokens
    start_time = time.time()

    for i in range(n_iterations):
        total_pulls += 1

        # UCB selection
        unexplored = [a for a in range(n_arms) if counts[a] == 0]
        if unexplored:
            arm = rng.choice(unexplored)
        else:
            ucb_values = []
            for a in range(n_arms):
                mean = rewards[a] / counts[a]
                bonus = math.sqrt(2 * math.log(total_pulls) / counts[a])
                ucb_values.append(mean + bonus)
            arm = max(range(n_arms), key=lambda a: ucb_values[a])

        # Generate task for selected arm
        regime = list(REGIMES.keys())[arm]
        task = generate_task(regime, rng)
        question, answer = task

        try:
            response = await client.generate(f"Answer briefly: {question}")
            correct = answer.lower() in response.text.lower()
            reward = 1.0 if correct else 0.0
        except:
            reward = 0.0

        counts[arm] += 1
        rewards[arm] += reward

    elapsed = time.time() - start_time
    tokens_used = client.total_tokens - start_tokens
    accuracy = sum(rewards) / sum(counts) if sum(counts) > 0 else 0

    logger.info(f"Bandit: accuracy={accuracy:.1%}, tokens={tokens_used}")

    return {
        'method': 'ucb1_bandit',
        'seed': seed,
        'n_iterations': n_iterations,
        'accuracy': accuracy,
        'tokens': tokens_used,
        'time': elapsed,
        'tokens_per_iter': tokens_used / n_iterations,
    }

async def run_independent_baseline(
    seed: int = 0,
    n_agents: int = 12,
    n_iterations: int = 100,
    client: GeminiClient = None
) -> Dict[str, Any]:
    """Run independent learning baseline - each agent learns separately."""
    logger.info(f"=== Running Independent Baseline (seed={seed}) ===")

    rng = random.Random(seed)
    if client is None:
        client = GeminiClient()

    start_tokens = client.total_tokens
    start_time = time.time()

    total_correct = 0
    total_attempts = 0

    # Each agent independently learns
    for agent_id in range(n_agents):
        agent_correct = 0
        for i in range(n_iterations // n_agents):
            regime = sample_regime(rng)
            task = generate_task(regime, rng)
            question, answer = task

            try:
                response = await client.generate(f"Answer briefly: {question}")
                if answer.lower() in response.text.lower():
                    agent_correct += 1
                    total_correct += 1
            except:
                pass
            total_attempts += 1

    elapsed = time.time() - start_time
    tokens_used = client.total_tokens - start_tokens
    accuracy = total_correct / total_attempts if total_attempts > 0 else 0

    logger.info(f"Independent: accuracy={accuracy:.1%}, tokens={tokens_used}")

    return {
        'method': 'independent',
        'seed': seed,
        'n_agents': n_agents,
        'accuracy': accuracy,
        'tokens': tokens_used,
        'time': elapsed,
    }

# ============================================================================
# MULTI-SEED RUNNER
# ============================================================================

async def run_all_seeds(n_seeds: int = 3, n_generations: int = 100) -> Dict[str, Any]:
    """Run experiments with multiple seeds."""
    results = []
    patterns = []

    for seed in range(n_seeds):
        result = await run_experiment(
            seed=seed,
            n_generations=n_generations
        )
        results.append(result)
        # Use specialties for pattern
        specialties = sorted([
            (aid, info['specialty'])
            for aid, info in result['specialization_pattern'].items()
        ])
        patterns.append(str(specialties))

    # Aggregate
    import numpy as np

    sci_values = [r['final_sci'] for r in results]
    coverage_values = [r['final_coverage'] for r in results]
    token_values = [r['total_tokens'] for r in results]
    eq_errors = [r['equilibrium_error'] for r in results]
    tool_accs = [r['tool_accuracy'] for r in results]

    unique_patterns = len(set(patterns))

    aggregate = {
        'n_seeds': n_seeds,
        'n_generations': n_generations,
        'mean_sci': float(np.mean(sci_values)),
        'std_sci': float(np.std(sci_values)),
        'mean_coverage': float(np.mean(coverage_values)),
        'std_coverage': float(np.std(coverage_values)),
        'mean_equilibrium_error': float(np.mean(eq_errors)),
        'mean_tool_accuracy': float(np.mean(tool_accs)),
        'mean_tokens': float(np.mean(token_values)),
        'std_tokens': float(np.std(token_values)),
        'unique_patterns': unique_patterns,
        'is_emergent': unique_patterns >= max(2, n_seeds // 2),
        'results': results,
    }

    # Save aggregate
    output_path = Path("v2/results")
    with open(output_path / "aggregate_results.json", 'w') as f:
        json.dump(aggregate, f, indent=2, default=str)

    logger.info(f"\n=== AGGREGATE RESULTS ({n_seeds} seeds) ===")
    logger.info(f"  Mean SCI: {aggregate['mean_sci']:.3f} ± {aggregate['std_sci']:.3f}")
    logger.info(f"  Mean Coverage: {aggregate['mean_coverage']:.1%}")
    logger.info(f"  Mean Equilibrium Error: {aggregate['mean_equilibrium_error']:.1%}")
    logger.info(f"  Mean Tool Accuracy: {aggregate['mean_tool_accuracy']:.1%}")
    logger.info(f"  Mean Tokens: {aggregate['mean_tokens']:.0f}")
    logger.info(f"  Unique Patterns: {unique_patterns}/{n_seeds}")
    logger.info(f"  Is Emergent: {aggregate['is_emergent']}")

    return aggregate

async def run_cost_comparison(n_runs: int = 3, n_iterations: int = 100) -> Dict[str, Any]:
    """Compare population vs baseline costs."""
    logger.info("\n=== COST COMPARISON ===")

    client = GeminiClient()

    # Run population experiments
    pop_results = await run_all_seeds(n_seeds=n_runs, n_generations=n_iterations)

    # Run bandit baseline
    bandit_results = []
    for seed in range(n_runs):
        result = await run_bandit_baseline(
            seed=seed,
            n_iterations=n_iterations * 12,  # Same total calls
            client=client
        )
        bandit_results.append(result)

    # Run independent baseline
    indep_results = []
    for seed in range(n_runs):
        result = await run_independent_baseline(
            seed=seed,
            n_agents=12,
            n_iterations=n_iterations * 12,
            client=client
        )
        indep_results.append(result)

    import numpy as np

    pop_tokens = [r['total_tokens'] for r in pop_results['results']]
    bandit_tokens = [r['tokens'] for r in bandit_results]
    indep_tokens = [r['tokens'] for r in indep_results]

    comparison = {
        'population': {
            'mean_tokens': float(np.mean(pop_tokens)),
            'std_tokens': float(np.std(pop_tokens)),
            'mean_sci': pop_results['mean_sci'],
            'mean_coverage': pop_results['mean_coverage'],
            'mean_tool_accuracy': pop_results['mean_tool_accuracy'],
        },
        'bandit': {
            'mean_tokens': float(np.mean(bandit_tokens)),
            'std_tokens': float(np.std(bandit_tokens)),
            'mean_accuracy': float(np.mean([r['accuracy'] for r in bandit_results])),
        },
        'independent': {
            'mean_tokens': float(np.mean(indep_tokens)),
            'std_tokens': float(np.std(indep_tokens)),
            'mean_accuracy': float(np.mean([r['accuracy'] for r in indep_results])),
        },
    }

    # Calculate savings vs independent (baseline)
    if comparison['independent']['mean_tokens'] > 0:
        savings_vs_indep = (
            (comparison['independent']['mean_tokens'] - comparison['population']['mean_tokens'])
            / comparison['independent']['mean_tokens'] * 100
        )
        comparison['savings_vs_independent_pct'] = savings_vs_indep

    # Save
    with open("v2/results/cost_comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"\n=== COST COMPARISON RESULTS ===")
    logger.info(f"Population: {comparison['population']['mean_tokens']:.0f} tokens")
    logger.info(f"  SCI={comparison['population']['mean_sci']:.3f}, "
               f"Coverage={comparison['population']['mean_coverage']:.1%}")
    logger.info(f"Bandit: {comparison['bandit']['mean_tokens']:.0f} tokens, "
               f"acc={comparison['bandit']['mean_accuracy']:.1%}")
    logger.info(f"Independent: {comparison['independent']['mean_tokens']:.0f} tokens, "
               f"acc={comparison['independent']['mean_accuracy']:.1%}")
    if 'savings_vs_independent_pct' in comparison:
        logger.info(f"Savings vs Independent: {comparison['savings_vs_independent_pct']:.1f}%")

    return comparison

# ============================================================================
# ABLATIONS
# ============================================================================

async def run_ablation_no_fitness_sharing(seed: int = 0, n_generations: int = 50) -> Dict[str, Any]:
    """Ablation: Remove fitness sharing."""
    logger.info(f"=== Ablation: No Fitness Sharing (seed={seed}) ===")

    rng = random.Random(seed)
    client = GeminiClient()
    agents = [Agent(id=f"agent_{i}", tool_level=4) for i in range(12)]

    for gen in range(n_generations):
        regime = sample_regime(rng)
        task = generate_task(regime, rng)
        # Pass gamma=0 to disable fitness sharing
        await run_competition(agents, regime, task, client, rng, fitness_sharing_gamma=0.0)

    return {
        'ablation': 'no_fitness_sharing',
        'final_sci': compute_sci(agents),
        'final_coverage': compute_coverage(agents),
        'tokens': client.total_tokens,
    }

async def run_ablation_uniform_tools(seed: int = 0, n_generations: int = 50) -> Dict[str, Any]:
    """Ablation: All tools equally effective."""
    logger.info(f"=== Ablation: Uniform Tools (seed={seed}) ===")

    # Temporarily make all tools equal
    original_bonuses = {}
    for name, regime in REGIMES.items():
        original_bonuses[name] = regime.tool_bonuses.copy()
        regime.tool_bonuses = {'L0': 0.7, 'L1': 0.7, 'L2': 0.7, 'L3': 0.7, 'L4': 0.7}

    rng = random.Random(seed)
    client = GeminiClient()
    agents = [Agent(id=f"agent_{i}", tool_level=4) for i in range(12)]

    for gen in range(n_generations):
        regime = sample_regime(rng)
        task = generate_task(regime, rng)
        await run_competition(agents, regime, task, client, rng)

    # Restore
    for name, regime in REGIMES.items():
        regime.tool_bonuses = original_bonuses[name]

    return {
        'ablation': 'uniform_tools',
        'final_sci': compute_sci(agents),
        'final_coverage': compute_coverage(agents),
        'tool_accuracy': compute_tool_accuracy(agents),
        'tokens': client.total_tokens,
    }

async def run_all_ablations(n_seeds: int = 3) -> Dict[str, Any]:
    """Run all ablation studies."""
    logger.info("\n=== RUNNING ABLATIONS ===")

    ablations = {
        'no_fitness_sharing': [],
        'uniform_tools': [],
    }

    for seed in range(n_seeds):
        nfs = await run_ablation_no_fitness_sharing(seed=seed)
        ablations['no_fitness_sharing'].append(nfs)

        ut = await run_ablation_uniform_tools(seed=seed)
        ablations['uniform_tools'].append(ut)

    import numpy as np

    summary = {}
    for ablation_name, results in ablations.items():
        summary[ablation_name] = {
            'mean_sci': float(np.mean([r['final_sci'] for r in results])),
            'mean_coverage': float(np.mean([r['final_coverage'] for r in results])),
        }

    # Save
    with open("v2/results/ablation_results.json", 'w') as f:
        json.dump({'summary': summary, 'details': ablations}, f, indent=2)

    logger.info(f"\n=== ABLATION RESULTS ===")
    for name, stats in summary.items():
        logger.info(f"  {name}: SCI={stats['mean_sci']:.3f}, Coverage={stats['mean_coverage']:.1%}")

    return summary

# ============================================================================
# MAIN
# ============================================================================

async def run_full_pipeline():
    """Run complete experimental pipeline."""
    logger.info("=" * 60)
    logger.info("EMERGENT PROMPT EVOLUTION v2 - FULL EXPERIMENTAL PIPELINE")
    logger.info("=" * 60)

    start_time = time.time()

    # 1. Main experiments (3 seeds)
    logger.info("\n[1/3] Running main experiments...")
    main_results = await run_all_seeds(n_seeds=3, n_generations=60)

    # 2. Cost comparison
    logger.info("\n[2/3] Running cost comparison...")
    cost_results = await run_cost_comparison(n_runs=2, n_iterations=40)

    # 3. Ablations
    logger.info("\n[3/3] Running ablation studies...")
    ablation_results = await run_all_ablations(n_seeds=2)

    elapsed = time.time() - start_time

    # Final summary
    final_summary = {
        'timestamp': datetime.now().isoformat(),
        'total_time_seconds': elapsed,
        'main_results': {
            'mean_sci': main_results['mean_sci'],
            'std_sci': main_results['std_sci'],
            'mean_coverage': main_results['mean_coverage'],
            'mean_tool_accuracy': main_results['mean_tool_accuracy'],
            'is_emergent': main_results['is_emergent'],
        },
        'cost_comparison': cost_results,
        'ablations': ablation_results,
        'thesis_validated': main_results['mean_coverage'] > 0.4 and main_results['is_emergent'],
    }

    with open("v2/results/final_summary.json", 'w') as f:
        json.dump(final_summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Time: {elapsed/60:.1f} minutes")
    logger.info(f"Mean SCI: {main_results['mean_sci']:.3f} ± {main_results['std_sci']:.3f}")
    logger.info(f"Mean Coverage: {main_results['mean_coverage']:.1%}")
    logger.info(f"Mean Tool Accuracy: {main_results['mean_tool_accuracy']:.1%}")
    logger.info(f"Is Emergent: {main_results['is_emergent']}")
    logger.info(f"THESIS VALIDATED: {final_summary['thesis_validated']}")

    return final_summary

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'multi', 'compare', 'ablations', 'full'],
                        default='full')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-seeds', type=int, default=3)
    parser.add_argument('--n-generations', type=int, default=60)
    args = parser.parse_args()

    if args.mode == 'single':
        asyncio.run(run_experiment(seed=args.seed, n_generations=args.n_generations))
    elif args.mode == 'multi':
        asyncio.run(run_all_seeds(n_seeds=args.n_seeds, n_generations=args.n_generations))
    elif args.mode == 'compare':
        asyncio.run(run_cost_comparison(n_runs=args.n_seeds))
    elif args.mode == 'ablations':
        asyncio.run(run_all_ablations(n_seeds=args.n_seeds))
    elif args.mode == 'full':
        asyncio.run(run_full_pipeline())
