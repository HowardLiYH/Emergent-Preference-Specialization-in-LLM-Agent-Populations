#!/usr/bin/env python3
"""
Architecture Comparison Experiment.

Compares population structures with FIXED inner algorithm (Thompson Sampling).
Only the population structure varies.

Architectures:
1. Independent Training - No interaction
2. MARL Shared Critic - Shared learning
3. Tournament Selection - Competition, no fitness sharing
4. Market-Based Bidding - Explicit prices
5. CSE (Ours) - Competition + fitness sharing
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
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import architectures
from experiments.architectures.base import Regime, BaseAgent
from experiments.architectures.independent import IndependentTraining
from experiments.architectures.marl_shared import MARLSharedCritic
from experiments.architectures.tournament import TournamentSelection
from experiments.architectures.market import MarketBasedBidding
from experiments.architectures.cse import CompetitiveSpecialistEcosystem


# ============================================================================
# LLM CLIENT
# ============================================================================

class GeminiClient:
    """Simple Gemini API client."""
    
    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.total_tokens = 0
        self.total_requests = 0
        
        if not self.api_key:
            raise ValueError("No API key found. Set GOOGLE_API_KEY in .env")
    
    async def generate(self, prompt: str, temperature: float = 0.7) -> Tuple[str, int]:
        """Generate text, return (response, tokens)."""
        import aiohttp
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 256,
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    return "", 10  # Return empty on error
                data = await response.json()
        
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            text = ""
        
        tokens = len(prompt.split()) + len(text.split())
        self.total_tokens += tokens
        self.total_requests += 1
        
        return text, tokens


# ============================================================================
# REGIME DEFINITIONS
# ============================================================================

REGIMES = {
    'pure_qa': Regime(
        name='pure_qa', optimal_tool='L0', frequency=0.30, reward=1.0, difficulty=0.2,
        tool_bonuses={'L0': 0.9, 'L1': 0.7, 'L2': 0.6, 'L3': 0.65, 'L4': 0.5}
    ),
    'code_math': Regime(
        name='code_math', optimal_tool='L1', frequency=0.25, reward=2.0, difficulty=0.5,
        tool_bonuses={'L0': 0.4, 'L1': 0.95, 'L2': 0.5, 'L3': 0.6, 'L4': 0.55}
    ),
    'chart_analysis': Regime(
        name='chart_analysis', optimal_tool='L2', frequency=0.15, reward=3.0, difficulty=0.7,
        tool_bonuses={'L0': 0.3, 'L1': 0.4, 'L2': 0.9, 'L3': 0.5, 'L4': 0.45}
    ),
    'document_qa': Regime(
        name='document_qa', optimal_tool='L3', frequency=0.20, reward=2.5, difficulty=0.6,
        tool_bonuses={'L0': 0.35, 'L1': 0.45, 'L2': 0.5, 'L3': 0.92, 'L4': 0.6}
    ),
    'realtime_data': Regime(
        name='realtime_data', optimal_tool='L4', frequency=0.10, reward=4.0, difficulty=0.8,
        tool_bonuses={'L0': 0.2, 'L1': 0.3, 'L2': 0.35, 'L3': 0.4, 'L4': 0.88}
    ),
}


# ============================================================================
# TASKS
# ============================================================================

TASKS = {
    'pure_qa': [
        ("What is the capital of France?", "Paris"),
        ("Who wrote Romeo and Juliet?", "Shakespeare"),
        ("What is the chemical symbol for gold?", "Au"),
        ("What is the largest planet?", "Jupiter"),
        ("Who painted the Mona Lisa?", "Leonardo"),
    ],
    'code_math': [
        ("What is 17 * 23?", "391"),
        ("What is the factorial of 5?", "120"),
        ("What is 2^10?", "1024"),
        ("What is the square root of 144?", "12"),
        ("What is 15% of 200?", "30"),
    ],
    'chart_analysis': [
        ("If a bar chart shows 10, 20, 30, what is the average?", "20"),
        ("In a pie chart with 4 equal sections, what percentage each?", "25"),
    ],
    'document_qa': [
        ("What keyword starts a Python function?", "def"),
        ("What HTML tag creates a link?", "a"),
    ],
    'realtime_data': [
        ("If price went from $100 to $110, what's the gain %?", "10"),
        ("Is 72F above freezing?", "yes"),
    ],
}


def sample_regime(rng: random.Random) -> str:
    """Sample regime according to frequencies."""
    names = list(REGIMES.keys())
    weights = [REGIMES[n].frequency for n in names]
    return rng.choices(names, weights=weights, k=1)[0]


def generate_task(regime: str, rng: random.Random) -> Tuple[str, str]:
    """Generate a task for a regime."""
    return rng.choice(TASKS.get(regime, TASKS['pure_qa']))


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

async def create_evaluator(client: GeminiClient, regimes: Dict[str, Regime]):
    """Create an evaluation function that uses the LLM."""
    
    async def evaluate(agent: BaseAgent, tool: str, question: str, answer: str) -> Tuple[bool, int]:
        """Evaluate agent's response."""
        prompts = {
            'L0': f"Answer briefly: {question}",
            'L1': f"Calculate and answer: {question}",
            'L2': f"Analyze and answer: {question}",
            'L3': f"Search knowledge and answer: {question}",
            'L4': f"Use latest data to answer: {question}",
        }
        prompt = prompts.get(tool, prompts['L0'])
        
        try:
            text, tokens = await client.generate(prompt, temperature=0.8)
            correct = answer.lower() in text.lower()
            return correct, tokens
        except Exception as e:
            return False, 10
    
    return evaluate


# ============================================================================
# RUN ARCHITECTURE
# ============================================================================

async def run_architecture(
    arch_class,
    n_agents: int,
    n_generations: int,
    seed: int,
    client: GeminiClient
) -> Dict:
    """Run a single architecture experiment."""
    rng = random.Random(seed)
    
    # Create architecture
    arch = arch_class(n_agents, REGIMES)
    
    # Create evaluator
    evaluate = await create_evaluator(client, REGIMES)
    
    # Track metrics over time
    coverage_history = []
    sci_history = []
    
    # Run training
    for gen in range(n_generations):
        regime = sample_regime(rng)
        task = generate_task(regime, rng)
        
        result = await asyncio.to_thread(
            lambda: asyncio.run(run_step(arch, task, regime, rng, evaluate))
        )
        
        if (gen + 1) % 20 == 0:
            coverage = arch.compute_coverage()
            sci = arch.compute_sci()
            coverage_history.append(coverage)
            sci_history.append(sci)
            logger.info(f"  {arch.name} Gen {gen+1}: Coverage={coverage:.1%}, SCI={sci:.3f}")
    
    # Get final results
    results = arch.get_results()
    results['name'] = arch.name
    results['seed'] = seed
    results['n_agents'] = n_agents
    results['n_generations'] = n_generations
    results['coverage_history'] = coverage_history
    results['sci_history'] = sci_history
    
    return results


async def run_step(arch, task, regime, rng, evaluate):
    """Run a single step with async evaluation."""
    
    async def eval_fn(agent, tool, question, answer):
        return await evaluate(agent, tool, question, answer)
    
    # For sync architectures, wrap the evaluation
    def sync_eval(agent, tool, question, answer):
        return asyncio.get_event_loop().run_until_complete(
            eval_fn(agent, tool, question, answer)
        )
    
    return arch.train_step(task, regime, rng, sync_eval)


# ============================================================================
# MAIN COMPARISON
# ============================================================================

async def run_comparison(
    n_agents: int = 12,
    n_generations: int = 60,
    n_seeds: int = 3
) -> Dict:
    """Run full architecture comparison."""
    logger.info("=" * 60)
    logger.info("ARCHITECTURE COMPARISON EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Agents: {n_agents}, Generations: {n_generations}, Seeds: {n_seeds}")
    
    architectures = [
        IndependentTraining,
        MARLSharedCritic,
        TournamentSelection,
        MarketBasedBidding,
        CompetitiveSpecialistEcosystem,
    ]
    
    all_results = defaultdict(list)
    client = GeminiClient()
    
    for seed in range(n_seeds):
        logger.info(f"\n--- Seed {seed} ---")
        
        for arch_class in architectures:
            logger.info(f"\nRunning {arch_class.__name__}...")
            start = time.time()
            
            result = await run_architecture(
                arch_class, n_agents, n_generations, seed, client
            )
            
            elapsed = time.time() - start
            result['elapsed'] = elapsed
            
            all_results[result['name']].append(result)
            
            logger.info(f"  Completed in {elapsed:.1f}s")
            logger.info(f"  Final Coverage: {result['coverage']:.1%}")
            logger.info(f"  Final SCI: {result['sci']:.3f}")
            logger.info(f"  Tokens: {result['total_tokens']}")
    
    # Aggregate results
    summary = {}
    for name, results in all_results.items():
        coverages = [r['coverage'] for r in results]
        scis = [r['sci'] for r in results]
        tokens = [r['total_tokens'] for r in results]
        
        summary[name] = {
            'mean_coverage': float(np.mean(coverages)),
            'std_coverage': float(np.std(coverages)),
            'mean_sci': float(np.mean(scis)),
            'std_sci': float(np.std(scis)),
            'mean_tokens': float(np.mean(tokens)),
            'std_tokens': float(np.std(tokens)),
            'coverage_efficiency': float(np.mean(coverages) / np.mean(tokens) * 1000),
        }
    
    # Save results
    output_dir = Path("v2/results/architecture_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(dict(all_results), f, indent=2, default=str)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    for name, stats in summary.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Coverage: {stats['mean_coverage']:.1%} ± {stats['std_coverage']:.1%}")
        logger.info(f"  SCI: {stats['mean_sci']:.3f} ± {stats['std_sci']:.3f}")
        logger.info(f"  Tokens: {stats['mean_tokens']:.0f} ± {stats['std_tokens']:.0f}")
        logger.info(f"  Coverage Efficiency: {stats['coverage_efficiency']:.4f}")
    
    return summary


# ============================================================================
# SCALING EXPERIMENT
# ============================================================================

async def run_scaling_experiment(n_seeds: int = 2) -> Dict:
    """Run scaling experiment to compute scaling exponents."""
    logger.info("\n" + "=" * 60)
    logger.info("SCALING EXPERIMENT")
    logger.info("=" * 60)
    
    n_values = [4, 8, 16, 32]  # Agent counts to test
    n_generations = 50
    
    architectures = [
        ('Independent', IndependentTraining),
        ('Tournament', TournamentSelection),
        ('Market', MarketBasedBidding),
        ('CSE', CompetitiveSpecialistEcosystem),
    ]
    
    scaling_results = defaultdict(list)
    client = GeminiClient()
    
    for n_agents in n_values:
        logger.info(f"\n--- N = {n_agents} agents ---")
        
        for name, arch_class in architectures:
            tokens_list = []
            coverage_list = []
            
            for seed in range(n_seeds):
                result = await run_architecture(
                    arch_class, n_agents, n_generations, seed, client
                )
                tokens_list.append(result['total_tokens'])
                coverage_list.append(result['coverage'])
            
            scaling_results[name].append({
                'n_agents': n_agents,
                'mean_tokens': np.mean(tokens_list),
                'mean_coverage': np.mean(coverage_list),
            })
            
            logger.info(f"  {name}: {np.mean(tokens_list):.0f} tokens, {np.mean(coverage_list):.1%} coverage")
    
    # Compute scaling exponents
    exponents = {}
    for name, results in scaling_results.items():
        ns = [r['n_agents'] for r in results]
        tokens = [r['mean_tokens'] for r in results]
        
        # Fit log-log linear regression
        log_n = np.log(ns)
        log_tokens = np.log(tokens)
        alpha, _ = np.polyfit(log_n, log_tokens, 1)
        
        exponents[name] = alpha
        logger.info(f"\n{name}: α = {alpha:.3f}")
    
    # Save
    output_dir = Path("v2/results/scaling")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "scaling_results.json", 'w') as f:
        json.dump({
            'results': dict(scaling_results),
            'exponents': exponents,
        }, f, indent=2, default=str)
    
    return exponents


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['compare', 'scaling', 'full'], default='compare')
    parser.add_argument('--n-agents', type=int, default=12)
    parser.add_argument('--n-generations', type=int, default=60)
    parser.add_argument('--n-seeds', type=int, default=3)
    args = parser.parse_args()
    
    if args.mode == 'compare':
        asyncio.run(run_comparison(args.n_agents, args.n_generations, args.n_seeds))
    elif args.mode == 'scaling':
        asyncio.run(run_scaling_experiment(args.n_seeds))
    elif args.mode == 'full':
        asyncio.run(run_comparison(args.n_agents, args.n_generations, args.n_seeds))
        asyncio.run(run_scaling_experiment(args.n_seeds))
