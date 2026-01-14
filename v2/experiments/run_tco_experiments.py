"""
TCO Efficiency Experiments - Main Runner

This script runs all experiments for the TCO Reframe Plan:
1. Subset Competition (K=3) vs Independent
2. Capability Unlocking Analysis
3. Parallel Training Benchmark
4. Latency Benchmarks
5. 50-Seed Validation (optional, long-running)

Usage:
    python run_tco_experiments.py --seeds 10
    python run_tco_experiments.py --seeds 50 --full
"""

import os
import sys
import json
import random
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Import our modules
from experiments.architectures.subset_cse import SubsetCSE, IndependentTraining
from experiments.architectures.dynamic_subset import DynamicSubsetCSE
from experiments.parallel_training import ParallelTrainer
from experiments.capability_analysis import (
    CAPABILITY_TASKS, analyze_capability_unlocking, 
    summarize_capability_results, save_capability_results
)
from experiments.cost_analysis.amortized import (
    full_statistical_analysis, find_breakeven_with_ci,
    compute_effect_size, save_analysis
)

# LLM Client
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google.generativeai not available, using mock LLM")


class LLMClient:
    """LLM client wrapper for experiments."""
    
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model_name = model
        self.total_tokens = 0
        self.total_calls = 0
        
        if GEMINI_AVAILABLE:
            api_key = os.getenv('GOOGLE_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model)
                self.is_real = True
            else:
                self.is_real = False
                print("Warning: GOOGLE_API_KEY not set, using mock responses")
        else:
            self.is_real = False
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response from LLM."""
        self.total_calls += 1
        
        if self.is_real:
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={'max_output_tokens': max_tokens}
                )
                text = response.text
                # Estimate tokens
                self.total_tokens += len(prompt.split()) + len(text.split())
                return text
            except Exception as e:
                print(f"LLM Error: {e}")
                return f"Error: {str(e)}"
        else:
            # Mock response
            self.total_tokens += 100
            return f"Mock response for: {prompt[:50]}..."
    
    def get_stats(self) -> Dict:
        return {
            'total_calls': self.total_calls,
            'total_tokens': self.total_tokens,
            'is_real': self.is_real,
        }


# Regimes for experiments
REGIMES = ['pure_qa', 'code_math', 'chart_analysis', 'document_qa', 'realtime_data']


def generate_task(regime: str, rng: random.Random) -> Tuple[str, str]:
    """Generate a task for a regime."""
    tasks = {
        'pure_qa': [
            ("What is the capital of Japan?", "Tokyo"),
            ("Who wrote 'Hamlet'?", "Shakespeare"),
            ("What is the boiling point of water in Celsius?", "100"),
        ],
        'code_math': [
            ("Calculate 15 * 7 + 23", "128"),
            ("What is the square root of 256?", "16"),
            ("Solve: 2x + 5 = 15", "5"),
        ],
        'chart_analysis': [
            ("Describe the trend if values go from 10 to 50 over 5 months", "increasing"),
            ("What type of chart shows proportions?", "pie"),
        ],
        'document_qa': [
            ("What is the main purpose of an abstract in a paper?", "summary"),
            ("Where are references typically found?", "end"),
        ],
        'realtime_data': [
            ("What format is commonly used for API responses?", "JSON"),
            ("What HTTP method retrieves data?", "GET"),
        ],
    }
    
    regime_tasks = tasks.get(regime, [("Generic question", "answer")])
    return rng.choice(regime_tasks)


def evaluate_agent(
    agent, 
    tool: str, 
    question: str, 
    expected: str,
    llm_client: LLMClient
) -> Tuple[bool, int, str]:
    """Evaluate an agent's response."""
    # Construct prompt based on tool
    prompts = {
        'L0': f"Answer concisely: {question}",
        'L1': f"Use calculation if needed: {question}",
        'L2': f"Analyze visually: {question}",
        'L3': f"Search and answer: {question}",
        'L4': f"Use real-time data: {question}",
    }
    prompt = prompts.get(tool, f"Answer: {question}")
    
    response = llm_client.generate(prompt, max_tokens=200)
    
    # Check success
    success = expected.lower() in response.lower()
    tokens = len(prompt.split()) + len(response.split())
    
    return success, tokens, response


def run_subset_comparison(
    n_agents: int = 12,
    n_generations: int = 60,
    n_seeds: int = 10,
    llm_client: LLMClient = None
) -> Dict:
    """
    Compare Subset CSE (K=3) vs Independent Training.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Subset Competition vs Independent")
    print("=" * 60)
    
    if llm_client is None:
        llm_client = LLMClient()
    
    results = {
        'subset_cse': [],
        'dynamic_cse': [],
        'independent': [],
    }
    
    for seed in range(n_seeds):
        print(f"\nSeed {seed + 1}/{n_seeds}")
        rng = random.Random(seed)
        
        # Initialize systems
        subset_cse = SubsetCSE(n_agents, REGIMES, K=3, epsilon=0.1)
        dynamic_cse = DynamicSubsetCSE(n_agents, REGIMES, K_min=2, K_max=6)
        independent = IndependentTraining(n_agents, REGIMES)
        
        # Run training
        for gen in range(n_generations):
            regime = rng.choice(REGIMES)
            task = generate_task(regime, rng)
            
            # Evaluate function
            def eval_fn(agent, tool, q, a):
                return evaluate_agent(agent, tool, q, a, llm_client)
            
            subset_cse.train_step(task, regime, rng, eval_fn)
            dynamic_cse.train_step(task, regime, rng, eval_fn)
            independent.train_step(task, regime, rng, eval_fn)
            
            if (gen + 1) % 20 == 0:
                print(f"  Generation {gen + 1}/{n_generations}")
        
        # Collect metrics
        results['subset_cse'].append({
            'seed': seed,
            'coverage': subset_cse.get_coverage(),
            'tokens': subset_cse.total_tokens,
            'llm_calls': subset_cse.total_llm_calls,
            'failure_rate': subset_cse.get_failure_rate(),
        })
        
        results['dynamic_cse'].append({
            'seed': seed,
            'coverage': dynamic_cse.get_coverage(),
            'tokens': dynamic_cse.total_tokens,
            'llm_calls': dynamic_cse.total_llm_calls,
            'failure_rate': dynamic_cse.get_failure_rate(),
            'k_stats': dynamic_cse.get_k_statistics(),
        })
        
        results['independent'].append({
            'seed': seed,
            'coverage': independent.get_coverage(),
            'tokens': independent.total_tokens,
            'llm_calls': independent.total_llm_calls,
            'failure_rate': independent.get_failure_rate(),
        })
    
    # Compute summary statistics
    summary = {}
    for method, runs in results.items():
        summary[method] = {
            'mean_coverage': sum(r['coverage'] for r in runs) / len(runs),
            'mean_tokens': sum(r['tokens'] for r in runs) / len(runs),
            'mean_failure_rate': sum(r['failure_rate'] for r in runs) / len(runs),
            'n_seeds': len(runs),
        }
    
    print("\n--- Results Summary ---")
    for method, stats in summary.items():
        print(f"{method}:")
        print(f"  Coverage: {stats['mean_coverage']:.2%}")
        print(f"  Tokens: {stats['mean_tokens']:.0f}")
        print(f"  Failure Rate: {stats['mean_failure_rate']:.2%}")
    
    return {'raw': results, 'summary': summary}


def run_amortized_analysis(subset_results: Dict) -> Dict:
    """
    Run amortized cost analysis on experiment results.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Amortized Cost Analysis")
    print("=" * 60)
    
    # Extract failure rates
    cse_failures = [r['failure_rate'] for r in subset_results['raw']['subset_cse']]
    ind_failures = [r['failure_rate'] for r in subset_results['raw']['independent']]
    
    cse_coverages = [r['coverage'] for r in subset_results['raw']['subset_cse']]
    ind_coverages = [r['coverage'] for r in subset_results['raw']['independent']]
    
    # Training costs (from token counts)
    cse_training = sum(r['tokens'] for r in subset_results['raw']['subset_cse']) / len(subset_results['raw']['subset_cse'])
    ind_training = sum(r['tokens'] for r in subset_results['raw']['independent']) / len(subset_results['raw']['independent'])
    
    # Convert tokens to $ (approx $0.001 per 1000 tokens)
    cse_training_cost = cse_training * 0.001 / 1000
    ind_training_cost = ind_training * 0.001 / 1000
    
    # Run full analysis
    analysis = full_statistical_analysis(
        cse_training=cse_training_cost,
        ind_training=ind_training_cost,
        cse_failure_rates=cse_failures,
        ind_failure_rates=ind_failures,
        cse_coverages=cse_coverages,
        ind_coverages=ind_coverages,
    )
    
    print("\n--- Break-Even Analysis ---")
    print(analysis['breakeven']['interpretation'])
    
    print("\n--- Effect Sizes ---")
    print(f"Failure Rate: Cohen's d = {analysis['effect_sizes']['failure_rate']['cohens_d']}")
    print(f"Coverage: Cohen's d = {analysis['effect_sizes']['coverage']['cohens_d']}")
    
    return analysis


def run_parallel_benchmark() -> Dict:
    """
    Benchmark parallel training speedup.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Parallel Training Benchmark")
    print("=" * 60)
    
    from experiments.parallel_training import run_parallel_benchmark as bench
    
    results = bench(n_agents=12, n_iterations=50, worker_configs=[1, 2, 4, 8])
    
    print("\n--- Speedup Results ---")
    for config, stats in results.items():
        print(f"{config}: {stats['speedup_factor']}x speedup")
    
    return results


def save_all_results(results: Dict, output_dir: str = "results/tco_experiments"):
    """Save all experiment results."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save each result set
    for name, data in results.items():
        path = f"{output_dir}/{name}_{timestamp}.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Saved: {path}")
    
    # Save summary
    summary_path = f"{output_dir}/summary_{timestamp}.json"
    summary = {
        'timestamp': timestamp,
        'experiments': list(results.keys()),
    }
    
    # Extract key metrics
    if 'subset_comparison' in results:
        summary['subset_comparison'] = results['subset_comparison'].get('summary', {})
    if 'amortized' in results:
        summary['breakeven'] = results['amortized'].get('breakeven', {})
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nSummary saved: {summary_path}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Run TCO Efficiency Experiments")
    parser.add_argument('--seeds', type=int, default=10, help='Number of seeds')
    parser.add_argument('--generations', type=int, default=60, help='Training generations')
    parser.add_argument('--agents', type=int, default=12, help='Number of agents')
    parser.add_argument('--full', action='store_true', help='Run full 50-seed validation')
    parser.add_argument('--skip-parallel', action='store_true', help='Skip parallel benchmark')
    args = parser.parse_args()
    
    if args.full:
        args.seeds = 50
    
    print("=" * 60)
    print("TCO EFFICIENCY EXPERIMENTS")
    print(f"Seeds: {args.seeds}, Generations: {args.generations}, Agents: {args.agents}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Initialize LLM
    llm_client = LLMClient()
    
    all_results = {}
    
    # Experiment 1: Subset Comparison
    subset_results = run_subset_comparison(
        n_agents=args.agents,
        n_generations=args.generations,
        n_seeds=args.seeds,
        llm_client=llm_client
    )
    all_results['subset_comparison'] = subset_results
    
    # Experiment 2: Amortized Analysis
    amortized_results = run_amortized_analysis(subset_results)
    all_results['amortized'] = amortized_results
    
    # Experiment 3: Parallel Benchmark
    if not args.skip_parallel:
        parallel_results = run_parallel_benchmark()
        all_results['parallel'] = parallel_results
    
    # LLM Stats
    all_results['llm_stats'] = llm_client.get_stats()
    
    # Save results
    output_dir = save_all_results(all_results)
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENTS COMPLETE")
    print(f"Time: {elapsed:.1f}s")
    print(f"Results: {output_dir}")
    print(f"{'=' * 60}")
    
    return all_results


if __name__ == "__main__":
    main()
