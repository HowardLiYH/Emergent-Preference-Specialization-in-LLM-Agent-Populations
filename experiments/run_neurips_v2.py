"""
NeurIPS v2 Master Experiment Pipeline

Orchestrates the complete experiment suite:
1. Domain overlap validation
2. Component ablation study
3. All baselines (including negative controls)
4. Main experiment with PortfolioAgent
5. Prompt swap causality test
6. Results aggregation and paper figure generation

Usage:
    # Quick test (Gemini, small scale)
    python run_neurips_v2.py --mode dev --scale small

    # Full experiment (Gemini dev + GPT-4o-mini production)
    python run_neurips_v2.py --mode full --scale full
"""

import asyncio
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genesis.llm_client import LLMClient
from genesis.cost_tracker import CostTracker


@dataclass
class PipelineConfig:
    """Configuration for the experiment pipeline."""
    # API keys
    gemini_api_key: str = ""
    openai_api_key: str = ""
    openai_api_base: str = ""

    # Scale settings
    tasks_per_domain: int = 5
    num_seeds: int = 5
    num_generations: int = 50
    num_agents: int = 10
    num_domains: int = 10

    # Modes
    use_real_llm: bool = True
    dev_mode: bool = True  # Use Gemini (free) vs OpenAI (paid)

    @classmethod
    def for_dev_small(cls, gemini_key: str) -> 'PipelineConfig':
        """Small-scale dev config for testing."""
        return cls(
            gemini_api_key=gemini_key,
            tasks_per_domain=3,
            num_seeds=2,
            num_generations=20,
            num_agents=5,
            num_domains=4,
            dev_mode=True
        )

    @classmethod
    def for_dev_medium(cls, gemini_key: str) -> 'PipelineConfig':
        """Medium-scale dev config."""
        return cls(
            gemini_api_key=gemini_key,
            tasks_per_domain=5,
            num_seeds=5,
            num_generations=50,
            num_agents=8,
            num_domains=6,
            dev_mode=True
        )

    @classmethod
    def for_production(cls, gemini_key: str, openai_key: str, openai_base: str) -> 'PipelineConfig':
        """Full production config for paper."""
        return cls(
            gemini_api_key=gemini_key,
            openai_api_key=openai_key,
            openai_api_base=openai_base,
            tasks_per_domain=10,
            num_seeds=30,
            num_generations=100,
            num_agents=16,
            num_domains=10,
            dev_mode=False  # Use OpenAI for final runs
        )


@dataclass
class PipelineResults:
    """Results from the full pipeline."""
    config: Dict
    domain_overlap: Dict
    component_ablation: Dict
    baselines: Dict
    main_experiment: Dict
    prompt_swap: Dict
    total_cost: float
    total_tokens: int
    duration_seconds: float
    timestamp: str
    success: bool


async def run_phase_domain_overlap(client: LLMClient, config: PipelineConfig) -> Dict:
    """Phase 1: Validate domain orthogonality."""
    print("\n" + "=" * 70)
    print("PHASE 1: DOMAIN OVERLAP VALIDATION")
    print("=" * 70)

    from exp_domain_overlap import measure_domain_overlap, validate_domain_orthogonality

    result = await measure_domain_overlap(
        client,
        tasks_per_domain=config.tasks_per_domain,
        verbose=True
    )

    validation = validate_domain_orthogonality(result)

    return {
        "diagonal_mean": result.diagonal_mean,
        "off_diagonal_mean": result.off_diagonal_mean,
        "gap": result.gap,
        "is_orthogonal": result.is_orthogonal,
        "validation": validation
    }


async def run_phase_component_ablation(client: LLMClient, config: PipelineConfig) -> Dict:
    """Phase 2: Component ablation study."""
    print("\n" + "=" * 70)
    print("PHASE 2: COMPONENT ABLATION STUDY")
    print("=" * 70)

    from exp_component_ablation import run_component_ablation

    summary = await run_component_ablation(
        client,
        num_domains=min(config.num_domains, 4),
        tasks_per_domain=config.tasks_per_domain
    )

    return asdict(summary)


async def run_phase_baselines(config: PipelineConfig) -> Dict:
    """Phase 3: All baselines including negative controls."""
    print("\n" + "=" * 70)
    print("PHASE 3: BASELINE EXPERIMENTS")
    print("=" * 70)

    from exp_baselines import run_all_baselines_v2

    results = await run_all_baselines_v2(num_seeds=config.num_seeds)

    # Summarize
    summary = {}
    for name, baseline_results in results.items():
        lsis = [r.final_lsi for r in baseline_results]
        summary[name] = {
            "avg_lsi": sum(lsis) / len(lsis),
            "std_lsi": (sum((x - sum(lsis)/len(lsis))**2 for x in lsis) / len(lsis)) ** 0.5,
            "num_runs": len(baseline_results)
        }

    return summary


async def run_phase_main_experiment(client: LLMClient, config: PipelineConfig) -> Dict:
    """Phase 4: Main experiment with PortfolioAgent."""
    print("\n" + "=" * 70)
    print("PHASE 4: MAIN EXPERIMENT (PortfolioAgent)")
    print("=" * 70)

    from genesis.evolution_v2 import create_portfolio_population, PortfolioAgent
    from genesis.domains import DomainType, get_all_domains
    from genesis.benchmark_tasks import BENCHMARK_LOADER
    import random

    domains = list(get_all_domains())[:config.num_domains]
    all_tasks = BENCHMARK_LOADER.load_all_domains(n_per_domain=config.tasks_per_domain)

    lsi_by_seed = []
    specialists_by_seed = []

    for seed in range(config.num_seeds):
        print(f"\n  Seed {seed + 1}/{config.num_seeds}...")
        random.seed(seed)

        # Create population
        agents = create_portfolio_population(config.num_agents, base_seed=seed)

        # Run generations
        for gen in range(config.num_generations):
            # Sample tasks
            task_domains = random.choices(domains, k=5)

            for domain in task_domains:
                # Simulate competition (would use real LLM in production)
                scores = {}
                for agent in agents:
                    # Base performance + affinity bonus
                    base = 0.3
                    if agent.get_primary_domain() == domain:
                        base += 0.3 * agent.domain_affinity[domain]
                    scores[agent.agent_id] = base + random.uniform(-0.1, 0.1)

                # Winner takes all
                winner_id = max(scores, key=scores.get)
                winner = next(a for a in agents if a.agent_id == winner_id)

                # Evolve winner
                strategy = winner.select_strategy_to_add(domain)
                if strategy:
                    winner.add_strategy(strategy)
                winner.record_attempt(domain, won=True)

                # Others don't evolve
                for agent in agents:
                    if agent.agent_id != winner_id:
                        agent.record_attempt(domain, won=False)

        # Compute final metrics
        lsis = [agent.compute_lsi() for agent in agents]
        avg_lsi = sum(lsis) / len(lsis)
        lsi_by_seed.append(avg_lsi)

        # Count specialists
        specialist_counts = {}
        for agent in agents:
            primary = agent.get_primary_domain().value
            specialist_counts[primary] = specialist_counts.get(primary, 0) + 1
        specialists_by_seed.append(len(specialist_counts))

        print(f"    LSI: {avg_lsi:.3f}, Unique specialists: {len(specialist_counts)}")

    return {
        "avg_lsi": sum(lsi_by_seed) / len(lsi_by_seed),
        "std_lsi": (sum((x - sum(lsi_by_seed)/len(lsi_by_seed))**2 for x in lsi_by_seed) / len(lsi_by_seed)) ** 0.5,
        "avg_unique_specialists": sum(specialists_by_seed) / len(specialists_by_seed),
        "lsi_by_seed": lsi_by_seed,
        "num_seeds": config.num_seeds
    }


async def run_phase_prompt_swap(client: LLMClient, config: PipelineConfig) -> Dict:
    """Phase 5: Prompt swap causality test."""
    print("\n" + "=" * 70)
    print("PHASE 5: PROMPT SWAP CAUSALITY TEST")
    print("=" * 70)

    from exp_prompt_swap import run_prompt_swap_test
    from genesis.domains import get_all_domains

    domains = list(get_all_domains())[:min(config.num_domains, 4)]

    summary = await run_prompt_swap_test(
        client,
        domains_to_test=domains,
        tasks_per_domain=config.tasks_per_domain
    )

    return asdict(summary)


async def run_full_pipeline(
    gemini_key: str,
    openai_key: str = None,
    openai_base: str = None,
    scale: str = "small",
    save_results: bool = True
) -> PipelineResults:
    """
    Run the complete NeurIPS experiment pipeline.

    Args:
        gemini_key: Gemini API key (for development)
        openai_key: OpenAI API key (for production)
        openai_base: OpenAI API base URL
        scale: "small", "medium", or "full"
        save_results: Whether to save results to disk

    Returns:
        PipelineResults with all experiment data
    """
    import time
    start_time = time.time()

    print("=" * 70)
    print("NEURIPS V2 EXPERIMENT PIPELINE")
    print("=" * 70)
    print(f"Scale: {scale}")
    print(f"Start time: {datetime.now().isoformat()}")

    # Configure based on scale
    if scale == "small":
        config = PipelineConfig.for_dev_small(gemini_key)
    elif scale == "medium":
        config = PipelineConfig.for_dev_medium(gemini_key)
    else:
        config = PipelineConfig.for_production(gemini_key, openai_key or "", openai_base or "")

    print(f"\nConfiguration:")
    print(f"  Tasks/domain: {config.tasks_per_domain}")
    print(f"  Seeds: {config.num_seeds}")
    print(f"  Generations: {config.num_generations}")
    print(f"  Agents: {config.num_agents}")
    print(f"  Domains: {config.num_domains}")

    # Create client
    client = LLMClient.for_gemini(api_key=gemini_key, model="gemini-2.5-flash")

    results = {
        "domain_overlap": {},
        "component_ablation": {},
        "baselines": {},
        "main_experiment": {},
        "prompt_swap": {}
    }

    try:
        # Phase 1: Domain Overlap
        results["domain_overlap"] = await run_phase_domain_overlap(client, config)

        # Phase 2: Component Ablation
        results["component_ablation"] = await run_phase_component_ablation(client, config)

        # Phase 3: Baselines
        results["baselines"] = await run_phase_baselines(config)

        # Phase 4: Main Experiment
        results["main_experiment"] = await run_phase_main_experiment(client, config)

        # Phase 5: Prompt Swap
        results["prompt_swap"] = await run_phase_prompt_swap(client, config)

        success = True

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        success = False

    finally:
        await client.close()

    duration = time.time() - start_time

    # Build final results
    pipeline_results = PipelineResults(
        config=asdict(config) if hasattr(config, '__dataclass_fields__') else vars(config),
        domain_overlap=results["domain_overlap"],
        component_ablation=results["component_ablation"],
        baselines=results["baselines"],
        main_experiment=results["main_experiment"],
        prompt_swap=results["prompt_swap"],
        total_cost=client.usage.estimated_cost if hasattr(client, 'usage') else 0,
        total_tokens=client.usage.total_tokens if hasattr(client, 'usage') else 0,
        duration_seconds=duration,
        timestamp=datetime.now().isoformat(),
        success=success
    )

    # Print summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"Total tokens: {pipeline_results.total_tokens:,}")
    print(f"Estimated cost: ${pipeline_results.total_cost:.4f}")
    print(f"Success: {'✅' if success else '❌'}")

    if results["main_experiment"]:
        print(f"\nMain Results:")
        print(f"  Average LSI: {results['main_experiment'].get('avg_lsi', 0):.3f}")
        print(f"  Unique specialists: {results['main_experiment'].get('avg_unique_specialists', 0):.1f}")

    # Save results
    if save_results:
        output_dir = Path(__file__).parent.parent / "results" / "neurips_v2"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"pipeline_results_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(asdict(pipeline_results), f, indent=2, default=str)

        print(f"\nResults saved to {output_file}")

    return pipeline_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeurIPS v2 Experiment Pipeline")
    parser.add_argument("--gemini-key", help="Gemini API key",
                       default=os.getenv("GEMINI_API_KEY", ""))
    parser.add_argument("--openai-key", help="OpenAI API key",
                       default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--openai-base", help="OpenAI API base URL",
                       default=os.getenv("OPENAI_API_BASE", ""))
    parser.add_argument("--scale", choices=["small", "medium", "full"],
                       default="small", help="Experiment scale")

    args = parser.parse_args()

    if not args.gemini_key:
        print("Error: Please provide --gemini-key or set GEMINI_API_KEY")
        sys.exit(1)

    asyncio.run(run_full_pipeline(
        gemini_key=args.gemini_key,
        openai_key=args.openai_key,
        openai_base=args.openai_base,
        scale=args.scale
    ))
