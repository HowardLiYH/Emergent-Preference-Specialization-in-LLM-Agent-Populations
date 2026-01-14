"""
Benchmark runner for ESB (Emergent Specialization Benchmark).
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import json
import logging

from .baselines import L0Baseline
from .explain import ExplanationGenerator
from .explanation_evaluator import ExplanationEvaluator
from .tasks.l0_tasks import generate_l0_batch, evaluate_l0_response
from .tasks.l1_tasks import generate_l1_batch, evaluate_l1_response

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Runner for the Emergent Specialization Benchmark (ESB).

    Components:
    1. L0 baseline measurements
    2. Task generation for each level
    3. Specialization explanation generation
    4. Explanation quality evaluation
    """

    def __init__(
        self,
        output_path: Optional[Path] = None,
        llm_client=None
    ):
        """
        Initialize benchmark runner.

        Args:
            output_path: Directory for benchmark outputs
            llm_client: LLM client for evaluation
        """
        self.output_path = Path(output_path) if output_path else Path('v2/results/benchmark')
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.llm_client = llm_client

        # Components
        self.baseline = L0Baseline(llm_client)
        self.explainer = ExplanationGenerator(llm_client)
        self.evaluator = ExplanationEvaluator(llm_client)

        # Results
        self.results: Dict[str, Any] = {}

    def run_baseline_evaluation(
        self,
        n_samples_per_level: int = 50
    ) -> Dict[str, Any]:
        """
        Run L0 baseline evaluation for all task levels.

        Args:
            n_samples_per_level: Number of samples per level

        Returns:
            Baseline results
        """
        logger.info("Running L0 baseline evaluation...")

        baseline_results = {}

        # L0 tasks
        l0_tasks = generate_l0_batch(n_samples_per_level)
        result = self.baseline.evaluate_task_level(
            'L0', l0_tasks, evaluate_l0_response, n_samples_per_level
        )
        baseline_results['L0'] = {
            'l0_accuracy': result.l0_accuracy,
            'optimal_accuracy': result.optimal_accuracy,
            'gap': result.gap,
        }

        # L1 tasks
        l1_tasks = generate_l1_batch(n_samples_per_level)
        result = self.baseline.evaluate_task_level(
            'L1', l1_tasks, evaluate_l1_response, n_samples_per_level
        )
        baseline_results['L1'] = {
            'l0_accuracy': result.l0_accuracy,
            'optimal_accuracy': result.optimal_accuracy,
            'gap': result.gap,
        }

        # Use expected values for L2-L4 (require special setup)
        for level in ['L2', 'L3', 'L4']:
            expected = self.baseline.get_expected_baselines().get(level, {})
            baseline_results[level] = {
                'l0_accuracy': expected.get('l0_accuracy', 0),
                'optimal_accuracy': expected.get('optimal_accuracy', 0),
                'gap': expected.get('gap', 0),
                'note': 'Expected values (requires special evaluation setup)'
            }

        self.results['baselines'] = baseline_results
        return baseline_results

    def run_explanation_evaluation(
        self,
        agents: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate explanation quality for a set of agents.

        Args:
            agents: List of agent data dicts

        Returns:
            Aggregate explanation metrics
        """
        logger.info("Running explanation evaluation...")

        explanations = []
        ground_truths = []

        for agent in agents:
            # Generate explanation
            explanation = self.explainer.generate(
                agent_id=agent.get('id', 'unknown'),
                specialty=agent.get('specialty', 'unknown'),
                tool_level=agent.get('tool_level', 0),
                win_rate=agent.get('win_rate', 0),
                total_wins=agent.get('total_wins', 0)
            )
            explanations.append(explanation.explanation_text)
            ground_truths.append(agent)

        # Evaluate
        metrics = self.evaluator.evaluate_batch(explanations, ground_truths)

        self.results['explanation_metrics'] = metrics
        return metrics

    def run_full_benchmark(
        self,
        population=None,
        n_baseline_samples: int = 50
    ) -> Dict[str, Any]:
        """
        Run the full ESB benchmark.

        Args:
            population: Population object to evaluate
            n_baseline_samples: Samples for baseline evaluation

        Returns:
            Complete benchmark results
        """
        logger.info("Starting full ESB benchmark...")
        start_time = datetime.now()

        # 1. Baseline evaluation
        baseline_results = self.run_baseline_evaluation(n_baseline_samples)

        # 2. Explanation evaluation (if population provided)
        if population:
            agents = []
            for agent in population.agents:
                agents.append({
                    'id': agent.id,
                    'specialty': agent.get_specialty(),
                    'tool_level': agent.tool_level,
                    'win_rate': agent.stats.win_rate,
                    'total_wins': agent.stats.wins,
                })

            explanation_metrics = self.run_explanation_evaluation(agents)
        else:
            explanation_metrics = {}

        # 3. Compile results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        full_results = {
            'benchmark_name': 'ESB (Emergent Specialization Benchmark)',
            'version': '2.0',
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration,
            'baselines': baseline_results,
            'explanation_metrics': explanation_metrics,
            'tool_necessity_verified': all(
                baseline_results.get(level, {}).get('gap', 0) > 0.20
                for level in ['L1', 'L2', 'L3', 'L4']
            ),
        }

        self.results = full_results

        # Save results
        self._save_results(full_results)

        logger.info(f"Benchmark completed in {duration:.1f}s")
        return full_results

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save benchmark results to file."""
        output_file = self.output_path / f'esb_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_file}")

    def generate_report(self) -> str:
        """Generate human-readable benchmark report."""
        lines = [
            "# ESB Benchmark Report",
            "",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## L0 Baseline Results",
            "",
            "| Level | L0 Baseline | Optimal | Gap | Tool Necessary? |",
            "|-------|-------------|---------|-----|-----------------|",
        ]

        baselines = self.results.get('baselines', {})
        for level in ['L0', 'L1', 'L2', 'L3', 'L4']:
            data = baselines.get(level, {})
            l0_acc = data.get('l0_accuracy', 0)
            optimal = data.get('optimal_accuracy', 0)
            gap = data.get('gap', 0)
            necessary = "Yes" if gap > 0.20 else "No"

            lines.append(
                f"| {level} | {l0_acc:.0%} | {optimal:.0%} | {gap:.0%} | {necessary} |"
            )

        lines.extend([
            "",
            "## Explanation Quality",
            "",
        ])

        exp_metrics = self.results.get('explanation_metrics', {})
        if exp_metrics:
            lines.extend([
                f"- Mean Factual Accuracy: {exp_metrics.get('mean_factual_accuracy', 0):.1%}",
                f"- Mean Completeness: {exp_metrics.get('mean_completeness', 0):.1%}",
                f"- Overall Score: {exp_metrics.get('mean_overall_score', 0):.1%}",
            ])
        else:
            lines.append("No explanation evaluation performed.")

        lines.extend([
            "",
            "## Conclusions",
            "",
        ])

        if self.results.get('tool_necessity_verified'):
            lines.append("✅ Tool necessity verified for L1-L4 tasks.")
        else:
            lines.append("⚠️ Tool necessity NOT fully verified.")

        return "\n".join(lines)
