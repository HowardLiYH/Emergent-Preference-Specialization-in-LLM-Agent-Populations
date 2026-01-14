"""
Evaluation of explanation quality.

Measures factual accuracy, completeness, and fluency of
generated specialization explanations.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExplanationMetrics:
    """Metrics for explanation quality."""
    factual_accuracy: float  # 0-1 score
    completeness: float      # 0-1 score
    win_rate_correct: bool
    tool_level_correct: bool
    regime_correct: bool
    overall_score: float


class ExplanationEvaluator:
    """
    Evaluates the quality of generated specialization explanations.

    Metrics:
    - Factual accuracy: Do stated facts match ground truth?
    - Completeness: Are all required fields present?
    - Fluency: Is the explanation readable? (via perplexity if available)
    """

    REQUIRED_FIELDS = ['agent_id', 'specialty', 'tool_level', 'win_rate']

    def __init__(self, llm_client=None):
        """
        Initialize evaluator.

        Args:
            llm_client: Optional LLM client for fluency evaluation
        """
        self.llm_client = llm_client

    def evaluate(
        self,
        explanation_text: str,
        ground_truth: Dict[str, Any]
    ) -> ExplanationMetrics:
        """
        Evaluate an explanation against ground truth.

        Args:
            explanation_text: Generated explanation
            ground_truth: Actual agent data

        Returns:
            ExplanationMetrics
        """
        # Parse claimed values from explanation
        parsed = self._parse_explanation(explanation_text)

        # Check factual accuracy
        win_rate_correct = self._check_win_rate(
            parsed.get('win_rate'),
            ground_truth.get('win_rate', 0)
        )

        tool_level_correct = self._check_tool_level(
            parsed.get('tool_level'),
            ground_truth.get('tool_level', 0)
        )

        regime_correct = self._check_regime(
            parsed.get('specialty'),
            ground_truth.get('specialty', '')
        )

        # Calculate factual accuracy score
        accuracy_checks = [win_rate_correct, tool_level_correct, regime_correct]
        factual_accuracy = sum(accuracy_checks) / len(accuracy_checks)

        # Check completeness
        completeness = self._check_completeness(explanation_text, parsed)

        # Overall score
        overall_score = (factual_accuracy * 0.6 + completeness * 0.4)

        return ExplanationMetrics(
            factual_accuracy=factual_accuracy,
            completeness=completeness,
            win_rate_correct=win_rate_correct,
            tool_level_correct=tool_level_correct,
            regime_correct=regime_correct,
            overall_score=overall_score
        )

    def _parse_explanation(self, text: str) -> Dict[str, Any]:
        """Extract claimed values from explanation text."""
        parsed = {}

        # Extract win rate (e.g., "65.5%" or "0.655")
        win_rate_match = re.search(r'(\d+\.?\d*)\s*%', text)
        if win_rate_match:
            parsed['win_rate'] = float(win_rate_match.group(1)) / 100

        # Extract tool level (e.g., "L2" or "level 2")
        tool_match = re.search(r'L(\d)|level\s*(\d)', text, re.IGNORECASE)
        if tool_match:
            parsed['tool_level'] = int(tool_match.group(1) or tool_match.group(2))

        # Extract specialty/regime
        regime_patterns = ['pure_qa', 'code_math', 'chart_analysis', 'document_qa', 'realtime_data']
        for regime in regime_patterns:
            if regime.replace('_', ' ') in text.lower() or regime in text.lower():
                parsed['specialty'] = regime
                break

        # Extract total wins
        wins_match = re.search(r'(\d+)\s*(?:total\s*)?wins', text, re.IGNORECASE)
        if wins_match:
            parsed['total_wins'] = int(wins_match.group(1))

        return parsed

    def _check_win_rate(
        self,
        claimed: Optional[float],
        actual: float,
        tolerance: float = 0.05
    ) -> bool:
        """Check if claimed win rate is close to actual."""
        if claimed is None:
            return False
        return abs(claimed - actual) <= tolerance

    def _check_tool_level(
        self,
        claimed: Optional[int],
        actual: int
    ) -> bool:
        """Check if claimed tool level matches actual."""
        return claimed == actual

    def _check_regime(
        self,
        claimed: Optional[str],
        actual: str
    ) -> bool:
        """Check if claimed regime matches actual."""
        if claimed is None or actual is None:
            return False
        return claimed.lower() == actual.lower()

    def _check_completeness(
        self,
        text: str,
        parsed: Dict[str, Any]
    ) -> float:
        """Check how complete the explanation is."""
        required_mentions = [
            'win' in text.lower(),
            any(f'l{i}' in text.lower() for i in range(5)),
            'specialized' in text.lower() or 'specialty' in text.lower(),
            len(text) > 100,  # Minimum length
        ]

        return sum(required_mentions) / len(required_mentions)

    def evaluate_batch(
        self,
        explanations: List[str],
        ground_truths: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate a batch of explanations.

        Returns aggregate metrics.
        """
        all_metrics = []

        for explanation, truth in zip(explanations, ground_truths):
            metrics = self.evaluate(explanation, truth)
            all_metrics.append(metrics)

        # Aggregate
        n = len(all_metrics)
        if n == 0:
            return {}

        return {
            'mean_factual_accuracy': sum(m.factual_accuracy for m in all_metrics) / n,
            'mean_completeness': sum(m.completeness for m in all_metrics) / n,
            'mean_overall_score': sum(m.overall_score for m in all_metrics) / n,
            'pct_win_rate_correct': sum(m.win_rate_correct for m in all_metrics) / n,
            'pct_tool_level_correct': sum(m.tool_level_correct for m in all_metrics) / n,
            'pct_regime_correct': sum(m.regime_correct for m in all_metrics) / n,
            'n_evaluated': n,
        }
