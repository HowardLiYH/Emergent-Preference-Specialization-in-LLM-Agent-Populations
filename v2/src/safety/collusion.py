"""
Collusion detection for multi-agent competition.

Detects suspicious patterns that might indicate agents
are not competing fairly.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import Counter
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class CollusionAlert:
    """Alert for potential collusion."""
    alert_type: str
    severity: str  # 'low', 'medium', 'high'
    regime: str
    agents_involved: List[str]
    evidence: str
    confidence: float


class CollusionDetector:
    """
    Detects suspicious win patterns that might indicate collusion.

    Patterns detected:
    - Perfect alternation (A, B, A, B, ...)
    - Round-robin (A, B, C, A, B, C, ...)
    - Statistical anomalies in win distribution
    """

    def __init__(
        self,
        min_window: int = 10,
        alternation_threshold: float = 0.8,
        uniformity_threshold: float = 0.1
    ):
        """
        Initialize collusion detector.

        Args:
            min_window: Minimum number of competitions to analyze
            alternation_threshold: Max alternation ratio before flagging
            uniformity_threshold: Chi-square p-value threshold
        """
        self.min_window = min_window
        self.alternation_threshold = alternation_threshold
        self.uniformity_threshold = uniformity_threshold

        # Win history per regime
        self.win_history: Dict[str, List[str]] = {}

        # Alerts
        self.alerts: List[CollusionAlert] = []

    def record_win(self, regime: str, winner_id: str) -> Optional[CollusionAlert]:
        """
        Record a competition win and check for collusion.

        Args:
            regime: Regime of competition
            winner_id: ID of winning agent

        Returns:
            CollusionAlert if suspicious pattern detected
        """
        if regime not in self.win_history:
            self.win_history[regime] = []

        self.win_history[regime].append(winner_id)

        # Only analyze if we have enough data
        if len(self.win_history[regime]) >= self.min_window:
            alert = self.detect(regime)
            if alert:
                self.alerts.append(alert)
                return alert

        return None

    def detect(self, regime: str, window: Optional[int] = None) -> Optional[CollusionAlert]:
        """
        Detect collusion in a regime.

        Args:
            regime: Regime to analyze
            window: Number of recent wins to analyze

        Returns:
            CollusionAlert if suspicious, None otherwise
        """
        if regime not in self.win_history:
            return None

        winners = self.win_history[regime]
        if window:
            winners = winners[-window:]

        if len(winners) < self.min_window:
            return None

        # Check for alternation pattern
        alert = self._check_alternation(regime, winners)
        if alert:
            return alert

        # Check for round-robin pattern
        alert = self._check_round_robin(regime, winners)
        if alert:
            return alert

        # Check for statistical anomalies
        alert = self._check_distribution(regime, winners)
        if alert:
            return alert

        return None

    def _check_alternation(
        self,
        regime: str,
        winners: List[str]
    ) -> Optional[CollusionAlert]:
        """Check for perfect alternation between two agents."""
        unique_winners = set(winners)

        if len(unique_winners) != 2:
            return None

        # Count transitions
        transitions = 0
        for i in range(1, len(winners)):
            if winners[i] != winners[i-1]:
                transitions += 1

        alternation_ratio = transitions / (len(winners) - 1)

        if alternation_ratio > self.alternation_threshold:
            agents = list(unique_winners)
            return CollusionAlert(
                alert_type='alternation',
                severity='high',
                regime=regime,
                agents_involved=agents,
                evidence=f"Alternation ratio: {alternation_ratio:.2%} "
                        f"(threshold: {self.alternation_threshold:.2%})",
                confidence=alternation_ratio
            )

        return None

    def _check_round_robin(
        self,
        regime: str,
        winners: List[str]
    ) -> Optional[CollusionAlert]:
        """Check for round-robin pattern."""
        unique_winners = list(set(winners))
        n = len(unique_winners)

        if n < 3:
            return None

        # Check if winners follow a repeating sequence
        # Try to find the period
        for period in range(n, n + 2):  # Period should be n or n+1
            if len(winners) < period * 2:
                continue

            matches = 0
            for i in range(len(winners) - period):
                if winners[i] == winners[i + period]:
                    matches += 1

            match_ratio = matches / (len(winners) - period)

            if match_ratio > 0.8:
                return CollusionAlert(
                    alert_type='round_robin',
                    severity='medium',
                    regime=regime,
                    agents_involved=unique_winners,
                    evidence=f"Periodic pattern detected with period {period}, "
                            f"match ratio: {match_ratio:.2%}",
                    confidence=match_ratio
                )

        return None

    def _check_distribution(
        self,
        regime: str,
        winners: List[str]
    ) -> Optional[CollusionAlert]:
        """Check for statistical anomalies using chi-square test."""
        counts = Counter(winners)
        unique_winners = list(counts.keys())
        n = len(unique_winners)

        if n < 2:
            return None

        # Expected wins if fair competition
        expected = len(winners) / n

        # Chi-square statistic
        chi_square = sum(
            (counts[w] - expected) ** 2 / expected
            for w in unique_winners
        )

        # Degrees of freedom
        df = n - 1

        # Approximate p-value (simplified)
        # For proper implementation, use scipy.stats.chi2
        p_value = self._chi2_survival(chi_square, df)

        # If distribution is TOO uniform (suspicious for natural competition)
        if p_value > 0.99:  # Extremely uniform
            return CollusionAlert(
                alert_type='too_uniform',
                severity='low',
                regime=regime,
                agents_involved=unique_winners,
                evidence=f"Win distribution suspiciously uniform. "
                        f"Chi-square: {chi_square:.2f}, p-value: {p_value:.4f}",
                confidence=p_value
            )

        return None

    def _chi2_survival(self, x: float, df: int) -> float:
        """Approximate chi-square survival function."""
        # Simplified approximation
        # For production, use scipy.stats.chi2.sf(x, df)
        try:
            import scipy.stats
            return scipy.stats.chi2.sf(x, df)
        except ImportError:
            # Rough approximation
            z = (x / df - 1) * math.sqrt(2 * df)
            return 0.5 * (1 - math.erf(z / math.sqrt(2)))

    def get_alerts(
        self,
        regime: Optional[str] = None,
        severity: Optional[str] = None
    ) -> List[CollusionAlert]:
        """
        Get collusion alerts.

        Args:
            regime: Filter by regime
            severity: Filter by severity

        Returns:
            List of matching alerts
        """
        alerts = self.alerts

        if regime:
            alerts = [a for a in alerts if a.regime == regime]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    def clear_history(self, regime: Optional[str] = None) -> None:
        """Clear win history."""
        if regime:
            self.win_history.pop(regime, None)
        else:
            self.win_history.clear()
