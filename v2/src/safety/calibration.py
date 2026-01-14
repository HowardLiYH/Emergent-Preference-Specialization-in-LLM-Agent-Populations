"""
Confidence calibration checking.

Verifies that agent's stated confidence aligns with actual accuracy.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class CalibrationAlert:
    """Alert for calibration issues."""
    agent_id: str
    stated_confidence: float
    actual_accuracy: float
    calibration_error: float
    direction: str  # 'overconfident' or 'underconfident'
    n_samples: int


@dataclass
class CalibrationRecord:
    """Record of a single confidence/correctness pair."""
    confidence: float
    correct: bool


class CalibrationChecker:
    """
    Checks if agents are well-calibrated.

    A well-calibrated agent's stated confidence should match
    its actual accuracy. For example, when an agent says
    "I'm 80% confident", it should be correct 80% of the time.
    """

    def __init__(
        self,
        threshold: float = 0.15,
        min_samples: int = 20,
        n_bins: int = 10
    ):
        """
        Initialize calibration checker.

        Args:
            threshold: Max acceptable calibration error
            min_samples: Minimum samples before checking
            n_bins: Number of bins for calibration curve
        """
        self.threshold = threshold
        self.min_samples = min_samples
        self.n_bins = n_bins

        # Records per agent
        self.records: Dict[str, List[CalibrationRecord]] = {}

        # Alerts
        self.alerts: List[CalibrationAlert] = []

    def record(
        self,
        agent_id: str,
        confidence: float,
        correct: bool
    ) -> Optional[CalibrationAlert]:
        """
        Record a confidence-correctness pair.

        Args:
            agent_id: Agent identifier
            confidence: Stated confidence (0-1)
            correct: Whether the answer was correct

        Returns:
            CalibrationAlert if agent is miscalibrated
        """
        if agent_id not in self.records:
            self.records[agent_id] = []

        # Clamp confidence to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        self.records[agent_id].append(CalibrationRecord(
            confidence=confidence,
            correct=correct
        ))

        # Check calibration if enough samples
        if len(self.records[agent_id]) >= self.min_samples:
            alert = self.check_agent(agent_id)
            if alert:
                self.alerts.append(alert)
                return alert

        return None

    def check_agent(self, agent_id: str) -> Optional[CalibrationAlert]:
        """
        Check calibration for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            CalibrationAlert if miscalibrated
        """
        if agent_id not in self.records:
            return None

        records = self.records[agent_id]

        if len(records) < self.min_samples:
            return None

        # Calculate average confidence and accuracy
        avg_confidence = sum(r.confidence for r in records) / len(records)
        avg_accuracy = sum(1 for r in records if r.correct) / len(records)

        calibration_error = abs(avg_confidence - avg_accuracy)

        if calibration_error > self.threshold:
            direction = 'overconfident' if avg_confidence > avg_accuracy else 'underconfident'

            return CalibrationAlert(
                agent_id=agent_id,
                stated_confidence=avg_confidence,
                actual_accuracy=avg_accuracy,
                calibration_error=calibration_error,
                direction=direction,
                n_samples=len(records)
            )

        return None

    def get_calibration_curve(
        self,
        agent_id: str
    ) -> Optional[List[Tuple[float, float, int]]]:
        """
        Get calibration curve data for an agent.

        Returns list of (bin_center, accuracy, count) tuples.

        Args:
            agent_id: Agent identifier

        Returns:
            Calibration curve data or None if insufficient data
        """
        if agent_id not in self.records:
            return None

        records = self.records[agent_id]

        if len(records) < self.min_samples:
            return None

        # Bin records by confidence
        bin_width = 1.0 / self.n_bins
        bins: Dict[int, List[bool]] = {i: [] for i in range(self.n_bins)}

        for record in records:
            bin_idx = min(int(record.confidence / bin_width), self.n_bins - 1)
            bins[bin_idx].append(record.correct)

        # Calculate accuracy per bin
        curve = []
        for i in range(self.n_bins):
            bin_center = (i + 0.5) * bin_width
            bin_records = bins[i]

            if bin_records:
                accuracy = sum(bin_records) / len(bin_records)
                curve.append((bin_center, accuracy, len(bin_records)))

        return curve

    def expected_calibration_error(self, agent_id: str) -> Optional[float]:
        """
        Calculate Expected Calibration Error (ECE).

        ECE is the weighted average of calibration errors across bins.

        Args:
            agent_id: Agent identifier

        Returns:
            ECE value or None if insufficient data
        """
        curve = self.get_calibration_curve(agent_id)

        if not curve:
            return None

        total_samples = sum(count for _, _, count in curve)

        if total_samples == 0:
            return None

        ece = sum(
            count * abs(confidence - accuracy)
            for confidence, accuracy, count in curve
        ) / total_samples

        return ece

    def get_summary(self, agent_id: str) -> Optional[dict]:
        """
        Get calibration summary for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Summary dictionary or None if insufficient data
        """
        if agent_id not in self.records:
            return None

        records = self.records[agent_id]

        if len(records) < self.min_samples:
            return None

        avg_confidence = sum(r.confidence for r in records) / len(records)
        avg_accuracy = sum(1 for r in records if r.correct) / len(records)
        ece = self.expected_calibration_error(agent_id)

        return {
            'agent_id': agent_id,
            'n_samples': len(records),
            'avg_confidence': avg_confidence,
            'avg_accuracy': avg_accuracy,
            'calibration_error': abs(avg_confidence - avg_accuracy),
            'ece': ece,
            'is_calibrated': abs(avg_confidence - avg_accuracy) <= self.threshold,
        }

    def get_all_alerts(self) -> List[CalibrationAlert]:
        """Get all calibration alerts."""
        return self.alerts

    def clear(self, agent_id: Optional[str] = None) -> None:
        """Clear calibration records."""
        if agent_id:
            self.records.pop(agent_id, None)
        else:
            self.records.clear()
