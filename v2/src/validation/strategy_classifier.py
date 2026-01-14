"""
Strategy classifier using LLM-as-judge.

Classifies memory entries as STRATEGY (generalizable approach)
vs ANSWER (specific solution) to prevent information leakage.
"""

from typing import Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


STRATEGY_CLASSIFIER_PROMPT = """You are evaluating whether a memory entry contains a STRATEGY or an ANSWER.

A STRATEGY is:
- A generalizable approach that can be applied to similar problems
- A method, technique, or heuristic
- Does NOT contain specific solutions to specific questions

An ANSWER is:
- A specific solution to a specific question
- Contains exact answers that could be memorized
- Would only help if the exact same question is asked again

Evaluate this memory entry:
---
{memory_content}
---

Respond with ONLY one word: STRATEGY or ANSWER"""


@dataclass
class ClassificationResult:
    """Result of strategy classification."""
    content: str
    classification: str  # 'STRATEGY' or 'ANSWER'
    confidence: float
    reasoning: Optional[str] = None


class StrategyClassifier:
    """
    Classifies memory entries to prevent information leakage.

    Uses LLM-as-judge approach to determine if a memory
    contains generalizable strategies vs specific answers.
    """

    def __init__(self, llm_client=None):
        """
        Initialize classifier.

        Args:
            llm_client: LLM client for classification
        """
        self.llm_client = llm_client

        # Statistics
        self.total_classified = 0
        self.strategy_count = 0
        self.answer_count = 0

    def classify(self, memory_content: str) -> ClassificationResult:
        """
        Classify a memory entry.

        Args:
            memory_content: The memory text to classify

        Returns:
            ClassificationResult
        """
        if self.llm_client is None:
            # Fallback to heuristic classification
            return self._heuristic_classify(memory_content)

        try:
            prompt = STRATEGY_CLASSIFIER_PROMPT.format(
                memory_content=memory_content
            )

            response = self.llm_client.generate(prompt, temperature=0.1)
            response_text = response.text.strip().upper()

            if 'STRATEGY' in response_text:
                classification = 'STRATEGY'
                self.strategy_count += 1
            else:
                classification = 'ANSWER'
                self.answer_count += 1

            self.total_classified += 1

            return ClassificationResult(
                content=memory_content,
                classification=classification,
                confidence=0.85,  # LLM-based
                reasoning="LLM-as-judge classification"
            )

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return self._heuristic_classify(memory_content)

    def _heuristic_classify(self, memory_content: str) -> ClassificationResult:
        """
        Fallback heuristic classification.

        Uses patterns to guess if content is strategy vs answer.
        """
        content_lower = memory_content.lower()

        # Strategy indicators
        strategy_keywords = [
            'when', 'always', 'typically', 'approach', 'method',
            'strategy', 'technique', 'pattern', 'if you see',
            'look for', 'check', 'consider', 'first', 'then',
            'step', 'process', 'rule', 'heuristic'
        ]

        # Answer indicators
        answer_keywords = [
            'the answer is', 'correct answer:', 'solution:',
            'result:', 'output:', 'answer:', '=', 'equals'
        ]

        strategy_score = sum(1 for kw in strategy_keywords if kw in content_lower)
        answer_score = sum(1 for kw in answer_keywords if kw in content_lower)

        if strategy_score > answer_score:
            classification = 'STRATEGY'
            self.strategy_count += 1
        else:
            classification = 'ANSWER'
            self.answer_count += 1

        self.total_classified += 1

        confidence = max(0.5, min(0.8, 0.5 + 0.1 * abs(strategy_score - answer_score)))

        return ClassificationResult(
            content=memory_content,
            classification=classification,
            confidence=confidence,
            reasoning=f"Heuristic: strategy_score={strategy_score}, answer_score={answer_score}"
        )

    def is_strategy(self, memory_content: str) -> bool:
        """Check if memory is a strategy."""
        result = self.classify(memory_content)
        return result.classification == 'STRATEGY'

    def get_stats(self) -> dict:
        """Get classification statistics."""
        return {
            'total_classified': self.total_classified,
            'strategy_count': self.strategy_count,
            'answer_count': self.answer_count,
            'strategy_ratio': (
                self.strategy_count / self.total_classified
                if self.total_classified > 0 else 0
            ),
        }
