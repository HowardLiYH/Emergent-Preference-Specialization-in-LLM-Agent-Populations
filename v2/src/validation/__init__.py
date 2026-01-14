"""
Anti-leakage validation protocol.

- Strategy classifier (LLM-as-judge)
- Generalization tests
- Fresh agent tests
- Ablation controls
"""

from .strategy_classifier import StrategyClassifier
from .generalization import GeneralizationTest
from .fresh_agent import FreshAgentTest
from .ablation import AblationControl

__all__ = [
    "StrategyClassifier",
    "GeneralizationTest",
    "FreshAgentTest",
    "AblationControl",
]
