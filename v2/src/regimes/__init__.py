"""
Non-uniform regime system with hidden 1:1 tool mapping.
"""

from .config import RegimeConfig, REGIMES
from .mapping import REGIME_TO_OPTIMAL_TOOL
from .sampler import RegimeSampler

__all__ = [
    "RegimeConfig",
    "REGIMES",
    "REGIME_TO_OPTIMAL_TOOL",
    "RegimeSampler",
]
