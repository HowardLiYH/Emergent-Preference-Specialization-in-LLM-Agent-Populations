"""
Hidden 1:1 mapping between regimes and optimal tools.

This mapping is KNOWN TO THE SYSTEM but UNKNOWN TO AGENTS.
Agents must discover the optimal tool for each regime through
Thompson Sampling and competitive experience.
"""

from typing import Dict, Optional
from .config import REGIMES, RegimeConfig


# The hidden mapping - agents don't have access to this
REGIME_TO_OPTIMAL_TOOL: Dict[str, str] = {
    regime.name: regime.optimal_tool
    for regime in REGIMES.values()
}


def get_optimal_tool(regime: str) -> Optional[str]:
    """
    Get the optimal tool for a regime.

    NOTE: This function should ONLY be called by the evaluation
    system, not by agents. Agents must discover this through
    competition.

    Args:
        regime: Regime name

    Returns:
        Optimal tool name (e.g., 'L0', 'L1', ...)
    """
    return REGIME_TO_OPTIMAL_TOOL.get(regime)


def verify_agent_discovery(
    agent_id: str,
    regime: str,
    selected_tool: str
) -> bool:
    """
    Verify if an agent has discovered the optimal tool.

    For evaluation purposes only.

    Args:
        agent_id: Agent identifier
        regime: Regime name
        selected_tool: Tool the agent selected

    Returns:
        True if agent selected optimal tool
    """
    optimal = get_optimal_tool(regime)
    return selected_tool == optimal


def calculate_discovery_rate(
    agent_selections: Dict[str, Dict[str, str]]
) -> Dict[str, float]:
    """
    Calculate what percentage of agents have discovered each regime's optimal tool.

    Args:
        agent_selections: Dict[agent_id -> Dict[regime -> most_used_tool]]

    Returns:
        Dict[regime -> discovery_rate]
    """
    discovery_rates = {}

    for regime in REGIME_TO_OPTIMAL_TOOL:
        optimal = REGIME_TO_OPTIMAL_TOOL[regime]

        correct = 0
        total = 0

        for agent_id, selections in agent_selections.items():
            if regime in selections:
                total += 1
                if selections[regime] == optimal:
                    correct += 1

        discovery_rates[regime] = correct / total if total > 0 else 0.0

    return discovery_rates


# For testing/oracle purposes only
def get_all_optimal_mappings() -> Dict[str, str]:
    """
    Get all regime-to-tool mappings.

    WARNING: This should only be used for evaluation, not agent access.
    """
    return REGIME_TO_OPTIMAL_TOOL.copy()
