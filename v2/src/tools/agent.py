"""
Agent class with tool-based capability levels.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import uuid
import logging

from .selection import ToolSelectionPolicy
from .base import BaseTool, ToolResult, L0Tool

logger = logging.getLogger(__name__)


@dataclass
class AgentStats:
    """Statistics for an agent."""
    wins: int = 0
    losses: int = 0
    total_competitions: int = 0
    regime_wins: Dict[str, int] = field(default_factory=dict)
    tool_usage: Dict[str, int] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        """Calculate overall win rate."""
        if self.total_competitions == 0:
            return 0.0
        return self.wins / self.total_competitions

    def record_win(self, regime: str, tool: str) -> None:
        """Record a competition win."""
        self.wins += 1
        self.total_competitions += 1
        self.regime_wins[regime] = self.regime_wins.get(regime, 0) + 1
        self.tool_usage[tool] = self.tool_usage.get(tool, 0) + 1

    def record_loss(self) -> None:
        """Record a competition loss."""
        self.losses += 1
        self.total_competitions += 1


class Agent:
    """
    Agent with tool-based capability levels.

    Tool Cumulativity:
    - L0 agent: L0 only (base LLM)
    - L1 agent: L0 + L1 (base + Python)
    - L2 agent: L0 + L1 + L2 (+ Vision)
    - L3 agent: L0 + L1 + L2 + L3 (+ RAG)
    - L4 agent: L0 + L1 + L2 + L3 + L4 (+ Web)

    Agents use Thompson Sampling to discover which tools
    work best for different regimes.
    """

    # Tool names by level
    TOOL_NAMES = ['L0', 'L1', 'L2', 'L3', 'L4']

    def __init__(
        self,
        tool_level: int = 0,
        agent_id: Optional[str] = None,
        llm_client=None
    ):
        """
        Initialize agent.

        Args:
            tool_level: Maximum tool level (0-4)
            agent_id: Unique identifier (auto-generated if not provided)
            llm_client: LLM client for tool execution
        """
        self.id = agent_id or str(uuid.uuid4())[:8]
        self.tool_level = tool_level
        self.llm_client = llm_client

        # Cumulative tool access
        self.available_tools = [
            self.TOOL_NAMES[i] for i in range(tool_level + 1)
        ]

        # Tool selection policy (Thompson Sampling)
        self.tool_policy = ToolSelectionPolicy(
            available_tools=self.available_tools,
            regime_specific=True
        )

        # Statistics
        self.stats = AgentStats()

        # Memory reference (set by memory system)
        self.memory = None

        # Tool instances (lazy loaded)
        self._tools: Dict[str, BaseTool] = {}

        logger.debug(f"Created Agent {self.id} with tools: {self.available_tools}")

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a tool instance by name.

        Args:
            tool_name: Name of tool (e.g., 'L0', 'L1', ...)

        Returns:
            Tool instance or None if not available
        """
        if tool_name not in self.available_tools:
            return None

        if tool_name not in self._tools:
            # Lazy load tool
            from .base import L0Tool
            from .python_tool import PythonTool
            from .vision import VisionTool
            from .rag import RAGTool
            from .web import SafeWebTool

            tool_classes = {
                'L0': L0Tool,
                'L1': PythonTool,
                'L2': VisionTool,
                'L3': RAGTool,
                'L4': SafeWebTool,
            }

            tool_class = tool_classes.get(tool_name)
            if tool_class:
                if tool_name in ('L0', 'L2', 'L3'):
                    self._tools[tool_name] = tool_class(llm_client=self.llm_client)
                else:
                    self._tools[tool_name] = tool_class()

        return self._tools.get(tool_name)

    def select_tool(self, regime: str) -> str:
        """
        Select a tool for the given regime using Thompson Sampling.

        Args:
            regime: The task regime

        Returns:
            Selected tool name
        """
        return self.tool_policy.select({'regime': regime})

    def execute_with_tool(
        self,
        task: str,
        regime: str,
        tool_name: Optional[str] = None
    ) -> ToolResult:
        """
        Execute a task using a tool.

        Args:
            task: Task description
            regime: Task regime
            tool_name: Specific tool to use (or select via policy)

        Returns:
            ToolResult from tool execution
        """
        # Select tool if not specified
        if tool_name is None:
            tool_name = self.select_tool(regime)

        tool = self.get_tool(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool {tool_name} not available (agent level: {self.tool_level})"
            )

        # Build context with memories if available
        context = {'regime': regime}
        if self.memory is not None:
            memories = self.memory.retrieve(task, regime)
            context['memories'] = [m.content for m in memories]

        # Execute tool
        result = tool.execute(task, context)

        return result

    def update_from_competition(
        self,
        regime: str,
        tool_used: str,
        won: bool,
        response: Optional[str] = None
    ) -> None:
        """
        Update agent state after a competition.

        Args:
            regime: The regime of the competition
            tool_used: Which tool was used
            won: Whether the agent won
            response: Agent's response (for memory extraction if won)
        """
        # Update tool selection policy
        self.tool_policy.update(tool_used, won, regime)

        # Update statistics
        if won:
            self.stats.record_win(regime, tool_used)

            # Extract and store memory if available
            if self.memory is not None and response:
                self.memory.save_from_win(response, regime)
        else:
            self.stats.record_loss()

    def upgrade_tool_level(self) -> bool:
        """
        Upgrade to next tool level.

        Returns:
            True if upgraded, False if already at max level
        """
        if self.tool_level >= 4:
            return False

        self.tool_level += 1
        new_tool = self.TOOL_NAMES[self.tool_level]
        self.available_tools.append(new_tool)

        # Update tool policy with new tool
        self.tool_policy = ToolSelectionPolicy(
            available_tools=self.available_tools,
            regime_specific=True
        )

        logger.info(f"Agent {self.id} upgraded to L{self.tool_level}, tools: {self.available_tools}")
        return True

    def get_specialty(self) -> Optional[str]:
        """
        Determine agent's specialty based on win distribution.

        Returns:
            Regime name with highest wins, or None if no wins
        """
        if not self.stats.regime_wins:
            return None

        return max(self.stats.regime_wins, key=self.stats.regime_wins.get)

    def get_confidence(self, regime: str) -> float:
        """
        Get agent's confidence for a regime.

        Args:
            regime: Regime name

        Returns:
            Confidence score based on tool policy beliefs
        """
        # Average confidence across tools for this regime
        stats = self.tool_policy.get_stats(regime)
        if not stats:
            return 0.5

        return max(s['mean'] for s in stats.values())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent to dictionary."""
        return {
            'id': self.id,
            'tool_level': self.tool_level,
            'available_tools': self.available_tools,
            'tool_policy': self.tool_policy.to_dict(),
            'stats': {
                'wins': self.stats.wins,
                'losses': self.stats.losses,
                'total_competitions': self.stats.total_competitions,
                'regime_wins': self.stats.regime_wins,
                'tool_usage': self.stats.tool_usage,
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], llm_client=None) -> 'Agent':
        """Deserialize agent from dictionary."""
        agent = cls(
            tool_level=data['tool_level'],
            agent_id=data['id'],
            llm_client=llm_client
        )

        # Restore tool policy
        agent.tool_policy = ToolSelectionPolicy.from_dict(data['tool_policy'])

        # Restore stats
        stats_data = data.get('stats', {})
        agent.stats.wins = stats_data.get('wins', 0)
        agent.stats.losses = stats_data.get('losses', 0)
        agent.stats.total_competitions = stats_data.get('total_competitions', 0)
        agent.stats.regime_wins = stats_data.get('regime_wins', {})
        agent.stats.tool_usage = stats_data.get('tool_usage', {})

        return agent

    def __repr__(self) -> str:
        specialty = self.get_specialty() or "none"
        return f"Agent({self.id}, L{self.tool_level}, specialty={specialty}, wins={self.stats.wins})"
