"""
Specialist Routing Mechanisms

Implements different strategies for routing tasks to the appropriate
specialist agent in a population.

Routing Options (Panel Modification #2):
1. Oracle Routing: Perfect task-type labels (upper bound)
2. Learned Routing: Classifier predicts task type
3. Confidence Routing: Route to highest-confidence specialist
4. Ensemble: All specialists answer, take majority vote

Ground Rules Applied:
- Checkpoints for long operations
- API keys from .env
- Unified model across experiments
"""

import os
import json
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from pathlib import Path
import random

# Local imports (avoid circular)
try:
    from .synthetic_rules import RuleType, RuleTask
    from .preference_agent import PreferenceAgent
    from .llm_client import LLMClient
except ImportError:
    # For standalone testing
    RuleType = None
    RuleTask = None
    PreferenceAgent = None
    LLMClient = None


class RoutingStrategy(Enum):
    """Available routing strategies."""
    ORACLE = "oracle"           # Perfect knowledge of task type
    LEARNED = "learned"         # Classifier-based routing
    CONFIDENCE = "confidence"   # Route to highest-confidence agent
    ENSEMBLE = "ensemble"       # All agents answer, majority vote
    RANDOM = "random"           # Random agent (baseline)


@dataclass
class RoutingResult:
    """Result of routing a task to agent(s)."""
    strategy: RoutingStrategy
    selected_agent_id: Optional[str]
    selected_agents: List[str]  # For ensemble
    confidence: float
    routing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingMetrics:
    """Metrics for evaluating routing quality."""
    accuracy: float                  # How often correct agent was selected
    oracle_gap: float               # Difference from oracle performance
    routing_overhead_ms: float      # Average routing time
    diversity: float                # How evenly tasks are distributed

    def to_dict(self) -> Dict:
        return {
            "accuracy": self.accuracy,
            "oracle_gap": self.oracle_gap,
            "routing_overhead_ms": self.routing_overhead_ms,
            "diversity": self.diversity
        }


# =============================================================================
# ROUTER BASE CLASS
# =============================================================================

class BaseRouter:
    """Base class for all routing strategies."""

    def __init__(self, agents: List["PreferenceAgent"]):
        self.agents = agents
        self.agent_map = {a.agent_id: a for a in agents}
        self.routing_history: List[RoutingResult] = []

    def route(self, task: "RuleTask") -> RoutingResult:
        """Route a task to the appropriate agent(s). Override in subclass."""
        raise NotImplementedError

    def get_metrics(self) -> RoutingMetrics:
        """Compute routing metrics from history."""
        if not self.routing_history:
            return RoutingMetrics(0.0, 0.0, 0.0, 0.0)

        # Compute average routing time
        avg_time = sum(r.routing_time_ms for r in self.routing_history) / len(self.routing_history)

        # Compute diversity (how evenly distributed)
        agent_counts = {}
        for r in self.routing_history:
            for aid in r.selected_agents or [r.selected_agent_id]:
                if aid:
                    agent_counts[aid] = agent_counts.get(aid, 0) + 1

        if agent_counts:
            total = sum(agent_counts.values())
            probs = [c / total for c in agent_counts.values()]
            # Normalized entropy as diversity measure
            import math
            max_entropy = math.log(len(self.agents)) if len(self.agents) > 1 else 1
            entropy = -sum(p * math.log(p) for p in probs if p > 0)
            diversity = entropy / max_entropy if max_entropy > 0 else 0
        else:
            diversity = 0.0

        return RoutingMetrics(
            accuracy=0.0,  # Computed externally with ground truth
            oracle_gap=0.0,  # Computed externally
            routing_overhead_ms=avg_time,
            diversity=diversity
        )


# =============================================================================
# ORACLE ROUTER (Upper Bound)
# =============================================================================

class OracleRouter(BaseRouter):
    """
    Perfect routing using ground truth task labels.

    This serves as an upper bound for routing performance.
    In practice, we know which rule each task tests.
    """

    def route(self, task: "RuleTask") -> RoutingResult:
        """Route task to the specialist for its rule."""
        import time
        start = time.time()

        # Find agent specialized in this task's rule
        best_agent = None
        best_level = -1

        for agent in self.agents:
            level = agent.strategy_levels.get(task.rule_type, 0)
            if level > best_level:
                best_level = level
                best_agent = agent

        elapsed_ms = (time.time() - start) * 1000

        result = RoutingResult(
            strategy=RoutingStrategy.ORACLE,
            selected_agent_id=best_agent.agent_id if best_agent else None,
            selected_agents=[best_agent.agent_id] if best_agent else [],
            confidence=1.0,
            routing_time_ms=elapsed_ms,
            metadata={"rule_type": task.rule_type.value, "agent_level": best_level}
        )

        self.routing_history.append(result)
        return result


# =============================================================================
# LEARNED ROUTER (Classifier-Based)
# =============================================================================

class LearnedRouter(BaseRouter):
    """
    Route using a trained classifier that predicts task type.

    The classifier is trained on task features to predict which
    specialist should handle it.
    """

    def __init__(
        self,
        agents: List["PreferenceAgent"],
        classifier: Optional[Callable] = None,
        client: "LLMClient" = None
    ):
        super().__init__(agents)
        self.classifier = classifier
        self.client = client
        self.training_data: List[Tuple[str, str]] = []  # (task_text, rule_type)

    async def train(self, training_tasks: List["RuleTask"]):
        """
        Train the classifier on labeled tasks.

        For simplicity, we use the LLM as a zero-shot classifier.
        """
        # Store training data for few-shot prompting
        self.training_data = [
            (task.prompt[:200], task.rule_type.value)
            for task in training_tasks[:20]  # Keep 20 examples
        ]

    def _build_classification_prompt(self, task: "RuleTask") -> str:
        """Build prompt for LLM-based classification."""
        rule_names = [r.value for r in RuleType] if RuleType else []

        examples = ""
        for text, rule in self.training_data[:5]:
            examples += f"Task: {text[:100]}...\nType: {rule}\n\n"

        prompt = f"""Classify this task into one of these categories: {rule_names}

Examples:
{examples}
Now classify this task:
Task: {task.prompt[:200]}
Type:"""

        return prompt

    async def route_async(self, task: "RuleTask") -> RoutingResult:
        """Route task using LLM classifier."""
        import time
        start = time.time()

        if self.client and RuleType:
            prompt = self._build_classification_prompt(task)
            try:
                response = await self.client.generate(prompt, temperature=0.1, max_tokens=50)
                predicted_rule = response.strip().lower()

                # Match to RuleType
                matched_rule = None
                for r in RuleType:
                    if r.value in predicted_rule:
                        matched_rule = r
                        break

                # Find best agent for predicted rule
                best_agent = None
                best_level = -1
                for agent in self.agents:
                    level = agent.strategy_levels.get(matched_rule, 0) if matched_rule else 0
                    if level > best_level:
                        best_level = level
                        best_agent = agent

            except Exception as e:
                # Fallback to random
                best_agent = random.choice(self.agents)
                matched_rule = None
        else:
            # Fallback: use rule from task directly
            best_agent = self.agents[0] if self.agents else None
            matched_rule = task.rule_type if hasattr(task, 'rule_type') else None

        elapsed_ms = (time.time() - start) * 1000

        result = RoutingResult(
            strategy=RoutingStrategy.LEARNED,
            selected_agent_id=best_agent.agent_id if best_agent else None,
            selected_agents=[best_agent.agent_id] if best_agent else [],
            confidence=0.8,  # Classifier confidence
            routing_time_ms=elapsed_ms,
            metadata={"predicted_rule": matched_rule.value if matched_rule else None}
        )

        self.routing_history.append(result)
        return result

    def route(self, task: "RuleTask") -> RoutingResult:
        """Synchronous wrapper for route_async."""
        return asyncio.run(self.route_async(task))


# =============================================================================
# CONFIDENCE ROUTER
# =============================================================================

class ConfidenceRouter(BaseRouter):
    """
    Route to the agent with highest confidence on the task.

    Each agent is asked to provide a confidence score, and the
    task is routed to the most confident agent.
    """

    def __init__(self, agents: List["PreferenceAgent"], client: "LLMClient" = None):
        super().__init__(agents)
        self.client = client

    async def get_agent_confidence(
        self,
        agent: "PreferenceAgent",
        task: "RuleTask"
    ) -> float:
        """Get agent's confidence on a task."""
        if not self.client:
            # Fallback: use strategy level as proxy for confidence
            max_level = max(agent.strategy_levels.values()) if agent.strategy_levels else 0
            return 0.3 + 0.2 * max_level

        # Build prompt asking for confidence
        from .rule_strategies import build_prompt_from_levels
        system_prompt = build_prompt_from_levels(agent.strategy_levels)

        confidence_prompt = f"""Rate your confidence (0-100) in answering this question correctly.
Only respond with a number.

Question: {task.prompt[:300]}"""

        try:
            response = await self.client.generate(
                confidence_prompt,
                system=system_prompt,
                temperature=0.1,
                max_tokens=20
            )
            # Parse confidence
            import re
            numbers = re.findall(r'\d+', response)
            if numbers:
                conf = min(100, max(0, int(numbers[0])))
                return conf / 100.0
        except:
            pass

        return 0.5

    async def route_async(self, task: "RuleTask") -> RoutingResult:
        """Route to highest-confidence agent."""
        import time
        start = time.time()

        # Get confidence from all agents (in parallel)
        confidences = await asyncio.gather(*[
            self.get_agent_confidence(agent, task)
            for agent in self.agents
        ])

        # Select highest confidence agent
        best_idx = max(range(len(confidences)), key=lambda i: confidences[i])
        best_agent = self.agents[best_idx]
        best_confidence = confidences[best_idx]

        elapsed_ms = (time.time() - start) * 1000

        result = RoutingResult(
            strategy=RoutingStrategy.CONFIDENCE,
            selected_agent_id=best_agent.agent_id,
            selected_agents=[best_agent.agent_id],
            confidence=best_confidence,
            routing_time_ms=elapsed_ms,
            metadata={
                "all_confidences": {
                    self.agents[i].agent_id: confidences[i]
                    for i in range(len(self.agents))
                }
            }
        )

        self.routing_history.append(result)
        return result

    def route(self, task: "RuleTask") -> RoutingResult:
        """Synchronous wrapper."""
        return asyncio.run(self.route_async(task))


# =============================================================================
# ENSEMBLE ROUTER
# =============================================================================

class EnsembleRouter(BaseRouter):
    """
    All specialists answer, take majority vote.

    Most expensive but potentially most accurate approach.
    """

    def __init__(self, agents: List["PreferenceAgent"], client: "LLMClient" = None):
        super().__init__(agents)
        self.client = client

    async def get_agent_answer(
        self,
        agent: "PreferenceAgent",
        task: "RuleTask"
    ) -> Tuple[str, float]:
        """Get agent's answer and confidence."""
        if not self.client:
            return "A", 0.5  # Fallback

        from .rule_strategies import build_prompt_from_levels
        system_prompt = build_prompt_from_levels(agent.strategy_levels)

        answer_prompt = f"""Answer this question. Provide your answer and confidence (0-100).

Format:
ANSWER: [your answer]
CONFIDENCE: [0-100]

Question: {task.prompt}"""

        try:
            response = await self.client.generate(
                answer_prompt,
                system=system_prompt,
                temperature=0.3,
                max_tokens=100
            )

            # Parse answer and confidence
            import re
            answer_match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
            conf_match = re.search(r'CONFIDENCE:\s*(\d+)', response, re.IGNORECASE)

            answer = answer_match.group(1).strip() if answer_match else response[:50]
            conf = int(conf_match.group(1)) / 100.0 if conf_match else 0.5

            return answer, conf
        except:
            return "A", 0.5

    async def route_async(self, task: "RuleTask") -> RoutingResult:
        """Get answers from all agents, return majority vote."""
        import time
        start = time.time()

        # Get all agent answers
        results = await asyncio.gather(*[
            self.get_agent_answer(agent, task)
            for agent in self.agents
        ])

        # Majority vote (weighted by confidence)
        votes: Dict[str, float] = {}
        for (answer, conf), agent in zip(results, self.agents):
            # Normalize answer
            answer_key = answer.strip().upper()[:10]
            votes[answer_key] = votes.get(answer_key, 0) + conf

        # Find winner
        if votes:
            winner = max(votes.keys(), key=lambda k: votes[k])
            total_conf = votes[winner] / len(self.agents)
        else:
            winner = "A"
            total_conf = 0.5

        elapsed_ms = (time.time() - start) * 1000

        result = RoutingResult(
            strategy=RoutingStrategy.ENSEMBLE,
            selected_agent_id=None,  # All agents participated
            selected_agents=[a.agent_id for a in self.agents],
            confidence=total_conf,
            routing_time_ms=elapsed_ms,
            metadata={
                "ensemble_answer": winner,
                "vote_distribution": votes,
                "individual_answers": {
                    self.agents[i].agent_id: results[i][0]
                    for i in range(len(self.agents))
                }
            }
        )

        self.routing_history.append(result)
        return result

    def route(self, task: "RuleTask") -> RoutingResult:
        """Synchronous wrapper."""
        return asyncio.run(self.route_async(task))


# =============================================================================
# RANDOM ROUTER (Baseline)
# =============================================================================

class RandomRouter(BaseRouter):
    """Random routing as baseline."""

    def __init__(self, agents: List["PreferenceAgent"], seed: int = 42):
        super().__init__(agents)
        self.rng = random.Random(seed)

    def route(self, task: "RuleTask") -> RoutingResult:
        """Randomly select an agent."""
        import time
        start = time.time()

        agent = self.rng.choice(self.agents)

        elapsed_ms = (time.time() - start) * 1000

        result = RoutingResult(
            strategy=RoutingStrategy.RANDOM,
            selected_agent_id=agent.agent_id,
            selected_agents=[agent.agent_id],
            confidence=1.0 / len(self.agents),
            routing_time_ms=elapsed_ms,
            metadata={}
        )

        self.routing_history.append(result)
        return result


# =============================================================================
# ROUTER FACTORY
# =============================================================================

def create_router(
    strategy: RoutingStrategy,
    agents: List["PreferenceAgent"],
    client: "LLMClient" = None,
    **kwargs
) -> BaseRouter:
    """
    Factory function to create routers.

    Args:
        strategy: Which routing strategy to use
        agents: List of specialist agents
        client: LLM client for learned/confidence/ensemble routing
        **kwargs: Additional arguments for specific routers

    Returns:
        Configured router instance
    """
    router_classes = {
        RoutingStrategy.ORACLE: OracleRouter,
        RoutingStrategy.LEARNED: LearnedRouter,
        RoutingStrategy.CONFIDENCE: ConfidenceRouter,
        RoutingStrategy.ENSEMBLE: EnsembleRouter,
        RoutingStrategy.RANDOM: RandomRouter,
    }

    router_class = router_classes.get(strategy)
    if not router_class:
        raise ValueError(f"Unknown routing strategy: {strategy}")

    if strategy in [RoutingStrategy.LEARNED, RoutingStrategy.CONFIDENCE, RoutingStrategy.ENSEMBLE]:
        return router_class(agents, client=client, **kwargs)
    else:
        return router_class(agents, **kwargs)


# =============================================================================
# ROUTING COMPARISON UTILITIES
# =============================================================================

async def compare_routing_strategies(
    agents: List["PreferenceAgent"],
    tasks: List["RuleTask"],
    client: "LLMClient" = None,
    strategies: List[RoutingStrategy] = None
) -> Dict[str, RoutingMetrics]:
    """
    Compare different routing strategies on same task set.

    Returns metrics for each strategy.
    """
    if strategies is None:
        strategies = list(RoutingStrategy)

    results = {}

    for strategy in strategies:
        router = create_router(strategy, agents, client)

        for task in tasks:
            if asyncio.iscoroutinefunction(router.route):
                await router.route_async(task)
            else:
                router.route(task)

        results[strategy.value] = router.get_metrics()

    return results


if __name__ == "__main__":
    # Demo routing
    print("Routing Module Loaded")
    print(f"Available strategies: {[s.value for s in RoutingStrategy]}")
