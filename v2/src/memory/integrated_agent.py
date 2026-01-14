"""
Agent with Integrated Memory System.

Extends BaseAgent with memory storage, retrieval, and learning.
Memory is earned through wins only (winner-only writes).
"""

import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict


@dataclass
class MemoryEntry:
    """A single memory entry."""
    content: str
    regime: str
    tool_used: str
    success: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    retrieval_count: int = 0
    
    def is_strategy(self) -> bool:
        """Check if this is a generalizable strategy vs specific answer."""
        strategy_keywords = ['always', 'when', 'if', 'use', 'apply', 'approach']
        answer_keywords = ['answer is', 'result:', 'equals', '=']
        
        content_lower = self.content.lower()
        strategy_score = sum(1 for k in strategy_keywords if k in content_lower)
        answer_score = sum(1 for k in answer_keywords if k in content_lower)
        
        return strategy_score > answer_score


@dataclass
class MemoryMetrics:
    """Tracks memory usage statistics."""
    total_writes: int = 0
    total_retrievals: int = 0
    strategy_count: int = 0
    answer_count: int = 0
    
    def retrieval_rate(self, total_tasks: int) -> float:
        return self.total_retrievals / total_tasks if total_tasks > 0 else 0
    
    def strategy_ratio(self) -> float:
        total = self.strategy_count + self.answer_count
        return self.strategy_count / total if total > 0 else 0


class MemoryIntegratedAgent:
    """
    Agent with integrated memory system.
    
    Key features:
    - Winner-only memory writes (memory is earned)
    - Hybrid retrieval (recency + semantic similarity)
    - Strategy vs answer classification
    """
    
    def __init__(self, agent_id: str, tool_level: int = 4):
        self.agent_id = agent_id
        self.tool_level = tool_level
        self.available_tools = [f'L{i}' for i in range(tool_level + 1)]
        
        # Thompson Sampling beliefs
        self.beliefs: Dict[str, Dict[str, 'BetaDistribution']] = {}
        
        # Memory system
        self.memories: List[MemoryEntry] = []
        self.memory_index: Dict[str, List[MemoryEntry]] = defaultdict(list)
        self.memory_metrics = MemoryMetrics()
        
        # Performance tracking
        self.wins = 0
        self.regime_wins: Dict[str, int] = defaultdict(int)
        self.total_attempts: Dict[str, int] = defaultdict(int)
    
    def initialize_beliefs(self, regimes: List[str]):
        """Initialize Thompson Sampling beliefs."""
        from .retriever import BetaDistribution
        for regime in regimes:
            if regime not in self.beliefs:
                self.beliefs[regime] = {t: BetaDistribution() for t in self.available_tools}
    
    def select_tool(self, regime: str, rng: random.Random) -> str:
        """Thompson Sampling for tool selection."""
        if regime not in self.beliefs:
            return rng.choice(self.available_tools)
        samples = {t: self.beliefs[regime][t].sample(rng) for t in self.available_tools}
        return max(samples, key=samples.get)
    
    def retrieve_memories(self, regime: str, query: str = None, top_k: int = 3) -> List[MemoryEntry]:
        """
        Retrieve relevant memories for a task.
        
        Uses hybrid scoring: recency + regime match + retrieval count.
        """
        candidates = self.memory_index.get(regime, [])
        
        if not candidates:
            return []
        
        # Score candidates
        scored = []
        for i, mem in enumerate(candidates):
            # Recency score (more recent = higher)
            recency = 1.0 / (len(candidates) - i + 1)
            
            # Relevance score (strategies preferred)
            relevance = 1.5 if mem.is_strategy() else 1.0
            
            # Popularity penalty (avoid over-retrieved memories)
            popularity_penalty = 1.0 / (mem.retrieval_count + 1)
            
            score = recency * relevance * popularity_penalty
            scored.append((mem, score))
        
        # Sort by score and return top-k
        scored.sort(key=lambda x: -x[1])
        retrieved = [m for m, _ in scored[:top_k]]
        
        # Update retrieval counts
        for mem in retrieved:
            mem.retrieval_count += 1
            self.memory_metrics.total_retrievals += 1
        
        return retrieved
    
    def add_memory(self, content: str, regime: str, tool: str, success: bool):
        """
        Add memory entry (only called on wins).
        
        Winner-only writes ensure memory quality.
        """
        if not success:
            return  # Only winners write to memory
        
        entry = MemoryEntry(
            content=content,
            regime=regime,
            tool_used=tool,
            success=success
        )
        
        self.memories.append(entry)
        self.memory_index[regime].append(entry)
        self.memory_metrics.total_writes += 1
        
        # Track strategy vs answer
        if entry.is_strategy():
            self.memory_metrics.strategy_count += 1
        else:
            self.memory_metrics.answer_count += 1
    
    def update(self, regime: str, tool: str, success: bool, 
               learning_content: str = None):
        """
        Update agent after task attempt.
        
        - Updates Thompson Sampling beliefs
        - Adds memory if successful (winner-only)
        """
        # Update beliefs
        if regime in self.beliefs and tool in self.beliefs[regime]:
            self.beliefs[regime][tool].update(success)
        
        self.total_attempts[regime] += 1
        
        if success:
            self.wins += 1
            self.regime_wins[regime] += 1
            
            # Add to memory (winner-only write)
            if learning_content:
                self.add_memory(learning_content, regime, tool, success)
    
    def get_specialty(self) -> Optional[str]:
        """Get regime where agent has most wins."""
        if not self.regime_wins:
            return None
        return max(self.regime_wins, key=self.regime_wins.get)
    
    def construct_prompt(self, task: str, regime: str) -> str:
        """
        Construct prompt with memory context.
        
        Retrieves relevant memories and incorporates them.
        """
        # Retrieve relevant memories
        memories = self.retrieve_memories(regime, task)
        
        prompt = f"You are Agent {self.agent_id}.\n\n"
        
        if memories:
            prompt += "Relevant past experiences:\n"
            for mem in memories:
                prompt += f"- {mem.content}\n"
            prompt += "\n"
        
        prompt += f"Task: {task}\n"
        prompt += "Provide your answer concisely."
        
        return prompt
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            'total_memories': len(self.memories),
            'total_writes': self.memory_metrics.total_writes,
            'total_retrievals': self.memory_metrics.total_retrievals,
            'strategy_ratio': self.memory_metrics.strategy_ratio(),
            'memories_per_regime': {r: len(mems) for r, mems in self.memory_index.items()},
        }


class BetaDistribution:
    """Thompson Sampling belief distribution."""
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta
    
    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(self.alpha, self.beta)
    
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)
    
    def update(self, success: bool):
        if success:
            self.alpha += 1
        else:
            self.beta += 1
