"""
Memory store for agent memories.

Handles persistence of memories as MD files.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
import json
import uuid
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    content: str
    regime: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'content': self.content,
            'regime': self.regime,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'MemoryEntry':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            content=data['content'],
            regime=data['regime'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {}),
        )


class MemoryStore:
    """
    Persistent storage for agent memories.

    Memories are stored as JSON files in agent-specific directories.
    Each agent has its own memory space.
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        max_entries_per_agent: int = 1000
    ):
        """
        Initialize memory store.

        Args:
            base_path: Base directory for memory storage
            max_entries_per_agent: Maximum entries before oldest are removed
        """
        self.base_path = Path(base_path) if base_path else Path('.memories')
        self.max_entries = max_entries_per_agent

        # In-memory cache
        self._cache: Dict[str, List[MemoryEntry]] = {}

        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_agent_path(self, agent_id: str) -> Path:
        """Get storage path for an agent."""
        return self.base_path / f"agent_{agent_id}"

    def _get_memories_file(self, agent_id: str) -> Path:
        """Get memories file path for an agent."""
        return self._get_agent_path(agent_id) / "memories.json"

    def save(
        self,
        agent_id: str,
        content: str,
        regime: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryEntry:
        """
        Save a new memory entry.

        Args:
            agent_id: Agent identifier
            content: Memory content
            regime: Regime this memory is associated with
            metadata: Optional metadata

        Returns:
            Created MemoryEntry
        """
        entry = MemoryEntry(
            id=str(uuid.uuid4())[:8],
            content=content,
            regime=regime,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        # Add to cache
        if agent_id not in self._cache:
            self._cache[agent_id] = self.load_all(agent_id)

        self._cache[agent_id].append(entry)

        # Enforce max entries
        if len(self._cache[agent_id]) > self.max_entries:
            self._cache[agent_id] = self._cache[agent_id][-self.max_entries:]

        # Persist
        self._persist(agent_id)

        logger.debug(f"Saved memory {entry.id} for agent {agent_id}")
        return entry

    def load_all(self, agent_id: str) -> List[MemoryEntry]:
        """
        Load all memories for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            List of MemoryEntry objects
        """
        # Check cache first
        if agent_id in self._cache:
            return self._cache[agent_id]

        # Load from disk
        file_path = self._get_memories_file(agent_id)

        if not file_path.exists():
            return []

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            entries = [MemoryEntry.from_dict(d) for d in data]
            self._cache[agent_id] = entries
            return entries

        except Exception as e:
            logger.error(f"Failed to load memories for {agent_id}: {e}")
            return []

    def load_by_regime(
        self,
        agent_id: str,
        regime: str,
        limit: Optional[int] = None
    ) -> List[MemoryEntry]:
        """
        Load memories for a specific regime.

        Args:
            agent_id: Agent identifier
            regime: Regime to filter by
            limit: Maximum number to return (most recent first)

        Returns:
            Filtered list of MemoryEntry objects
        """
        all_memories = self.load_all(agent_id)
        filtered = [m for m in all_memories if m.regime == regime]

        # Sort by timestamp (most recent first)
        filtered.sort(key=lambda m: m.timestamp, reverse=True)

        if limit:
            filtered = filtered[:limit]

        return filtered

    def _persist(self, agent_id: str) -> None:
        """Persist memories to disk."""
        agent_path = self._get_agent_path(agent_id)
        agent_path.mkdir(parents=True, exist_ok=True)

        file_path = self._get_memories_file(agent_id)

        memories = self._cache.get(agent_id, [])
        data = [m.to_dict() for m in memories]

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def delete(self, agent_id: str, memory_id: str) -> bool:
        """
        Delete a specific memory.

        Args:
            agent_id: Agent identifier
            memory_id: Memory ID to delete

        Returns:
            True if deleted, False if not found
        """
        if agent_id not in self._cache:
            self._cache[agent_id] = self.load_all(agent_id)

        original_count = len(self._cache[agent_id])
        self._cache[agent_id] = [
            m for m in self._cache[agent_id]
            if m.id != memory_id
        ]

        if len(self._cache[agent_id]) < original_count:
            self._persist(agent_id)
            return True

        return False

    def clear(self, agent_id: str) -> None:
        """Clear all memories for an agent."""
        self._cache[agent_id] = []
        self._persist(agent_id)

    def get_stats(self, agent_id: str) -> dict:
        """Get memory statistics for an agent."""
        memories = self.load_all(agent_id)

        regime_counts = {}
        for m in memories:
            regime_counts[m.regime] = regime_counts.get(m.regime, 0) + 1

        return {
            'total_memories': len(memories),
            'memories_by_regime': regime_counts,
            'oldest': memories[0].timestamp.isoformat() if memories else None,
            'newest': memories[-1].timestamp.isoformat() if memories else None,
        }
