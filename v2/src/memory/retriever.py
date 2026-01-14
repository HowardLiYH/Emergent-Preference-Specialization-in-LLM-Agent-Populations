"""
Memory retrieval with frozen embeddings.

Uses hybrid retrieval: recency (40%) + semantic similarity (60%).
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from collections import defaultdict
import logging
import time

from .store import MemoryStore, MemoryEntry

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from memory retrieval."""
    memory: MemoryEntry
    score: float
    recency_score: float
    similarity_score: float


class MemoryRetriever:
    """
    Hybrid memory retrieval system.

    Features:
    - Frozen embeddings (text-embedding-004)
    - Regime-indexed for O(1) lookup
    - Hybrid scoring: recency + semantic similarity
    """

    # Retrieval weights
    RECENCY_WEIGHT = 0.4
    SIMILARITY_WEIGHT = 0.6

    def __init__(
        self,
        memory_store: Optional[MemoryStore] = None,
        embedder=None,
        recency_decay: float = 0.95  # Per-day decay
    ):
        """
        Initialize memory retriever.

        Args:
            memory_store: Memory store instance
            embedder: Embedding model (frozen, pre-trained)
            recency_decay: Exponential decay rate for recency
        """
        self.store = memory_store or MemoryStore()
        self.embedder = embedder
        self.recency_decay = recency_decay

        # Regime index for O(1) lookup
        self._regime_index: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )  # agent_id -> regime -> [memory_ids]

        # Embedding cache
        self._embedding_cache: Dict[str, List[float]] = {}

    def _compute_embedding(self, text: str) -> Optional[List[float]]:
        """
        Compute embedding for text.

        Uses frozen, pre-trained embeddings.
        """
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        if self.embedder is None:
            return None

        try:
            embedding = self.embedder.encode(text).tolist()
            self._embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            logger.warning(f"Failed to compute embedding: {e}")
            return None

    def _compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between embeddings."""
        import numpy as np

        e1 = np.array(embedding1)
        e2 = np.array(embedding2)

        dot_product = np.dot(e1, e2)
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _compute_recency_score(self, memory: MemoryEntry) -> float:
        """
        Compute recency score for a memory.

        Uses exponential decay based on age.
        """
        from datetime import datetime

        now = datetime.now()
        age_days = (now - memory.timestamp).total_seconds() / 86400

        return self.recency_decay ** age_days

    def retrieve(
        self,
        agent_id: str,
        query: str,
        regime: Optional[str] = None,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant memories for a query.

        Args:
            agent_id: Agent identifier
            query: Query text
            regime: Optional regime filter (O(1) lookup if provided)
            top_k: Number of memories to return

        Returns:
            List of RetrievalResult sorted by relevance
        """
        # Get candidate memories
        if regime:
            # O(1) regime-indexed lookup
            memories = self.store.load_by_regime(agent_id, regime)
        else:
            # Load all memories
            memories = self.store.load_all(agent_id)

        if not memories:
            return []

        # Compute query embedding
        query_embedding = self._compute_embedding(query)

        # Score all memories
        results = []
        for memory in memories:
            # Recency score
            recency_score = self._compute_recency_score(memory)

            # Similarity score
            if query_embedding and memory.embedding:
                similarity_score = self._compute_similarity(
                    query_embedding, memory.embedding
                )
            elif query_embedding:
                # Compute and cache embedding for memory
                memory_embedding = self._compute_embedding(memory.content)
                if memory_embedding:
                    similarity_score = self._compute_similarity(
                        query_embedding, memory_embedding
                    )
                    memory.embedding = memory_embedding
                else:
                    similarity_score = 0.5  # Neutral
            else:
                # No embeddings available, use keyword overlap
                similarity_score = self._keyword_similarity(query, memory.content)

            # Hybrid score
            score = (
                self.RECENCY_WEIGHT * recency_score +
                self.SIMILARITY_WEIGHT * similarity_score
            )

            results.append(RetrievalResult(
                memory=memory,
                score=score,
                recency_score=recency_score,
                similarity_score=similarity_score
            ))

        # Sort by score and return top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _keyword_similarity(self, query: str, content: str) -> float:
        """Fallback keyword-based similarity."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words:
            return 0.0

        overlap = len(query_words & content_words)
        return overlap / len(query_words)

    def index_memory(
        self,
        agent_id: str,
        memory: MemoryEntry
    ) -> None:
        """
        Add memory to regime index.

        Args:
            agent_id: Agent identifier
            memory: Memory to index
        """
        self._regime_index[agent_id][memory.regime].append(memory.id)

    def rebuild_index(self, agent_id: str) -> None:
        """Rebuild regime index for an agent."""
        memories = self.store.load_all(agent_id)

        self._regime_index[agent_id] = defaultdict(list)

        for memory in memories:
            self._regime_index[agent_id][memory.regime].append(memory.id)

    def get_regime_memory_counts(self, agent_id: str) -> Dict[str, int]:
        """Get memory counts by regime."""
        if agent_id not in self._regime_index:
            self.rebuild_index(agent_id)

        return {
            regime: len(ids)
            for regime, ids in self._regime_index[agent_id].items()
        }
