"""
L3: RAG (Retrieval-Augmented Generation) tool.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
import json

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A document in the knowledge base."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class RetrievalResult:
    """Result from document retrieval."""
    document: Document
    score: float


class RAGTool(BaseTool):
    """
    L3: RAG/Retrieval tool.

    Provides document retrieval capabilities:
    - Semantic search over documents
    - File reading and processing
    - Knowledge base querying
    """

    name: str = "rag"
    level: int = 3

    def __init__(
        self,
        embedder=None,
        llm_client=None,
        knowledge_base_path: Optional[Path] = None
    ):
        """
        Initialize RAG tool.

        Args:
            embedder: Embedding model for semantic search
            llm_client: LLM client for generation
            knowledge_base_path: Path to knowledge base directory
        """
        self.embedder = embedder
        self.llm_client = llm_client
        self.knowledge_base_path = knowledge_base_path
        self.documents: Dict[str, Document] = {}
        self._index_built = False

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a document to the knowledge base.

        Args:
            doc_id: Unique document identifier
            content: Document content
            metadata: Optional metadata

        Returns:
            True if added successfully
        """
        try:
            # Compute embedding if embedder available
            embedding = None
            if self.embedder is not None:
                embedding = self.embedder.encode(content).tolist()

            self.documents[doc_id] = Document(
                id=doc_id,
                content=content,
                metadata=metadata or {},
                embedding=embedding
            )
            return True

        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {e}")
            return False

    def _compute_similarity(
        self,
        query_embedding: List[float],
        doc_embedding: List[float]
    ) -> float:
        """Compute cosine similarity between embeddings."""
        import numpy as np

        q = np.array(query_embedding)
        d = np.array(doc_embedding)

        # Cosine similarity
        return float(np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d) + 1e-8))

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filter_metadata: Optional metadata filter

        Returns:
            List of RetrievalResult sorted by relevance
        """
        if not self.documents:
            return []

        if self.embedder is None:
            # Fallback to keyword matching
            return self._keyword_retrieve(query, top_k, filter_metadata)

        # Compute query embedding
        query_embedding = self.embedder.encode(query).tolist()

        # Score all documents
        results = []
        for doc in self.documents.values():
            # Apply metadata filter
            if filter_metadata:
                if not all(
                    doc.metadata.get(k) == v
                    for k, v in filter_metadata.items()
                ):
                    continue

            if doc.embedding is None:
                continue

            score = self._compute_similarity(query_embedding, doc.embedding)
            results.append(RetrievalResult(document=doc, score=score))

        # Sort by score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _keyword_retrieve(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Fallback keyword-based retrieval."""
        query_words = set(query.lower().split())

        results = []
        for doc in self.documents.values():
            # Apply metadata filter
            if filter_metadata:
                if not all(
                    doc.metadata.get(k) == v
                    for k, v in filter_metadata.items()
                ):
                    continue

            # Simple keyword overlap score
            doc_words = set(doc.content.lower().split())
            overlap = len(query_words & doc_words)
            score = overlap / len(query_words) if query_words else 0

            if score > 0:
                results.append(RetrievalResult(document=doc, score=score))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def execute(
        self,
        query: str,
        context: Optional[dict] = None
    ) -> ToolResult:
        """
        Execute RAG: retrieve relevant documents and generate answer.

        Args:
            query: Question to answer
            context: Optional additional context

        Returns:
            ToolResult with answer grounded in retrieved documents
        """
        # Retrieve relevant documents
        filter_metadata = context.get('filter') if context else None
        top_k = context.get('top_k', 5) if context else 5

        retrieved = self.retrieve(query, top_k=top_k, filter_metadata=filter_metadata)

        if not retrieved:
            return ToolResult(
                success=True,
                output="No relevant documents found in knowledge base.",
                tokens_used=0
            )

        # Build context from retrieved documents
        doc_context = "\n\n".join([
            f"[Document {i+1}] (score: {r.score:.3f})\n{r.document.content}"
            for i, r in enumerate(retrieved)
        ])

        if self.llm_client is None:
            # Return just the retrieved context
            return ToolResult(
                success=True,
                output=f"Retrieved documents:\n\n{doc_context}",
                tokens_used=0
            )

        try:
            # Generate answer using retrieved context
            prompt = f"""Answer the following question based on the provided context.

Context:
{doc_context}

Question: {query}

Answer based only on the provided context. If the context doesn't contain enough information, say so."""

            response = self.llm_client.generate(prompt)

            return ToolResult(
                success=True,
                output=response.text,
                tokens_used=response.tokens_used
            )

        except Exception as e:
            logger.error(f"RAG generation failed: {e}")
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )

    def load_documents_from_directory(self, directory: Path) -> int:
        """
        Load all text documents from a directory.

        Args:
            directory: Path to directory containing documents

        Returns:
            Number of documents loaded
        """
        count = 0
        directory = Path(directory)

        for file_path in directory.glob("**/*.txt"):
            try:
                content = file_path.read_text()
                doc_id = str(file_path.relative_to(directory))

                self.add_document(
                    doc_id=doc_id,
                    content=content,
                    metadata={
                        'path': str(file_path),
                        'filename': file_path.name,
                    }
                )
                count += 1

            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        # Also load JSON documents
        for file_path in directory.glob("**/*.json"):
            try:
                content = file_path.read_text()
                data = json.loads(content)

                # Handle structured JSON
                if isinstance(data, dict):
                    content = json.dumps(data, indent=2)
                elif isinstance(data, list):
                    content = "\n".join(str(item) for item in data)

                doc_id = str(file_path.relative_to(directory))
                self.add_document(
                    doc_id=doc_id,
                    content=content,
                    metadata={
                        'path': str(file_path),
                        'filename': file_path.name,
                        'type': 'json'
                    }
                )
                count += 1

            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        logger.info(f"Loaded {count} documents from {directory}")
        return count
