"""
Memory indexer combining embedding and storage.

Provides high-level interface for indexing and retrieving memories.
Automatically generates embeddings for text content.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from src.exocortex.memory import Memory
from src.exocortex.storage import QdrantStorage
from src.exocortex.embedder import MemoryEmbedder

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A search result with memory and similarity score."""
    memory: Memory
    score: float  # 0.0 to 1.0, higher is more similar


class MemoryIndexer:
    """High-level memory indexing and retrieval."""

    def __init__(
        self,
        storage_path: Path,
        collection_name: str = "memories",
        device: str = "cpu"
    ):
        """Initialize memory indexer.

        Args:
            storage_path: Directory for storage files
            collection_name: Qdrant collection name
            device: Device for embeddings ("cpu", "cuda", "mps")

        Raises:
            RuntimeError: If initialization fails
        """
        self.storage_path = Path(storage_path)
        self.collection_name = collection_name

        logger.info(f"Initializing memory indexer at {storage_path}")

        try:
            self.storage = QdrantStorage(
                storage_path=self.storage_path,
                collection_name=collection_name
            )
            self.embedder = MemoryEmbedder(device=device)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize indexer: {e}") from e

    def index_memory(self, memory: Memory):
        """Index a memory with auto-embedding if needed.

        Args:
            memory: Memory instance to index

        Raises:
            ValueError: If memory is invalid
            RuntimeError: If indexing fails
        """
        # Auto-generate embedding if missing
        if memory.embedding is None:
            if not memory.text:
                raise ValueError(
                    f"Cannot index memory {memory.memory_id}: "
                    "no text and no embedding"
                )

            logger.debug(f"Generating embedding for memory {memory.memory_id}")
            memory.embedding = self.embedder.embed(memory.text)

        # Store memory
        logger.debug(f"Storing memory {memory.memory_id}")
        self.storage.store_memory(memory)
        logger.info(f"Indexed memory {memory.memory_id}")

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID.

        Args:
            memory_id: Memory ID to retrieve

        Returns:
            Memory instance if found, None otherwise

        Raises:
            RuntimeError: If retrieval fails
        """
        return self.storage.get_memory_by_id(memory_id)

    def search_memories(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0
    ) -> list[SearchResult]:
        """Search for memories semantically similar to query.

        Args:
            query: Natural language search query
            limit: Maximum number of results (default 10)
            min_score: Minimum similarity score 0.0-1.0 (default 0.0)

        Returns:
            List of SearchResult objects, sorted by relevance

        Raises:
            ValueError: If query is empty or parameters invalid
            RuntimeError: If search fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")

        logger.debug(f"Searching for: {query[:50]}... (limit={limit})")

        try:
            # Generate embedding for query
            query_embedding = self.embedder.embed(query)

            # Search storage
            results = self.storage.search_similar(
                query_embedding=query_embedding,
                limit=limit,
                min_score=min_score
            )

            # Convert to SearchResult objects
            search_results = [
                SearchResult(memory=memory, score=score)
                for memory, score in results
            ]

            logger.info(f"Found {len(search_results)} results for query")
            return search_results

        except Exception as e:
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            raise RuntimeError(f"Search failed: {e}") from e

    def close(self):
        """Close indexer and release resources.

        Closes storage connections and releases embedding model.
        Safe to call multiple times.
        """
        logger.debug("Closing memory indexer")

        if hasattr(self, 'storage'):
            self.storage.close()

        if hasattr(self, 'embedder'):
            self.embedder.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
