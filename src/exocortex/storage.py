"""
Storage layer for exocortex memories using Qdrant + SQLite.

This module provides hybrid storage:
- Qdrant: Fast vector similarity search
- SQLite: Rich metadata queries

Both are embedded (no external servers required).
"""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
from src.exocortex.memory import Memory, EMBEDDING_DIM


class QdrantStorage:
    """Hybrid storage using Qdrant (vectors) + SQLite (metadata)."""

    def __init__(self, storage_path: Path, collection_name: str = "memories"):
        """Initialize storage.

        Args:
            storage_path: Directory for storage files
            collection_name: Qdrant collection name

        Raises:
            ValueError: If storage_path is invalid
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        # Initialize Qdrant (embedded mode)
        qdrant_path = self.storage_path / "qdrant"
        self.qdrant = QdrantClient(path=str(qdrant_path))

        # Create collection if it doesn't exist
        try:
            self.qdrant.get_collection(collection_name)
        except (UnexpectedResponse, ValueError) as e:
            # Only create if collection truly doesn't exist
            if "not found" in str(e).lower() or "doesn't exist" in str(e).lower():
                self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIM,
                        distance=Distance.COSINE
                    )
                )
            else:
                raise  # Re-raise other Qdrant errors

        # Initialize SQLite with thread-local connections
        self._db_path = self.storage_path / "metadata.db"
        self._thread_local = threading.local()

        # Initialize schema using a temporary connection
        init_db = sqlite3.connect(str(self._db_path))
        self._init_db_schema(init_db)
        init_db.close()

    @property
    def db(self):
        """Thread-local database connection."""
        if not hasattr(self._thread_local, 'connection'):
            self._thread_local.connection = sqlite3.connect(str(self._db_path))
        return self._thread_local.connection

    def _init_db_schema(self, conn):
        """Initialize SQLite schema on given connection."""
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

    def _memory_id_to_int(self, memory_id: str) -> int:
        """Convert memory ID to deterministic integer for Qdrant.

        Args:
            memory_id: Memory ID in format "mem_<12-hex-chars>"

        Returns:
            Deterministic integer representation
        """
        # Extract hex portion and convert to int
        hex_part = memory_id.replace("mem_", "")
        return int(hex_part, 16)

    def store_memory(self, memory: Memory):
        """Store a memory with its embedding.

        Args:
            memory: Memory instance to store

        Raises:
            ValueError: If memory has no embedding or is invalid
            RuntimeError: If storage operation fails
        """
        if memory.embedding is None:
            raise ValueError("Memory must have an embedding to be stored")

        if len(memory.embedding) != EMBEDDING_DIM:
            raise ValueError(f"Embedding must be {EMBEDDING_DIM}-dim, got {len(memory.embedding)}")

        try:
            # Store vector in Qdrant first
            point = PointStruct(
                id=self._memory_id_to_int(memory.memory_id),
                vector=memory.embedding,
                payload={"memory_id": memory.memory_id}
            )
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            # Then store metadata in SQLite
            cursor = self.db.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO memories (memory_id, data) VALUES (?, ?)",
                (memory.memory_id, json.dumps(memory.to_dict()))
            )
            self.db.commit()

        except Exception as e:
            # Rollback SQLite if needed
            self.db.rollback()
            raise RuntimeError(f"Failed to store memory {memory.memory_id}: {e}") from e

    def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID.

        Args:
            memory_id: Memory ID to retrieve

        Returns:
            Memory instance if found, None otherwise

        Raises:
            RuntimeError: If retrieval fails
        """
        try:
            cursor = self.db.cursor()
            cursor.execute(
                "SELECT data FROM memories WHERE memory_id = ?",
                (memory_id,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            # Deserialize memory
            data = json.loads(row[0])
            return Memory.from_dict(data)

        except json.JSONDecodeError as e:
            raise RuntimeError(f"Corrupted data for memory {memory_id}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve memory {memory_id}: {e}") from e

    def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 10,
        min_score: float = 0.0
    ) -> list[tuple[Memory, float]]:
        """Search for memories similar to query embedding.

        Args:
            query_embedding: 384-dim query vector
            limit: Maximum number of results
            min_score: Minimum similarity score (0.0-1.0)

        Returns:
            List of (Memory, score) tuples, sorted by similarity

        Raises:
            ValueError: If query_embedding is invalid
            RuntimeError: If search fails
        """
        if len(query_embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"Query embedding must be {EMBEDDING_DIM}-dim, got {len(query_embedding)}"
            )

        if not 0.0 <= min_score <= 1.0:
            raise ValueError(f"min_score must be 0.0-1.0, got {min_score}")

        try:
            # Search Qdrant for similar vectors
            response = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=limit,
                score_threshold=min_score
            )

            # Retrieve full memories from SQLite
            memories_with_scores = []
            for point in response.points:
                memory_id = point.payload["memory_id"]
                memory = self.get_memory_by_id(memory_id)
                if memory:
                    memories_with_scores.append((memory, point.score))

            return memories_with_scores

        except Exception as e:
            raise RuntimeError(f"Search failed: {e}") from e

    def close(self):
        """Close storage connections.

        Raises:
            RuntimeError: If cleanup fails
        """
        errors = []

        # Close thread-local SQLite connections
        try:
            if hasattr(self, '_thread_local') and hasattr(self._thread_local, 'connection'):
                self._thread_local.connection.close()
                delattr(self._thread_local, 'connection')
        except Exception as e:
            errors.append(f"SQLite close error: {e}")

        if errors:
            raise RuntimeError(f"Errors during close: {'; '.join(errors)}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
