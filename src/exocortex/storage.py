"""
Storage layer for exocortex memories using Qdrant + SQLite.

This module provides hybrid storage:
- Qdrant: Fast vector similarity search
- SQLite: Rich metadata queries

Both are embedded (no external servers required).
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
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
        except Exception:
            # Collection doesn't exist, create it
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )

        # Initialize SQLite
        sqlite_path = self.storage_path / "metadata.db"
        self.db = sqlite3.connect(str(sqlite_path), check_same_thread=False)
        self._init_db()

    def _init_db(self):
        """Initialize SQLite schema."""
        cursor = self.db.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.db.commit()

    def store_memory(self, memory: Memory):
        """Store a memory with its embedding.

        Args:
            memory: Memory instance to store

        Raises:
            ValueError: If memory has no embedding
        """
        if memory.embedding is None:
            raise ValueError("Memory must have an embedding to be stored")

        # Store vector in Qdrant
        point = PointStruct(
            id=hash(memory.memory_id),  # Convert string ID to int for Qdrant
            vector=memory.embedding,
            payload={"memory_id": memory.memory_id}
        )
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

        # Store metadata in SQLite (using proper JSON serialization)
        cursor = self.db.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO memories (memory_id, data) VALUES (?, ?)",
            (memory.memory_id, json.dumps(memory.to_dict()))
        )
        self.db.commit()

    def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID.

        Args:
            memory_id: Memory ID to retrieve

        Returns:
            Memory instance if found, None otherwise
        """
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT data FROM memories WHERE memory_id = ?",
            (memory_id,)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        # Deserialize memory from JSON
        data = json.loads(row[0])
        return Memory.from_dict(data)

    def close(self):
        """Close storage connections."""
        self.db.close()
        # Qdrant client doesn't need explicit closing in embedded mode
