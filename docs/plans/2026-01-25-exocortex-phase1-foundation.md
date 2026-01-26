# Exocortex Phase 1: Foundation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build basic memory capture and semantic retrieval - capture audio memories from translation pipeline, store with embeddings in Qdrant, retrieve via CLI.

**Architecture:** Create `src/exocortex/` package with Memory data model, Qdrant indexer, and CLI interface. Integrate with existing translation pipeline to automatically capture audio memories. English-only indexing using sentence-transformers (all-MiniLM-L6-v2).

**Tech Stack:** Qdrant (embedded), sentence-transformers, SQLite, existing Whisper pipeline

---

## Task 1: Set Up Core Infrastructure

**Files:**
- Create: `src/exocortex/__init__.py`
- Create: `src/exocortex/memory.py`
- Modify: `pyproject.toml` (add dependencies)

**Step 1: Add dependencies to pyproject.toml**

```toml
dependencies = [
    # ... existing dependencies ...
    "qdrant-client>=1.7.0",
    "sentence-transformers>=2.2.0",
]
```

**Step 2: Create package structure**

```bash
mkdir -p src/exocortex
touch src/exocortex/__init__.py
```

**Step 3: Install new dependencies**

```bash
pip install -e .
```

Expected: Dependencies install successfully

**Step 4: Commit**

```bash
git add pyproject.toml src/exocortex/__init__.py
git commit -m "feat(exocortex): add package structure and dependencies

Add Qdrant and sentence-transformers for vector search.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Create Memory Data Model

**Files:**
- Create: `src/exocortex/memory.py`
- Create: `tests/test_memory.py`

**Step 1: Write failing test for Memory model**

Create `tests/test_memory.py`:

```python
"""Tests for memory data model."""
from datetime import datetime
import pytest
from src.exocortex.memory import Memory, MemoryMetadata


def test_memory_creation():
    """Test creating a memory with required fields."""
    metadata = MemoryMetadata(
        timestamp=datetime.now(),
        source_type="audio",
        source_id="session_123"
    )

    memory = Memory(
        memory_id="mem_001",
        metadata=metadata,
        text="Hello world",
        language="en"
    )

    assert memory.memory_id == "mem_001"
    assert memory.text == "Hello world"
    assert memory.language == "en"


def test_memory_to_dict():
    """Test memory serialization to dictionary."""
    metadata = MemoryMetadata(
        timestamp=datetime.now(),
        source_type="audio",
        source_id="session_123"
    )

    memory = Memory(
        memory_id="mem_001",
        metadata=metadata,
        text="Hello world",
        language="en"
    )

    data = memory.to_dict()

    assert data["memory_id"] == "mem_001"
    assert data["text"] == "Hello world"
    assert data["language"] == "en"
    assert "timestamp" in data["metadata"]
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_memory.py -v
```

Expected: FAIL with "No module named 'src.exocortex.memory'"

**Step 3: Write minimal Memory model implementation**

Create `src/exocortex/memory.py`:

```python
"""Core memory data models for exocortex."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
from uuid import uuid4


@dataclass
class MemoryMetadata:
    """Metadata for a memory."""

    timestamp: datetime
    source_type: str  # "audio", "screen", "photo", etc.
    source_id: str    # Session ID or capture device ID

    # Optional context
    location: Optional[tuple[float, float]] = None  # (lat, lon)
    activity: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> MemoryMetadata:
        """Create from dictionary."""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class Memory:
    """A single memory point in the exocortex.

    Represents one captured moment with text content, embeddings,
    and rich metadata for retrieval.
    """

    memory_id: str
    metadata: MemoryMetadata

    # Content
    text: str  # Transcription, OCR, or extracted text
    language: str  # ISO language code (en, ja, etc.)

    # Optional content
    audio_path: Optional[str] = None
    image_path: Optional[str] = None
    translation: Optional[str] = None

    # Semantic understanding (populated later)
    importance: float = 0.5  # 0-1 relevance score
    sentiment: Optional[float] = None  # -1 to 1
    summary: Optional[str] = None

    # Speaker info (if audio)
    speaker_label: Optional[str] = None  # SPEAKER_00, etc.
    speaker_name: Optional[str] = None   # Recognized name

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        data = {
            "memory_id": self.memory_id,
            "metadata": self.metadata.to_dict(),
            "text": self.text,
            "language": self.language,
            "audio_path": self.audio_path,
            "image_path": self.image_path,
            "translation": self.translation,
            "importance": self.importance,
            "sentiment": self.sentiment,
            "summary": self.summary,
            "speaker_label": self.speaker_label,
            "speaker_name": self.speaker_name,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Memory:
        """Create from dictionary."""
        data = data.copy()
        data["metadata"] = MemoryMetadata.from_dict(data["metadata"])
        return cls(**data)

    @staticmethod
    def generate_id() -> str:
        """Generate a unique memory ID."""
        return f"mem_{uuid4().hex[:12]}"
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_memory.py -v
```

Expected: 2 PASSED

**Step 5: Commit**

```bash
git add src/exocortex/memory.py tests/test_memory.py
git commit -m "feat(exocortex): add Memory data model

Add core Memory and MemoryMetadata classes with serialization.
English-only for Phase 1, multi-modal support structured in.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Create Qdrant Storage Layer

**Files:**
- Create: `src/exocortex/storage.py`
- Create: `tests/test_storage.py`
- Create: `data/memories/.gitkeep`

**Step 1: Create data directory**

```bash
mkdir -p data/memories
touch data/memories/.gitkeep
git add data/memories/.gitkeep
git commit -m "feat(exocortex): add memories data directory"
```

**Step 2: Write failing test for QdrantStorage**

Create `tests/test_storage.py`:

```python
"""Tests for Qdrant storage layer."""
import tempfile
from pathlib import Path
from datetime import datetime
import pytest

from src.exocortex.memory import Memory, MemoryMetadata
from src.exocortex.storage import QdrantStorage


@pytest.fixture
def temp_storage_path():
    """Create temporary storage path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_qdrant_storage_initialization(temp_storage_path):
    """Test QdrantStorage initializes without error."""
    storage = QdrantStorage(storage_path=temp_storage_path)
    assert storage is not None


def test_store_and_retrieve_memory(temp_storage_path):
    """Test storing a memory and retrieving it by ID."""
    storage = QdrantStorage(storage_path=temp_storage_path)

    metadata = MemoryMetadata(
        timestamp=datetime.now(),
        source_type="audio",
        source_id="test_session"
    )

    memory = Memory(
        memory_id=Memory.generate_id(),
        metadata=metadata,
        text="Test memory content",
        language="en"
    )

    # Store memory
    storage.store_memory(memory, embedding=[0.1] * 384)

    # Retrieve by ID
    retrieved = storage.get_memory_by_id(memory.memory_id)

    assert retrieved is not None
    assert retrieved.memory_id == memory.memory_id
    assert retrieved.text == "Test memory content"
```

**Step 3: Run test to verify it fails**

```bash
pytest tests/test_storage.py -v
```

Expected: FAIL with "No module named 'src.exocortex.storage'"

**Step 4: Write minimal QdrantStorage implementation**

Create `src/exocortex/storage.py`:

```python
"""Storage layer for memories using Qdrant vector database."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from .memory import Memory


class QdrantStorage:
    """Storage layer combining Qdrant (vectors) and SQLite (metadata)."""

    COLLECTION_NAME = "memories"
    VECTOR_DIM = 384  # all-MiniLM-L6-v2 dimension

    def __init__(self, storage_path: Path):
        """Initialize storage.

        Args:
            storage_path: Base directory for storage (e.g., data/memories/)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize Qdrant (embedded mode)
        qdrant_path = self.storage_path / "qdrant"
        self.qdrant = QdrantClient(path=str(qdrant_path))

        # Create collection if it doesn't exist
        collections = self.qdrant.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.COLLECTION_NAME not in collection_names:
            self.qdrant.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.VECTOR_DIM,
                    distance=Distance.COSINE
                )
            )

        # Initialize SQLite for metadata
        db_path = self.storage_path / "memory.db"
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_db()

    def _init_db(self):
        """Initialize SQLite schema."""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.db.commit()

    def store_memory(self, memory: Memory, embedding: List[float]):
        """Store a memory with its embedding.

        Args:
            memory: Memory object to store
            embedding: 384-dim vector embedding
        """
        # Validate embedding dimension
        if len(embedding) != self.VECTOR_DIM:
            raise ValueError(f"Embedding must be {self.VECTOR_DIM} dimensions")

        # Store in Qdrant (vector + minimal payload)
        point = PointStruct(
            id=hash(memory.memory_id) & 0x7FFFFFFF,  # Convert to positive int
            vector=embedding,
            payload={
                "memory_id": memory.memory_id,
                "text_preview": memory.text[:200],  # First 200 chars
                "language": memory.language,
                "timestamp": memory.metadata.timestamp.isoformat(),
            }
        )

        self.qdrant.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[point]
        )

        # Store full memory in SQLite
        memory_json = json.dumps(memory.to_dict())
        self.db.execute(
            "INSERT OR REPLACE INTO memories (memory_id, data) VALUES (?, ?)",
            (memory.memory_id, memory_json)
        )
        self.db.commit()

    def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by its ID.

        Args:
            memory_id: Unique memory identifier

        Returns:
            Memory object or None if not found
        """
        cursor = self.db.execute(
            "SELECT data FROM memories WHERE memory_id = ?",
            (memory_id,)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        memory_data = json.loads(row[0])
        return Memory.from_dict(memory_data)

    def close(self):
        """Close database connections."""
        self.db.close()
        # Qdrant client doesn't need explicit closing in embedded mode
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_storage.py -v
```

Expected: 2 PASSED

**Step 6: Commit**

```bash
git add src/exocortex/storage.py tests/test_storage.py
git commit -m "feat(exocortex): add Qdrant storage layer

Combine Qdrant (vectors) + SQLite (metadata) for memory storage.
Embedded mode for fully local operation.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Create Embedding Generator

**Files:**
- Create: `src/exocortex/embedder.py`
- Create: `tests/test_embedder.py`

**Step 1: Write failing test for MemoryEmbedder**

Create `tests/test_embedder.py`:

```python
"""Tests for embedding generation."""
import pytest
from src.exocortex.embedder import MemoryEmbedder


def test_embedder_initialization():
    """Test MemoryEmbedder initializes and downloads model."""
    embedder = MemoryEmbedder()
    assert embedder is not None
    assert embedder.model is not None


def test_generate_embedding():
    """Test generating embedding for text."""
    embedder = MemoryEmbedder()

    text = "This is a test sentence for embedding generation."
    embedding = embedder.embed(text)

    assert isinstance(embedding, list)
    assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension
    assert all(isinstance(x, float) for x in embedding)


def test_embedding_similarity():
    """Test that similar texts have similar embeddings."""
    embedder = MemoryEmbedder()

    text1 = "The cat sat on the mat"
    text2 = "A cat was sitting on a mat"
    text3 = "Python is a programming language"

    emb1 = embedder.embed(text1)
    emb2 = embedder.embed(text2)
    emb3 = embedder.embed(text3)

    # Cosine similarity
    def cosine_sim(a, b):
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_12 = cosine_sim(emb1, emb2)
    sim_13 = cosine_sim(emb1, emb3)

    # Similar sentences should be more similar than unrelated ones
    assert sim_12 > sim_13
    assert sim_12 > 0.7  # High similarity threshold
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_embedder.py -v
```

Expected: FAIL with "No module named 'src.exocortex.embedder'"

**Step 3: Write minimal MemoryEmbedder implementation**

Create `src/exocortex/embedder.py`:

```python
"""Embedding generation for memories using sentence-transformers."""
from __future__ import annotations

from typing import List
from sentence_transformers import SentenceTransformer


class MemoryEmbedder:
    """Generate embeddings for memory text using sentence-transformers.

    Uses all-MiniLM-L6-v2: lightweight (80MB), fast, good quality.
    384-dimensional embeddings optimized for semantic similarity.
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    def __init__(self):
        """Initialize embedder and load model.

        First run will download model (~80MB).
        Subsequent runs load from cache.
        """
        self.model = SentenceTransformer(self.MODEL_NAME)

    def embed(self, text: str) -> List[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed (transcription, OCR, etc.)

        Returns:
            384-dimensional embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (more efficient).

        Args:
            texts: List of texts to embed

        Returns:
            List of 384-dimensional embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_embedder.py -v
```

Expected: 3 PASSED (Note: First run will download model ~80MB)

**Step 5: Commit**

```bash
git add src/exocortex/embedder.py tests/test_embedder.py
git commit -m "feat(exocortex): add embedding generator

Use sentence-transformers/all-MiniLM-L6-v2 for semantic embeddings.
Lightweight (80MB) and optimized for local inference.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Create Memory Indexer

**Files:**
- Create: `src/exocortex/indexer.py`
- Create: `tests/test_indexer.py`

**Step 1: Write failing test for MemoryIndexer**

Create `tests/test_indexer.py`:

```python
"""Tests for memory indexing."""
import tempfile
from pathlib import Path
from datetime import datetime
import pytest

from src.exocortex.memory import Memory, MemoryMetadata
from src.exocortex.indexer import MemoryIndexer


@pytest.fixture
def temp_indexer():
    """Create temporary memory indexer for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        indexer = MemoryIndexer(storage_path=Path(tmpdir))
        yield indexer
        indexer.close()


def test_indexer_initialization(temp_indexer):
    """Test MemoryIndexer initializes successfully."""
    assert temp_indexer is not None


def test_index_memory(temp_indexer):
    """Test indexing a memory."""
    metadata = MemoryMetadata(
        timestamp=datetime.now(),
        source_type="audio",
        source_id="test_session"
    )

    memory = Memory(
        memory_id=Memory.generate_id(),
        metadata=metadata,
        text="This is a test memory about artificial intelligence",
        language="en"
    )

    # Index memory (should generate embedding automatically)
    temp_indexer.index_memory(memory)

    # Retrieve to verify storage
    retrieved = temp_indexer.get_memory(memory.memory_id)
    assert retrieved is not None
    assert retrieved.text == memory.text


def test_index_only_english(temp_indexer):
    """Test that only English memories are indexed."""
    # English memory - should be indexed
    en_memory = Memory(
        memory_id=Memory.generate_id(),
        metadata=MemoryMetadata(
            timestamp=datetime.now(),
            source_type="audio",
            source_id="test"
        ),
        text="English text",
        language="en"
    )

    # Japanese memory - should be skipped
    ja_memory = Memory(
        memory_id=Memory.generate_id(),
        metadata=MemoryMetadata(
            timestamp=datetime.now(),
            source_type="audio",
            source_id="test"
        ),
        text="日本語のテキスト",
        language="ja"
    )

    temp_indexer.index_memory(en_memory)
    temp_indexer.index_memory(ja_memory)

    # English should be retrievable
    assert temp_indexer.get_memory(en_memory.memory_id) is not None

    # Japanese should NOT be indexed (Phase 1 constraint)
    assert temp_indexer.get_memory(ja_memory.memory_id) is None
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_indexer.py -v
```

Expected: FAIL with "No module named 'src.exocortex.indexer'"

**Step 3: Write minimal MemoryIndexer implementation**

Create `src/exocortex/indexer.py`:

```python
"""Memory indexing - combines storage and embedding generation."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .memory import Memory
from .storage import QdrantStorage
from .embedder import MemoryEmbedder


class MemoryIndexer:
    """High-level interface for indexing and retrieving memories.

    Combines:
    - Embedding generation (MemoryEmbedder)
    - Vector + metadata storage (QdrantStorage)
    - English-only filtering (Phase 1 constraint)
    """

    def __init__(self, storage_path: Path):
        """Initialize indexer.

        Args:
            storage_path: Base directory for storage
        """
        self.storage = QdrantStorage(storage_path)
        self.embedder = MemoryEmbedder()

    def index_memory(self, memory: Memory):
        """Index a memory (generate embedding and store).

        Phase 1: Only English memories are indexed.

        Args:
            memory: Memory to index
        """
        # Phase 1 constraint: English only
        if memory.language != "en":
            return  # Skip non-English memories

        # Generate embedding
        embedding = self.embedder.embed(memory.text)

        # Store memory with embedding
        self.storage.store_memory(memory, embedding)

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID.

        Args:
            memory_id: Unique memory identifier

        Returns:
            Memory object or None
        """
        return self.storage.get_memory_by_id(memory_id)

    def close(self):
        """Close storage connections."""
        self.storage.close()
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_indexer.py -v
```

Expected: 3 PASSED

**Step 5: Commit**

```bash
git add src/exocortex/indexer.py tests/test_indexer.py
git commit -m "feat(exocortex): add memory indexer

High-level interface combining embedding + storage.
English-only filtering for Phase 1.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Add Semantic Search

**Files:**
- Modify: `src/exocortex/indexer.py`
- Modify: `tests/test_indexer.py`

**Step 1: Write failing test for semantic search**

Add to `tests/test_indexer.py`:

```python
def test_search_memories(temp_indexer):
    """Test semantic search for memories."""
    # Index several memories
    memories = [
        Memory(
            memory_id=Memory.generate_id(),
            metadata=MemoryMetadata(
                timestamp=datetime.now(),
                source_type="audio",
                source_id="session1"
            ),
            text="I love machine learning and artificial intelligence",
            language="en"
        ),
        Memory(
            memory_id=Memory.generate_id(),
            metadata=MemoryMetadata(
                timestamp=datetime.now(),
                source_type="audio",
                source_id="session2"
            ),
            text="The weather is nice today",
            language="en"
        ),
        Memory(
            memory_id=Memory.generate_id(),
            metadata=MemoryMetadata(
                timestamp=datetime.now(),
                source_type="audio",
                source_id="session3"
            ),
            text="Deep learning is a subset of machine learning",
            language="en"
        ),
    ]

    for memory in memories:
        temp_indexer.index_memory(memory)

    # Search for AI-related content
    results = temp_indexer.search("artificial intelligence research", limit=2)

    assert len(results) == 2
    # First result should be the AI memory
    assert "artificial intelligence" in results[0].memory.text.lower()
    # Should have similarity score
    assert 0.0 <= results[0].score <= 1.0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_indexer.py::test_search_memories -v
```

Expected: FAIL with "MemoryIndexer has no attribute 'search'"

**Step 3: Add SearchResult and search method**

Modify `src/exocortex/indexer.py`:

```python
"""Memory indexing - combines storage and embedding generation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from .memory import Memory
from .storage import QdrantStorage
from .embedder import MemoryEmbedder


@dataclass
class SearchResult:
    """A single search result with similarity score."""

    memory: Memory
    score: float  # Cosine similarity (0-1, higher = more similar)


class MemoryIndexer:
    """High-level interface for indexing and retrieving memories.

    Combines:
    - Embedding generation (MemoryEmbedder)
    - Vector + metadata storage (QdrantStorage)
    - Semantic search
    - English-only filtering (Phase 1 constraint)
    """

    def __init__(self, storage_path: Path):
        """Initialize indexer.

        Args:
            storage_path: Base directory for storage
        """
        self.storage = QdrantStorage(storage_path)
        self.embedder = MemoryEmbedder()

    def index_memory(self, memory: Memory):
        """Index a memory (generate embedding and store).

        Phase 1: Only English memories are indexed.

        Args:
            memory: Memory to index
        """
        # Phase 1 constraint: English only
        if memory.language != "en":
            return  # Skip non-English memories

        # Generate embedding
        embedding = self.embedder.embed(memory.text)

        # Store memory with embedding
        self.storage.store_memory(memory, embedding)

    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Semantic search for memories.

        Args:
            query: Natural language search query
            limit: Maximum number of results

        Returns:
            List of SearchResult ordered by similarity (highest first)
        """
        # Generate query embedding
        query_embedding = self.embedder.embed(query)

        # Search Qdrant
        search_results = self.storage.qdrant.search(
            collection_name=self.storage.COLLECTION_NAME,
            query_vector=query_embedding,
            limit=limit
        )

        # Convert to SearchResult objects
        results = []
        for hit in search_results:
            memory_id = hit.payload["memory_id"]
            memory = self.get_memory(memory_id)

            if memory:
                results.append(SearchResult(
                    memory=memory,
                    score=hit.score
                ))

        return results

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID.

        Args:
            memory_id: Unique memory identifier

        Returns:
            Memory object or None
        """
        return self.storage.get_memory_by_id(memory_id)

    def close(self):
        """Close storage connections."""
        self.storage.close()
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_indexer.py -v
```

Expected: 4 PASSED

**Step 5: Commit**

```bash
git add src/exocortex/indexer.py tests/test_indexer.py
git commit -m "feat(exocortex): add semantic search

Query memories using natural language.
Returns results ordered by cosine similarity.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Create CLI Interface

**Files:**
- Create: `src/exocortex/cli.py`
- Modify: `pyproject.toml` (add console script)

**Step 1: Add CLI entry point to pyproject.toml**

Modify `pyproject.toml`:

```toml
[project.scripts]
live-translator = "src.cli:cli"
exo = "src.exocortex.cli:cli"  # Add this line
```

**Step 2: Create basic CLI**

Create `src/exocortex/cli.py`:

```python
"""CLI interface for exocortex memory system."""
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table

from .indexer import MemoryIndexer


console = Console()


@click.group()
def cli():
    """Exocortex - Personal Memory Externalization System."""
    pass


@cli.command()
@click.argument("query")
@click.option("--limit", default=5, help="Maximum number of results")
@click.option("--storage", default="./data/memories", help="Storage directory path")
def recall(query: str, limit: int, storage: str):
    """Search memories using natural language.

    Example:
        exo recall "artificial intelligence"
        exo recall "what did I say about the project?" --limit 10
    """
    storage_path = Path(storage)

    if not storage_path.exists():
        console.print("[red]No memories found. Start live-translator to capture memories.[/red]")
        return

    with console.status("[dim]Searching memories...[/dim]"):
        indexer = MemoryIndexer(storage_path)
        results = indexer.search(query, limit=limit)
        indexer.close()

    if not results:
        console.print(f"[yellow]No memories found matching: {query}[/yellow]")
        return

    # Display results in table
    table = Table(title=f"Search Results: {query}")
    table.add_column("Score", style="cyan", width=8)
    table.add_column("Text", style="white")
    table.add_column("Date", style="dim", width=20)

    for result in results:
        score = f"{result.score:.3f}"
        text = result.memory.text[:100] + "..." if len(result.memory.text) > 100 else result.memory.text
        timestamp = result.memory.metadata.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        table.add_row(score, text, timestamp)

    console.print(table)
    console.print(f"\n[dim]Found {len(results)} memories[/dim]")


@cli.command()
@click.option("--storage", default="./data/memories", help="Storage directory path")
def stats(storage: str):
    """Show exocortex statistics."""
    storage_path = Path(storage)

    if not storage_path.exists():
        console.print("[red]No memories found.[/red]")
        return

    indexer = MemoryIndexer(storage_path)

    # Count memories in SQLite
    cursor = indexer.storage.db.execute("SELECT COUNT(*) FROM memories")
    count = cursor.fetchone()[0]

    indexer.close()

    console.print(f"[green]Total Memories:[/green] {count}")
    console.print(f"[dim]Storage:[/dim] {storage_path}")


if __name__ == "__main__":
    cli()
```

**Step 3: Install and test CLI**

```bash
pip install -e .
exo --help
```

Expected: CLI help text displays

**Step 4: Commit**

```bash
git add src/exocortex/cli.py pyproject.toml
git commit -m "feat(exocortex): add CLI interface

Add 'exo recall' for semantic search and 'exo stats' for statistics.
Rich terminal output for readable results.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Integrate with Translation Pipeline

**Files:**
- Modify: `src/pipeline.py`
- Modify: `config.yaml`

**Step 1: Add exocortex config to config.yaml**

Add to `config.yaml`:

```yaml
exocortex:
  enabled: false      # Enable memory capture
  storage_path: "./data/memories"
  auto_index: true    # Automatically index transcriptions
```

**Step 2: Modify pipeline to capture memories**

Modify `src/pipeline.py`:

```python
# Add imports at top
from pathlib import Path
from datetime import datetime

# In PipelineConfig dataclass, add:
@dataclass
class ExocortexConfig:
    enabled: bool = False
    storage_path: str = "./data/memories"
    auto_index: bool = True

@dataclass
class PipelineConfig:
    # ... existing fields ...
    exocortex: ExocortexConfig = field(default_factory=ExocortexConfig)

# In load_config function, add exocortex loading:
def load_config(path: str = "config.yaml") -> PipelineConfig:
    # ... existing code ...

    # Add after g2 config loading
    exocortex_data = data.get("exocortex", {})

    return PipelineConfig(
        # ... existing fields ...
        exocortex=ExocortexConfig(
            enabled=exocortex_data.get("enabled", False),
            storage_path=exocortex_data.get("storage_path", "./data/memories"),
            auto_index=exocortex_data.get("auto_index", True),
        ),
    )

# In Pipeline.__init__, add:
class Pipeline:
    def __init__(self, config: PipelineConfig):
        # ... existing initialization ...

        # Initialize exocortex if enabled
        self.memory_indexer = None
        if config.exocortex.enabled:
            from src.exocortex.indexer import MemoryIndexer
            from src.exocortex.memory import Memory, MemoryMetadata

            self.memory_indexer = MemoryIndexer(
                storage_path=Path(config.exocortex.storage_path)
            )
            console.print("[dim]→ Exocortex memory capture enabled[/dim]")

# In _process_audio_chunk, after successful transcription, add:
def _process_audio_chunk(self, audio: np.ndarray):
    # ... existing transcription code ...

    # After getting transcription result:
    if self.memory_indexer and transcription.text:
        # Create memory from transcription
        from src.exocortex.memory import Memory, MemoryMetadata

        metadata = MemoryMetadata(
            timestamp=datetime.now(),
            source_type="audio",
            source_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        memory = Memory(
            memory_id=Memory.generate_id(),
            metadata=metadata,
            text=transcription.text,
            language=transcription.language,
            speaker_label=self._current_speaker,
            speaker_name=self._current_speaker_name,
        )

        # Index in background (don't block audio processing)
        try:
            self.memory_indexer.index_memory(memory)
        except Exception as e:
            console.print(f"[yellow]⚠ Memory indexing failed: {e}[/yellow]")
```

**Step 3: Test integration**

```bash
# 1. Enable exocortex in config.yaml
# 2. Run live-translator
# 3. Speak some English
# 4. Check memories were captured:
exo stats
exo recall "test"
```

Expected: Memories captured and searchable

**Step 4: Commit**

```bash
git add src/pipeline.py config.yaml
git commit -m "feat(exocortex): integrate with translation pipeline

Automatically capture English transcriptions as memories.
Configurable via config.yaml exocortex section.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Add Documentation

**Files:**
- Create: `docs/exocortex-quickstart.md`

**Step 1: Write quickstart guide**

Create `docs/exocortex-quickstart.md`:

```markdown
# Exocortex Quick Start Guide

## What is Exocortex?

Your personal memory externalization system. Captures, indexes, and makes searchable everything you say (in English).

**Current Features (Phase 1):**
- Automatic capture of English transcriptions
- Semantic search (find memories by meaning, not keywords)
- Local-only operation (no cloud, full privacy)

## Setup

### 1. Enable Memory Capture

Edit `config.yaml`:

```yaml
exocortex:
  enabled: true
  storage_path: "./data/memories"
  auto_index: true
```

### 2. Run Live Translator

```bash
live-translator run
```

Speak in English. Your transcriptions are automatically captured as memories.

### 3. Search Your Memories

```bash
# Search by topic
exo recall "artificial intelligence"

# Natural language queries
exo recall "what did I say about the project?"

# Get more results
exo recall "machine learning" --limit 10

# Check statistics
exo stats
```

## How It Works

**Capture:**
1. You speak → Whisper transcribes → English text extracted
2. Text embedded using sentence-transformers (semantic vector)
3. Stored in Qdrant (vector DB) + SQLite (metadata)

**Search:**
1. You query in natural language
2. Query embedded to same vector space
3. Qdrant finds semantically similar memories
4. Results ranked by similarity

## Storage Location

- **Vectors**: `./data/memories/qdrant/`
- **Metadata**: `./data/memories/memory.db`
- **Total Size**: ~100KB per hour of English speech

## Privacy

- **100% local** - no cloud, no API calls (after model download)
- **English only** - non-English text not indexed (Phase 1)
- **Your data** - stored on your machine, full control

## Limitations (Phase 1)

- English-only indexing
- Audio memories only (no screen/photos yet)
- Basic semantic search (no filters, time ranges, speakers)
- No graph relationships or consolidation

See design doc for future phases: `docs/plans/2026-01-25-exocortex-foundation-design.md`

## Troubleshooting

**"No memories found"**
- Check `exocortex.enabled: true` in config.yaml
- Make sure you're speaking in English
- Verify storage path exists: `ls data/memories/`

**Search returns no results**
- Wait a few seconds after speaking (indexing happens async)
- Try broader queries ("AI" vs "specific AI technique")
- Check memories exist: `exo stats`

**Model download fails**
- First run downloads sentence-transformers model (~80MB)
- Requires internet connection (one-time)
- Cached in `~/.cache/torch/sentence_transformers/`
```

**Step 2: Commit**

```bash
git add docs/exocortex-quickstart.md
git commit -m "docs(exocortex): add quickstart guide

User-facing guide for Phase 1 functionality.
Covers setup, usage, and troubleshooting.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Final Testing & Polish

**Step 1: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests pass

**Step 2: Test end-to-end workflow**

```bash
# 1. Enable exocortex in config
# 2. Run live-translator, speak English for 1 minute
# 3. Stop translator
# 4. Search memories:
exo recall "test"
exo stats
```

Expected: Memories captured and searchable

**Step 3: Update main README**

Add to `README.md`:

```markdown
## Exocortex (Phase 1 - Beta)

Personal memory externalization system. Automatically captures and makes searchable everything you say in English.

**Quick Start:**
1. Enable in `config.yaml`: `exocortex.enabled: true`
2. Run: `live-translator run`
3. Search: `exo recall "topic"`

See [Exocortex Quick Start](docs/exocortex-quickstart.md) for details.
```

**Step 4: Final commit**

```bash
git add README.md
git commit -m "docs: add exocortex to main README

Announce Phase 1 beta availability.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 1 Complete!

**Deliverables:**
- ✅ Memory data model (Memory, MemoryMetadata)
- ✅ Qdrant + SQLite storage layer
- ✅ Embedding generation (sentence-transformers)
- ✅ Memory indexer (English-only)
- ✅ Semantic search
- ✅ CLI interface (`exo recall`, `exo stats`)
- ✅ Translation pipeline integration
- ✅ Documentation

**Total Tasks:** 10
**Estimated Time:** 4-6 hours

**Next Phase:** Phase 2 - Graph & Entity Linking (see design doc)

---

**Testing Checklist:**
- [ ] All unit tests pass
- [ ] CLI commands work
- [ ] Memories captured during live translation
- [ ] Search returns relevant results
- [ ] Storage uses <1MB per hour of speech
- [ ] English-only constraint verified
