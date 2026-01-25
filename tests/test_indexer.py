import pytest
from pathlib import Path
from datetime import datetime
from src.exocortex.memory import Memory, MemoryMetadata
from src.exocortex.indexer import MemoryIndexer

@pytest.fixture
def temp_indexer(tmp_path):
    """Create a temporary MemoryIndexer instance."""
    storage_path = tmp_path / "test_indexer"
    indexer = MemoryIndexer(storage_path=storage_path)
    yield indexer
    indexer.close()

def test_indexer_initialization(temp_indexer):
    """Test that indexer can be initialized."""
    assert temp_indexer is not None
    assert temp_indexer.storage is not None
    assert temp_indexer.embedder is not None

def test_index_memory_with_embedding(temp_indexer):
    """Test indexing a memory that already has an embedding."""
    metadata = MemoryMetadata(
        timestamp=datetime.now(),
        source_type="audio",
        source_id="session_123"
    )
    memory = Memory(
        memory_id=Memory.generate_id(),
        metadata=metadata,
        text="Hello world",
        language="en",
        embedding=[0.1] * 384  # Pre-existing embedding
    )

    temp_indexer.index_memory(memory)

    # Should be able to retrieve it
    retrieved = temp_indexer.get_memory(memory.memory_id)
    assert retrieved is not None
    assert retrieved.text == "Hello world"

def test_index_memory_auto_embed(temp_indexer):
    """Test that indexer auto-generates embeddings."""
    metadata = MemoryMetadata(
        timestamp=datetime.now(),
        source_type="audio",
        source_id="session_123"
    )
    memory = Memory(
        memory_id=Memory.generate_id(),
        metadata=metadata,
        text="Hello world",
        language="en",
        embedding=None  # No embedding!
    )

    temp_indexer.index_memory(memory)

    # Should generate embedding and store
    retrieved = temp_indexer.get_memory(memory.memory_id)
    assert retrieved is not None
    assert retrieved.embedding is not None
    assert len(retrieved.embedding) == 384

def test_get_nonexistent_memory(temp_indexer):
    """Test getting a memory that doesn't exist."""
    result = temp_indexer.get_memory("nonexistent_id")
    assert result is None

def test_context_manager_support(tmp_path):
    """Test that indexer works as context manager."""
    storage_path = tmp_path / "ctx_indexer"

    with MemoryIndexer(storage_path=storage_path) as indexer:
        metadata = MemoryMetadata(
            timestamp=datetime.now(),
            source_type="audio",
            source_id="session_123"
        )
        memory = Memory(
            memory_id=Memory.generate_id(),
            metadata=metadata,
            text="Hello",
            language="en"
        )
        indexer.index_memory(memory)

    # Should be closed after context
