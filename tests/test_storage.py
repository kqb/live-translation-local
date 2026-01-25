import pytest
from pathlib import Path
from datetime import datetime
from src.exocortex.memory import Memory, MemoryMetadata
from src.exocortex.storage import QdrantStorage

@pytest.fixture
def temp_storage(tmp_path):
    """Create a temporary QdrantStorage instance."""
    storage_path = tmp_path / "test_storage"
    storage = QdrantStorage(storage_path=storage_path, collection_name="test_memories")
    yield storage
    storage.close()

def test_storage_initialization(temp_storage):
    """Test that storage can be initialized."""
    assert temp_storage is not None
    assert temp_storage.collection_name == "test_memories"

def test_store_memory(temp_storage):
    """Test storing a memory with embedding."""
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
        embedding=[0.1] * 384  # 384-dim embedding
    )

    # Store should succeed
    temp_storage.store_memory(memory)

    # Should be able to retrieve it
    retrieved = temp_storage.get_memory_by_id(memory.memory_id)
    assert retrieved is not None
    assert retrieved.memory_id == memory.memory_id
    assert retrieved.text == memory.text

def test_get_nonexistent_memory(temp_storage):
    """Test getting a memory that doesn't exist."""
    result = temp_storage.get_memory_by_id("nonexistent_id")
    assert result is None

def test_store_memory_without_embedding(temp_storage):
    """Test that storing memory without embedding raises error."""
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

    with pytest.raises(ValueError, match="Memory must have an embedding"):
        temp_storage.store_memory(memory)

def test_memory_id_to_int_deterministic(temp_storage):
    """Test that ID conversion is deterministic."""
    memory_id = "mem_abc123def456"
    int1 = temp_storage._memory_id_to_int(memory_id)
    int2 = temp_storage._memory_id_to_int(memory_id)
    assert int1 == int2

def test_store_memory_validates_embedding_dimension(temp_storage):
    """Test that storing memory validates embedding dimension.

    Note: Memory class already validates dimension in __post_init__,
    but storage layer provides defense-in-depth validation.
    """
    metadata = MemoryMetadata(
        timestamp=datetime.now(),
        source_type="audio",
        source_id="session_123"
    )

    # Memory class validates dimension in __post_init__, so we expect it there
    with pytest.raises(ValueError, match="Embedding must be 384-dim"):
        memory = Memory(
            memory_id=Memory.generate_id(),
            metadata=metadata,
            text="Hello",
            language="en",
            embedding=[0.1] * 100  # Wrong dimension!
        )

def test_storage_validates_embedding_dimension_defense_in_depth(temp_storage):
    """Test storage layer validates dimension even if Memory validation bypassed.

    This tests defense-in-depth: storage validates even if somehow
    invalid data gets past Memory.__post_init__.
    """
    metadata = MemoryMetadata(
        timestamp=datetime.now(),
        source_type="audio",
        source_id="session_123"
    )
    # Create valid memory first
    memory = Memory(
        memory_id=Memory.generate_id(),
        metadata=metadata,
        text="Hello",
        language="en",
        embedding=[0.1] * 384
    )

    # Manually corrupt the embedding after construction to bypass Memory validation
    memory.embedding = [0.1] * 100

    # Storage layer should still catch this
    with pytest.raises(ValueError, match="Embedding must be 384-dim"):
        temp_storage.store_memory(memory)

def test_context_manager_support(tmp_path):
    """Test that QdrantStorage works as context manager."""
    storage_path = tmp_path / "ctx_storage"

    with QdrantStorage(storage_path=storage_path) as storage:
        metadata = MemoryMetadata(
            timestamp=datetime.now(),
            source_type="audio",
            source_id="session_123"
        )
        memory = Memory(
            memory_id=Memory.generate_id(),
            metadata=metadata,
            text="Hello",
            language="en",
            embedding=[0.1] * 384
        )
        storage.store_memory(memory)

    # Storage should be closed after context
