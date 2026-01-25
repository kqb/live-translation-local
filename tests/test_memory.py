from datetime import datetime
from src.exocortex.memory import Memory, MemoryMetadata

def test_memory_metadata_creation():
    """Test creating MemoryMetadata with required fields."""
    metadata = MemoryMetadata(
        timestamp=datetime.now(),
        source_type="audio",
        source_id="session_123"
    )
    assert metadata.source_type == "audio"
    assert metadata.source_id == "session_123"
    assert metadata.timestamp is not None

def test_memory_creation():
    """Test creating Memory with minimal fields."""
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
    assert memory.metadata.source_type == "audio"

def test_memory_serialization():
    """Test Memory can be converted to/from dict."""
    metadata = MemoryMetadata(
        timestamp=datetime.fromisoformat("2026-01-25T10:00:00"),
        source_type="audio",
        source_id="session_123"
    )
    memory = Memory(
        memory_id="mem_001",
        metadata=metadata,
        text="Hello world",
        language="en",
        importance=0.7
    )

    # Serialize
    data = memory.to_dict()
    assert data["memory_id"] == "mem_001"
    assert data["text"] == "Hello world"
    assert data["metadata"]["source_type"] == "audio"

    # Deserialize
    restored = Memory.from_dict(data)
    assert restored.memory_id == memory.memory_id
    assert restored.text == memory.text
    assert restored.metadata.source_type == memory.metadata.source_type

def test_memory_id_generation():
    """Test Memory can generate unique IDs."""
    id1 = Memory.generate_id()
    id2 = Memory.generate_id()

    assert id1.startswith("mem_")
    assert id2.startswith("mem_")
    assert id1 != id2  # Should be unique
    assert len(id1) == 16  # "mem_" + 12 hex chars
