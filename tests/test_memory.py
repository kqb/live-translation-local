import pytest
from datetime import datetime
from src.exocortex.memory import Memory, MemoryMetadata


def test_memory_metadata_creation():
    """Test creating MemoryMetadata with required fields."""
    metadata = MemoryMetadata(
        timestamp=datetime.now(), source_type="audio", source_id="session_123"
    )
    assert metadata.source_type == "audio"
    assert metadata.source_id == "session_123"
    assert metadata.timestamp is not None


def test_memory_creation():
    """Test creating Memory with minimal fields."""
    metadata = MemoryMetadata(
        timestamp=datetime.now(), source_type="audio", source_id="session_123"
    )
    memory = Memory(memory_id="mem_001", metadata=metadata, text="Hello world", language="en")
    assert memory.memory_id == "mem_001"
    assert memory.text == "Hello world"
    assert memory.language == "en"
    assert memory.metadata.source_type == "audio"


def test_memory_serialization():
    """Test Memory can be converted to/from dict."""
    metadata = MemoryMetadata(
        timestamp=datetime.fromisoformat("2026-01-25T10:00:00"),
        source_type="audio",
        source_id="session_123",
    )
    memory = Memory(
        memory_id="mem_001", metadata=metadata, text="Hello world", language="en", importance=0.7
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


# Validation tests
def test_memory_metadata_requires_source_type():
    """Test that source_type cannot be empty."""
    with pytest.raises(ValueError, match="source_type cannot be empty"):
        MemoryMetadata(timestamp=datetime.now(), source_type="", source_id="session_123")


def test_memory_metadata_requires_source_id():
    """Test that source_id cannot be empty."""
    with pytest.raises(ValueError, match="source_id cannot be empty"):
        MemoryMetadata(timestamp=datetime.now(), source_type="audio", source_id="")


def test_memory_metadata_validates_tags_type():
    """Test that tags must be a list."""
    with pytest.raises(TypeError, match="tags must be a list"):
        MemoryMetadata(
            timestamp=datetime.now(),
            source_type="audio",
            source_id="session_123",
            tags="not-a-list",  # Invalid type
        )


def test_memory_requires_memory_id():
    """Test that memory_id cannot be empty."""
    metadata = MemoryMetadata(
        timestamp=datetime.now(), source_type="audio", source_id="session_123"
    )
    with pytest.raises(ValueError, match="memory_id cannot be empty"):
        Memory(memory_id="", metadata=metadata, text="Hello", language="en")


def test_memory_requires_content():
    """Test that memory needs text, audio, or image."""
    metadata = MemoryMetadata(
        timestamp=datetime.now(), source_type="audio", source_id="session_123"
    )
    with pytest.raises(ValueError, match="must have at least"):
        Memory(memory_id="mem_001", metadata=metadata, text="", language="en")  # Empty


def test_memory_allows_audio_only():
    """Test that memory can have audio without text."""
    metadata = MemoryMetadata(
        timestamp=datetime.now(), source_type="audio", source_id="session_123"
    )
    memory = Memory(
        memory_id="mem_001",
        metadata=metadata,
        text="",
        language="en",
        audio_path="/path/to/audio.wav",  # Has audio
    )
    assert memory.audio_path == "/path/to/audio.wav"


def test_memory_allows_image_only():
    """Test that memory can have image without text."""
    metadata = MemoryMetadata(
        timestamp=datetime.now(), source_type="image", source_id="session_123"
    )
    memory = Memory(
        memory_id="mem_001",
        metadata=metadata,
        text="",
        language="en",
        image_path="/path/to/image.png",  # Has image
    )
    assert memory.image_path == "/path/to/image.png"


def test_memory_validates_importance_range():
    """Test that importance must be 0.0-1.0."""
    metadata = MemoryMetadata(
        timestamp=datetime.now(), source_type="audio", source_id="session_123"
    )
    with pytest.raises(ValueError, match="importance must be 0.0-1.0"):
        Memory(
            memory_id="mem_001",
            metadata=metadata,
            text="Hello",
            language="en",
            importance=50.0,  # Invalid
        )


def test_memory_validates_importance_negative():
    """Test that importance cannot be negative."""
    metadata = MemoryMetadata(
        timestamp=datetime.now(), source_type="audio", source_id="session_123"
    )
    with pytest.raises(ValueError, match="importance must be 0.0-1.0"):
        Memory(
            memory_id="mem_001",
            metadata=metadata,
            text="Hello",
            language="en",
            importance=-0.5,  # Invalid
        )


def test_memory_validates_sentiment_range():
    """Test that sentiment must be -1.0 to 1.0."""
    metadata = MemoryMetadata(
        timestamp=datetime.now(), source_type="audio", source_id="session_123"
    )
    with pytest.raises(ValueError, match="sentiment must be -1.0 to 1.0"):
        Memory(
            memory_id="mem_001",
            metadata=metadata,
            text="Hello",
            language="en",
            sentiment=5.0,  # Invalid
        )


def test_memory_validates_sentiment_negative_range():
    """Test that sentiment cannot be below -1.0."""
    metadata = MemoryMetadata(
        timestamp=datetime.now(), source_type="audio", source_id="session_123"
    )
    with pytest.raises(ValueError, match="sentiment must be -1.0 to 1.0"):
        Memory(
            memory_id="mem_001",
            metadata=metadata,
            text="Hello",
            language="en",
            sentiment=-5.0,  # Invalid
        )


def test_memory_validates_embedding_dimension():
    """Test that embedding must be 384-dimensional."""
    metadata = MemoryMetadata(
        timestamp=datetime.now(), source_type="audio", source_id="session_123"
    )
    with pytest.raises(ValueError, match="Embedding must be 384-dim"):
        Memory(
            memory_id="mem_001",
            metadata=metadata,
            text="Hello",
            language="en",
            embedding=[1.0, 2.0],  # Wrong size
        )


def test_memory_accepts_valid_embedding():
    """Test that valid 384-dimensional embedding is accepted."""
    metadata = MemoryMetadata(
        timestamp=datetime.now(), source_type="audio", source_id="session_123"
    )
    embedding = [0.1] * 384  # Valid 384-dimensional embedding
    memory = Memory(
        memory_id="mem_001", metadata=metadata, text="Hello", language="en", embedding=embedding
    )
    assert len(memory.embedding) == 384


# Deserialization error handling tests
def test_from_dict_handles_missing_fields():
    """Test deserialization with missing required fields."""
    with pytest.raises(ValueError, match="Missing required field"):
        MemoryMetadata.from_dict({"timestamp": "2026-01-25T10:00:00"})


def test_from_dict_handles_missing_source_id():
    """Test deserialization with missing source_id."""
    with pytest.raises(ValueError, match="Missing required field"):
        MemoryMetadata.from_dict({"timestamp": "2026-01-25T10:00:00", "source_type": "audio"})


def test_from_dict_handles_invalid_timestamp():
    """Test deserialization with malformed timestamp."""
    with pytest.raises(ValueError, match="Invalid timestamp format"):
        MemoryMetadata.from_dict(
            {"timestamp": "not-a-date", "source_type": "audio", "source_id": "123"}
        )


def test_memory_from_dict_handles_missing_fields():
    """Test Memory deserialization with missing required fields."""
    with pytest.raises(ValueError, match="Missing required field"):
        Memory.from_dict(
            {
                "memory_id": "mem_001",
                "text": "Hello",
                # Missing metadata and language
            }
        )


def test_memory_from_dict_handles_invalid_metadata():
    """Test Memory deserialization with invalid metadata."""
    with pytest.raises(ValueError, match="Invalid timestamp format"):
        Memory.from_dict(
            {
                "memory_id": "mem_001",
                "metadata": {
                    "timestamp": "invalid-date",
                    "source_type": "audio",
                    "source_id": "123",
                },
                "text": "Hello",
                "language": "en",
            }
        )
