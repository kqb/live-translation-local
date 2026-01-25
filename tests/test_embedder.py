import pytest
from src.exocortex.embedder import MemoryEmbedder, EMBEDDING_DIM

@pytest.fixture
def embedder():
    """Create a MemoryEmbedder instance."""
    return MemoryEmbedder()

def test_embedder_initialization(embedder):
    """Test that embedder can be initialized."""
    assert embedder is not None
    assert embedder.model is not None

def test_embed_single_text(embedder):
    """Test embedding a single text string."""
    text = "Hello world"
    embedding = embedder.embed(text)

    assert embedding is not None
    assert len(embedding) == EMBEDDING_DIM  # 384 dimensions
    assert all(isinstance(x, float) for x in embedding)

def test_embed_batch(embedder):
    """Test embedding multiple texts at once."""
    texts = ["Hello world", "How are you?", "Good morning"]
    embeddings = embedder.embed_batch(texts)

    assert len(embeddings) == 3
    for embedding in embeddings:
        assert len(embedding) == EMBEDDING_DIM
        assert all(isinstance(x, float) for x in embedding)

def test_embed_empty_string(embedder):
    """Test that embedding empty string raises error."""
    with pytest.raises(ValueError, match="Cannot embed empty text"):
        embedder.embed("")

def test_embed_consistency(embedder):
    """Test that same text produces same embedding."""
    text = "Hello world"
    embedding1 = embedder.embed(text)
    embedding2 = embedder.embed(text)

    # Should be identical (or very close due to float precision)
    assert len(embedding1) == len(embedding2)
    for a, b in zip(embedding1, embedding2):
        assert abs(a - b) < 1e-6  # Allow tiny floating point differences
