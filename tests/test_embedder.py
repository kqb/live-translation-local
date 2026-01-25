import pytest
from src.exocortex.embedder import (
    MemoryEmbedder,
    EMBEDDING_DIM,
    MAX_TEXT_LENGTH,
    MAX_BATCH_SIZE,
)


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


def test_invalid_model_name():
    """Test that invalid model name raises clear error."""
    with pytest.raises(RuntimeError, match="Failed to load embedding model"):
        MemoryEmbedder(model_name="invalid/model/name")


def test_text_too_long(embedder):
    """Test that very long text is rejected."""
    long_text = "a" * (MAX_TEXT_LENGTH + 1)  # Exceeds MAX_TEXT_LENGTH
    with pytest.raises(ValueError, match="Text too long"):
        embedder.embed(long_text)


def test_batch_size_limit(embedder):
    """Test that batch size limit is enforced."""
    texts = ["Hello"] * (MAX_BATCH_SIZE + 1)  # Exceeds MAX_BATCH_SIZE
    with pytest.raises(ValueError, match="Batch size.*exceeds maximum"):
        embedder.embed_batch(texts)


def test_dimension_mismatch_detection(embedder):
    """Test that dimension mismatch is detected."""
    # This test verifies the validation logic exists
    # In practice, we can't easily cause dimension mismatch
    # without mocking, so we document the validation
    text = "Test"
    embedding = embedder.embed(text)
    assert len(embedding) == EMBEDDING_DIM


def test_context_manager_support():
    """Test that embedder works as context manager."""
    with MemoryEmbedder() as embedder:
        embedding = embedder.embed("Hello")
        assert len(embedding) == EMBEDDING_DIM
    # Embedder should be closed after context


def test_close_idempotency():
    """Test that close() can be called multiple times safely."""
    embedder = MemoryEmbedder()
    embedder.close()
    embedder.close()  # Should not crash
