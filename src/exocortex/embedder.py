"""
Embedding generator for exocortex memories.

Uses sentence-transformers to generate semantic embeddings for text.
Model: all-MiniLM-L6-v2 (384-dimensional, English-optimized).
"""

import logging
from sentence_transformers import SentenceTransformer

# Model configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_TEXT_LENGTH = 20000  # Maximum characters per text (~40 pages of text)
MAX_BATCH_SIZE = 100  # Maximum texts per batch

logger = logging.getLogger(__name__)


class MemoryEmbedder:
    """Generate semantic embeddings for memory text."""

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        """Initialize embedder with specified model.

        Args:
            model_name: SentenceTransformer model name
            device: Device to use ("cpu", "cuda", "mps")

        Raises:
            RuntimeError: If model loading fails
        """
        try:
            logger.info(f"Loading embedding model: {model_name} on device: {device}")
            self.model = SentenceTransformer(model_name, device=device)
            self.device = device
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}'. "
                f"First run downloads ~150MB. Check network connection. Error: {e}"
            ) from e

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            384-dimensional embedding vector

        Raises:
            ValueError: If text is empty or too long
            RuntimeError: If encoding fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text too long: {len(text)} chars (max {MAX_TEXT_LENGTH})")

        try:
            logger.debug(f"Encoding text of length {len(text)}")
            embedding = self.model.encode(text, convert_to_numpy=True)

            # Verify dimension matches expected
            if len(embedding) != EMBEDDING_DIM:
                raise RuntimeError(
                    f"Model produced {len(embedding)}-dim embedding, expected {EMBEDDING_DIM}"
                )

            return embedding.tolist()
        except Exception as e:
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            raise RuntimeError(f"Failed to encode text: {e}") from e

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed (max 100 per batch)

        Returns:
            List of 384-dimensional embedding vectors

        Raises:
            ValueError: If any text is invalid or batch too large
            RuntimeError: If encoding fails
        """
        if len(texts) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(texts)} exceeds maximum {MAX_BATCH_SIZE}. "
                f"Process in smaller batches."
            )

        # Validate all texts
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Cannot embed empty text at index {i}")
            if len(text) > MAX_TEXT_LENGTH:
                raise ValueError(f"Text at index {i} too long: {len(text)} chars")

        try:
            logger.debug(f"Encoding batch of {len(texts)} texts")
            embeddings = self.model.encode(texts, convert_to_numpy=True)

            # Verify dimensions
            for i, emb in enumerate(embeddings):
                if len(emb) != EMBEDDING_DIM:
                    raise RuntimeError(
                        f"Embedding {i} has wrong dimension: {len(emb)} vs {EMBEDDING_DIM}"
                    )

            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            raise RuntimeError(f"Failed to encode batch: {e}") from e

    def close(self):
        """Release model resources.

        Cleans up the loaded model to free GPU/CPU memory.
        Safe to call multiple times.
        """
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
