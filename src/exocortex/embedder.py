"""
Embedding generator for exocortex memories.

Uses sentence-transformers to generate semantic embeddings for text.
Model: all-MiniLM-L6-v2 (384-dimensional, English-optimized).
"""

from typing import List
from sentence_transformers import SentenceTransformer

# Model configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class MemoryEmbedder:
    """Generate semantic embeddings for memory text."""

    def __init__(self, model_name: str = MODEL_NAME):
        """Initialize embedder with specified model.

        Args:
            model_name: SentenceTransformer model name
        """
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            384-dimensional embedding vector

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of 384-dimensional embedding vectors

        Raises:
            ValueError: If any text is empty
        """
        # Validate all texts
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Cannot embed empty text at index {i}")

        # Generate embeddings in batch
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]
