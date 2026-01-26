"""
Memory data models for the exocortex system.

This module defines the core data structures for representing memories:
- MemoryMetadata: When/where/how a memory was captured
- Memory: The memory itself with text, embeddings, and metadata
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import uuid4

# Embedding dimension for sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM = 384


@dataclass
class MemoryMetadata:
    """Metadata about when/where/how a memory was captured."""

    timestamp: datetime
    source_type: str  # "audio", "screen", "image", "text", etc.
    source_id: str  # Session ID, file path, URL, etc.

    # Optional context
    location: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate metadata fields after initialization.

        Raises:
            ValueError: If source_type or source_id is empty.
            TypeError: If tags is not a list.
        """
        if not self.source_type:
            raise ValueError("source_type cannot be empty")
        if not self.source_id:
            raise ValueError("source_id cannot be empty")
        if not isinstance(self.tags, list):
            raise TypeError("tags must be a list")

    def to_dict(self) -> dict:
        """Convert metadata to dictionary for storage.

        Returns:
            Dictionary containing all metadata fields with datetime serialized
            to ISO format string.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "source_type": self.source_type,
            "source_id": self.source_id,
            "location": self.location,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryMetadata":
        """Restore MemoryMetadata from dictionary.

        Args:
            data: Dictionary with keys matching MemoryMetadata fields.
                  Required: timestamp (ISO format string), source_type, source_id

        Returns:
            MemoryMetadata instance reconstructed from dictionary.

        Raises:
            ValueError: If required fields missing or timestamp invalid.
        """
        try:
            return cls(
                timestamp=datetime.fromisoformat(data["timestamp"]),
                source_type=data["source_type"],
                source_id=data["source_id"],
                location=data.get("location"),
                tags=data.get("tags", []),
            )
        except KeyError as e:
            raise ValueError(f"Missing required field in metadata: {e}") from e
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid timestamp format: {data.get('timestamp')}") from e


@dataclass
class Memory:
    """A captured memory with text, embeddings, and metadata."""

    memory_id: str
    metadata: MemoryMetadata
    text: str
    language: str

    # Optional fields
    audio_path: Optional[str] = None
    image_path: Optional[str] = None
    translation: Optional[str] = None
    embedding: Optional[list[float]] = None
    importance: float = 0.5  # 0.0 to 1.0
    sentiment: Optional[float] = None  # -1.0 to 1.0
    summary: Optional[str] = None

    # Speaker information (if from audio with diarization)
    speaker_label: Optional[str] = None  # "SPEAKER_00", etc.
    speaker_name: Optional[str] = None  # "Alice", "Bob", etc.

    def __post_init__(self):
        """Validate memory fields after initialization.

        Raises:
            ValueError: If memory_id is empty, no content provided, or numeric
                       fields are out of valid range.
        """
        if not self.memory_id:
            raise ValueError("memory_id cannot be empty")

        # At least one content type must be provided
        if not self.text and not self.audio_path and not self.image_path:
            raise ValueError("Memory must have at least text, audio, or image content")

        # Validate importance range
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError(f"importance must be 0.0-1.0, got {self.importance}")

        # Validate sentiment range if provided
        if self.sentiment is not None and not -1.0 <= self.sentiment <= 1.0:
            raise ValueError(f"sentiment must be -1.0 to 1.0, got {self.sentiment}")

        # Validate embedding dimension if provided
        if self.embedding is not None and len(self.embedding) != EMBEDDING_DIM:
            raise ValueError(f"Embedding must be {EMBEDDING_DIM}-dim, got {len(self.embedding)}")

    def to_dict(self) -> dict:
        """Convert memory to dictionary for storage.

        Returns:
            Dictionary containing all memory fields with nested metadata
            serialized via MemoryMetadata.to_dict().
        """
        return {
            "memory_id": self.memory_id,
            "metadata": self.metadata.to_dict(),
            "text": self.text,
            "language": self.language,
            "audio_path": self.audio_path,
            "image_path": self.image_path,
            "translation": self.translation,
            "embedding": self.embedding,
            "importance": self.importance,
            "sentiment": self.sentiment,
            "summary": self.summary,
            "speaker_label": self.speaker_label,
            "speaker_name": self.speaker_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        """Restore Memory from dictionary.

        Args:
            data: Dictionary with keys matching Memory fields.
                  Required: memory_id, metadata (dict), text, language

        Returns:
            Memory instance reconstructed from dictionary.

        Raises:
            ValueError: If required fields missing or metadata is invalid.
        """
        try:
            return cls(
                memory_id=data["memory_id"],
                metadata=MemoryMetadata.from_dict(data["metadata"]),
                text=data["text"],
                language=data["language"],
                audio_path=data.get("audio_path"),
                image_path=data.get("image_path"),
                translation=data.get("translation"),
                embedding=data.get("embedding"),
                importance=data.get("importance", 0.5),
                sentiment=data.get("sentiment"),
                summary=data.get("summary"),
                speaker_label=data.get("speaker_label"),
                speaker_name=data.get("speaker_name"),
            )
        except KeyError as e:
            raise ValueError(f"Missing required field in memory: {e}") from e

    @staticmethod
    def generate_id() -> str:
        """Generate a unique memory ID.

        Returns:
            String in format 'mem_<12 hex chars>' suitable for use as memory_id.
        """
        return f"mem_{uuid4().hex[:12]}"
