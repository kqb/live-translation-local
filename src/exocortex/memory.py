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


@dataclass
class MemoryMetadata:
    """Metadata about when/where/how a memory was captured."""

    timestamp: datetime
    source_type: str  # "audio", "screen", "image", "text", etc.
    source_id: str    # Session ID, file path, URL, etc.

    # Optional context
    location: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "source_type": self.source_type,
            "source_id": self.source_id,
            "location": self.location,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryMetadata":
        """Restore from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source_type=data["source_type"],
            source_id=data["source_id"],
            location=data.get("location"),
            tags=data.get("tags", []),
        )


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
    speaker_name: Optional[str] = None   # "Alice", "Bob", etc.

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
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
        """Restore from dictionary."""
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

    @staticmethod
    def generate_id() -> str:
        """Generate a unique memory ID."""
        return f"mem_{uuid4().hex[:12]}"
