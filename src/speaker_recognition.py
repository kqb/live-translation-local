"""Speaker recognition module for persistent speaker identification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class SpeakerProfile:
    """A speaker profile with name and voice embeddings."""

    name: str  # Speaker name (e.g., "Alice", "Bob")
    embeddings: list[np.ndarray]  # List of voice embeddings for this speaker
    sample_count: int = 0  # Number of samples collected

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "embeddings": [emb.tolist() for emb in self.embeddings],
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SpeakerProfile:
        """Load from dictionary."""
        return cls(
            name=data["name"],
            embeddings=[np.array(emb) for emb in data["embeddings"]],
            sample_count=data["sample_count"],
        )


class SpeakerDatabase:
    """Database for storing and matching speaker profiles."""

    def __init__(self, db_path: Path):
        """Initialize speaker database.

        Args:
            db_path: Path to speaker database JSON file.
        """
        self.db_path = Path(db_path)
        self.speakers: dict[str, SpeakerProfile] = {}
        self._load()

    def _load(self) -> None:
        """Load speaker profiles from disk."""
        if not self.db_path.exists():
            return

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.speakers = {
                name: SpeakerProfile.from_dict(profile_data)
                for name, profile_data in data.items()
            }
        except Exception as e:
            print(f"Warning: Failed to load speaker database: {e}")

    def save(self) -> None:
        """Save speaker profiles to disk."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        data = {name: profile.to_dict() for name, profile in self.speakers.items()}

        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_speaker(self, name: str, embedding: np.ndarray) -> None:
        """Add or update a speaker profile.

        Args:
            name: Speaker name.
            embedding: Voice embedding vector.
        """
        if name in self.speakers:
            # Add to existing profile
            self.speakers[name].embeddings.append(embedding)
            self.speakers[name].sample_count += 1
        else:
            # Create new profile
            self.speakers[name] = SpeakerProfile(
                name=name,
                embeddings=[embedding],
                sample_count=1,
            )

        self.save()

    def match_speaker(
        self, embedding: np.ndarray, threshold: float = 0.75
    ) -> Optional[str]:
        """Match an embedding against known speakers.

        Args:
            embedding: Voice embedding to match.
            threshold: Similarity threshold (0.0-1.0), higher = stricter.

        Returns:
            Speaker name if match found, None otherwise.
        """
        if not self.speakers:
            return None

        best_match = None
        best_score = threshold

        for name, profile in self.speakers.items():
            # Calculate average similarity to all embeddings for this speaker
            similarities = [
                self._cosine_similarity(embedding, ref_emb)
                for ref_emb in profile.embeddings
            ]
            avg_similarity = np.mean(similarities)

            if avg_similarity > best_score:
                best_score = avg_similarity
                best_match = name

        return best_match

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Similarity score (0.0-1.0).
        """
        # Normalize vectors
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)

        # Cosine similarity
        return float(np.dot(a_norm, b_norm))

    def list_speakers(self) -> list[dict]:
        """List all registered speakers.

        Returns:
            List of speaker info dictionaries.
        """
        return [
            {
                "name": profile.name,
                "sample_count": profile.sample_count,
            }
            for profile in self.speakers.values()
        ]

    def remove_speaker(self, name: str) -> bool:
        """Remove a speaker from the database.

        Args:
            name: Speaker name to remove.

        Returns:
            True if removed, False if not found.
        """
        if name in self.speakers:
            del self.speakers[name]
            self.save()
            return True
        return False


class SpeakerRecognizer:
    """Recognizes speakers using embeddings and a speaker database."""

    def __init__(self, diarizer, db_path: Path, recognition_threshold: float = 0.75):
        """Initialize speaker recognizer.

        Args:
            diarizer: Diarizer instance for extracting embeddings.
            db_path: Path to speaker database file.
            recognition_threshold: Similarity threshold for matching (0.0-1.0).
        """
        self.diarizer = diarizer
        self.db = SpeakerDatabase(db_path)
        self.recognition_threshold = recognition_threshold

    def extract_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """Extract speaker embedding from audio.

        Args:
            audio: Audio data as numpy array.
            sample_rate: Sample rate of the audio.

        Returns:
            Speaker embedding vector or None if extraction fails.
        """
        try:
            # Use pyannote pipeline to extract embeddings
            import torch

            pipeline = self.diarizer._ensure_pipeline()

            # Prepare audio for pyannote
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            if audio.ndim > 1:
                audio = audio.flatten()

            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            audio_dict = {
                "waveform": audio_tensor,
                "sample_rate": sample_rate,
            }

            # Extract embedding using the segmentation model
            # pyannote's pipeline has an embedding model we can access
            embedding_model = pipeline.embedding
            with torch.no_grad():
                embedding = embedding_model(audio_dict)

            # Convert to numpy and average over time if needed
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()
                if embedding.ndim > 1:
                    embedding = embedding.mean(axis=0)

            return embedding

        except Exception as e:
            print(f"Failed to extract embedding: {e}")
            return None

    def enroll_speaker(self, name: str, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """Enroll a new speaker or add sample to existing speaker.

        Args:
            name: Speaker name.
            audio: Audio sample of the speaker.
            sample_rate: Sample rate of the audio.

        Returns:
            True if enrollment succeeded, False otherwise.
        """
        embedding = self.extract_embedding(audio, sample_rate)
        if embedding is None:
            return False

        self.db.add_speaker(name, embedding)
        return True

    def recognize_speaker(self, embedding: np.ndarray) -> Optional[str]:
        """Recognize a speaker from their embedding.

        Args:
            embedding: Speaker embedding to match.

        Returns:
            Speaker name if recognized, None otherwise.
        """
        return self.db.match_speaker(embedding, self.recognition_threshold)
