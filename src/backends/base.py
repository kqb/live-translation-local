"""Base protocol for Whisper backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass
class TranscriptionResult:
    """Result of a transcription."""

    text: str  # Transcribed text
    language: str  # Detected or specified language
    confidence: float  # Language detection probability
    segments: list[dict]  # Detailed segments with timestamps


class WhisperBackend(Protocol):
    """Protocol defining the interface for Whisper backends."""

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (float32, mono, 16kHz).
            language: Optional language hint.

        Returns:
            TranscriptionResult with text and metadata.
        """
        ...

    def detect_language(self, audio: np.ndarray) -> tuple[str, float]:
        """Detect the language of audio.

        Args:
            audio: Audio data as numpy array.

        Returns:
            Tuple of (language_code, probability).
        """
        ...

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        ...

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        ...
