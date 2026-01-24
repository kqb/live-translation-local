"""Backend implementations for Whisper transcription."""

from .base import WhisperBackend, TranscriptionResult
from .detector import detect_best_backend, get_backend

__all__ = [
    "WhisperBackend",
    "TranscriptionResult",
    "detect_best_backend",
    "get_backend",
]
