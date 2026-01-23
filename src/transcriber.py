"""Transcriber module using faster-whisper for speech-to-text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel


# Supported model sizes
WHISPER_MODELS = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v2",
    "large-v3",
]

# Language code mapping (ISO 639-1 to Whisper language names)
LANGUAGE_CODES = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "nl": "dutch",
    "pl": "polish",
    "ru": "russian",
    "ja": "japanese",
    "zh": "chinese",
    "ko": "korean",
    "ar": "arabic",
    "hi": "hindi",
    "tr": "turkish",
    "vi": "vietnamese",
    "th": "thai",
    "id": "indonesian",
    "ms": "malay",
    "tl": "tagalog",
    "uk": "ukrainian",
    "cs": "czech",
    "ro": "romanian",
    "hu": "hungarian",
    "el": "greek",
    "he": "hebrew",
    "sv": "swedish",
    "da": "danish",
    "fi": "finnish",
    "no": "norwegian",
}


@dataclass
class TranscriberConfig:
    """Configuration for the Whisper transcriber."""

    model: str = "base"  # Model size
    device: str = "auto"  # cpu, cuda, auto
    compute_type: str = "int8"  # int8, float16, float32
    language: Optional[str] = None  # None = auto-detect


@dataclass
class TranscriptionResult:
    """Result of a transcription."""

    text: str  # Transcribed text
    language: str  # Detected or specified language
    confidence: float  # Language detection probability
    segments: list[dict]  # Detailed segments with timestamps


class Transcriber:
    """Speech-to-text transcriber using faster-whisper."""

    def __init__(self, config: TranscriberConfig):
        """Initialize the transcriber.

        Args:
            config: Transcriber configuration.
        """
        self.config = config
        self._model: Optional[WhisperModel] = None

    def _get_device(self) -> str:
        """Determine the device to use."""
        if self.config.device != "auto":
            return self.config.device

        # Try to use CUDA if available
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass

        return "cpu"

    def _ensure_model(self) -> WhisperModel:
        """Ensure the model is loaded (lazy loading)."""
        if self._model is None:
            device = self._get_device()
            compute_type = self.config.compute_type

            # Adjust compute type based on device
            if device == "cpu" and compute_type == "float16":
                compute_type = "int8"  # float16 not well supported on CPU

            self._model = WhisperModel(
                self.config.model,
                device=device,
                compute_type=compute_type,
            )

        return self._model

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (float32, mono, 16kHz).
            language: Optional language hint (overrides config).

        Returns:
            TranscriptionResult with text and metadata.
        """
        model = self._ensure_model()

        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Ensure audio is 1D
        if audio.ndim > 1:
            audio = audio.flatten()

        # Determine language setting
        lang = language or self.config.language

        # Map simple code to Whisper language name if needed
        if lang and lang in LANGUAGE_CODES:
            lang = LANGUAGE_CODES[lang]

        # Transcribe with VAD filter for better segmentation
        segments, info = model.transcribe(
            audio,
            language=lang,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )

        # Collect segments
        segment_list = []
        text_parts = []

        for segment in segments:
            segment_list.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
            )
            text_parts.append(segment.text.strip())

        # Combine text
        full_text = " ".join(text_parts)

        return TranscriptionResult(
            text=full_text,
            language=info.language,
            confidence=info.language_probability,
            segments=segment_list,
        )

    def detect_language(self, audio: np.ndarray) -> tuple[str, float]:
        """Detect the language of audio.

        Args:
            audio: Audio data as numpy array.

        Returns:
            Tuple of (language_code, probability).
        """
        model = self._ensure_model()

        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Ensure audio is 1D
        if audio.ndim > 1:
            audio = audio.flatten()

        # Use only first 30 seconds for language detection
        max_samples = 30 * 16000
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Detect language
        _, info = model.transcribe(audio, task="transcribe")

        return info.language, info.language_probability

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        self._model = None

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None


def get_available_models() -> list[str]:
    """Get list of available Whisper models.

    Returns:
        List of model names.
    """
    return WHISPER_MODELS.copy()


def get_supported_languages() -> dict[str, str]:
    """Get mapping of supported language codes to names.

    Returns:
        Dictionary mapping ISO 639-1 codes to language names.
    """
    return LANGUAGE_CODES.copy()
