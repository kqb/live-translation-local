"""Transcriber module using whisper.cpp for speech-to-text."""

from __future__ import annotations

import json
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# Common hallucination phrases to filter out (multilingual)
HALLUCINATION_PATTERNS = [
    # English
    "thank you for watching",
    "thanks for watching",
    "thank you",
    "thanks",
    "subscribe",
    "like and subscribe",
    "please subscribe",
    "hello",
    "ayo",
    "why",
    "music",
    "♪",
    ".",
    "...",
    # Norwegian/Swedish/Danish
    "takk for at du",
    "takk for deg",
    "tack för att",
    "tak fordi",
    "hei",
    # German
    "danke fürs zuschauen",
    "danke schön",
    # French
    "merci d'avoir regardé",
    "merci",
    # Spanish
    "gracias por ver",
    "gracias",
    # Generic patterns
    "org-",
]


@dataclass
class WhisperCppConfig:
    """Configuration for the whisper.cpp transcriber."""

    model_path: str  # Path to GGML model file
    language: Optional[str] = None  # None = auto-detect
    threads: int = 8  # Number of threads to use
    use_gpu: bool = True  # Use GPU acceleration


@dataclass
class TranscriptionResult:
    """Result of a transcription."""

    text: str  # Transcribed text
    language: str  # Detected or specified language
    confidence: float  # Language detection probability (estimated)
    segments: list[dict]  # Detailed segments with timestamps


class WhisperCppTranscriber:
    """Speech-to-text transcriber using whisper.cpp."""

    def __init__(self, config: WhisperCppConfig):
        """Initialize the transcriber.

        Args:
            config: Transcriber configuration.
        """
        self.config = config

        # Verify model file exists
        model_path = Path(config.model_path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = str(model_path)

        # Find whisper-cli binary
        self.whisper_cli = self._find_whisper_cli()

    def _find_whisper_cli(self) -> str:
        """Find the whisper-cli binary."""
        # Try common locations
        locations = [
            "whisper-cli",  # In PATH
            "/opt/homebrew/bin/whisper-cli",
            "/usr/local/bin/whisper-cli",
        ]

        for location in locations:
            try:
                result = subprocess.run(
                    [location, "--help"],
                    capture_output=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return location
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

        raise RuntimeError("whisper-cli not found. Install whisper.cpp first.")

    def _save_audio_to_wav(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Save audio to a temporary WAV file.

        Args:
            audio: Audio data as numpy array.
            sample_rate: Sample rate in Hz.

        Returns:
            Path to temporary WAV file.
        """
        # Create temporary file
        fd, temp_path = tempfile.mkstemp(suffix=".wav")

        # CRITICAL: Close the file descriptor immediately to prevent leak
        # wave.open() will open the file independently
        import os
        os.close(fd)

        # Ensure audio is float32 and in range [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Ensure audio is 1D
        if audio.ndim > 1:
            audio = audio.flatten()

        # Convert to int16 for WAV format
        audio_int16 = (audio * 32767).astype(np.int16)

        # Write WAV file
        with wave.open(temp_path, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        return temp_path

    def _is_hallucination(self, text: str) -> bool:
        """Check if text appears to be a hallucination.

        Args:
            text: Transcribed text to check.

        Returns:
            True if text is likely a hallucination.
        """
        if not text or len(text.strip()) == 0:
            return True

        # Remove punctuation and convert to lowercase for comparison
        import string
        text_clean = text.strip().rstrip(string.punctuation).lower()

        # Check for very short transcriptions (often hallucinations)
        if len(text_clean) < 2:
            return True

        # Check for common hallucination patterns
        for pattern in HALLUCINATION_PATTERNS:
            pattern_lower = pattern.lower()
            # Exact match or starts with pattern
            if text_clean == pattern_lower or text_clean.startswith(pattern_lower):
                return True

        # Check for repetitive patterns (e.g., "thank you thank you thank you")
        words = text_clean.split()
        if len(words) > 2:
            # If more than 50% of words are the same, it's likely a hallucination
            unique_words = set(words)
            if len(unique_words) == 1:
                return True

        return False

    def _detect_language_from_text(self, text: str) -> str:
        """Detect language from text characters.

        Args:
            text: Text to analyze.

        Returns:
            Language code (ja, en, etc.).
        """
        if not text:
            return "en"  # Default to English for empty text

        # Count character types
        hiragana_count = 0
        katakana_count = 0
        kanji_count = 0
        latin_count = 0

        for char in text:
            code_point = ord(char)
            # Hiragana: U+3040 to U+309F
            if 0x3040 <= code_point <= 0x309F:
                hiragana_count += 1
            # Katakana: U+30A0 to U+30FF
            elif 0x30A0 <= code_point <= 0x30FF:
                katakana_count += 1
            # CJK Unified Ideographs (Kanji): U+4E00 to U+9FFF
            elif 0x4E00 <= code_point <= 0x9FFF:
                kanji_count += 1
            # Latin alphabet
            elif (0x0041 <= code_point <= 0x005A) or (0x0061 <= code_point <= 0x007A):
                latin_count += 1

        # If any Japanese characters detected, it's Japanese
        japanese_chars = hiragana_count + katakana_count + kanji_count
        if japanese_chars > 0:
            return "ja"

        # Otherwise default to English
        return "en"

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
        # Save audio to temporary WAV file
        temp_wav = self._save_audio_to_wav(audio)

        try:
            # Build command
            lang = language or self.config.language or "auto"

            cmd = [
                self.whisper_cli,
                "-m", self.model_path,
                "-l", lang,
                "-t", str(self.config.threads),
                "-f", temp_wav,
                "-oj",  # Output JSON
                "-np",  # No prints
            ]

            # Disable GPU if configured
            if not self.config.use_gpu:
                cmd.append("-ng")

            # Run whisper-cli
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                raise RuntimeError(f"whisper-cli failed: {result.stderr}")

            # Parse JSON output
            # whisper-cli outputs JSON to stdout
            json_output = result.stdout.strip()

            # The output might be in a .json file instead
            # Try to find it
            output_json_path = Path(temp_wav).with_suffix(".wav.json")
            if output_json_path.exists():
                with open(output_json_path) as f:
                    data = json.load(f)
                output_json_path.unlink()  # Clean up
            else:
                # Parse from stdout
                data = json.loads(json_output)

            # Extract transcription data
            transcription_data = data.get("transcription", [])

            # Combine text from all segments
            text_parts = []
            segments = []

            for segment in transcription_data:
                text = segment.get("text", "").strip()
                if text:
                    text_parts.append(text)
                    segments.append({
                        "start": segment.get("offsets", {}).get("from", 0) / 1000.0,
                        "end": segment.get("offsets", {}).get("to", 0) / 1000.0,
                        "text": text,
                    })

            full_text = " ".join(text_parts)

            # Filter hallucinations
            if self._is_hallucination(full_text):
                full_text = ""
                segments = []

            # Get detected language from whisper.cpp output
            # whisper.cpp may include language in the model result or we use text-based detection
            if lang != "auto":
                # User specified language
                detected_lang = lang
            else:
                # Try to get language from whisper.cpp output
                # Some whisper.cpp versions include "result" -> "language"
                whisper_detected_lang = None
                if "result" in data and isinstance(data["result"], dict):
                    whisper_detected_lang = data["result"].get("language")

                # Fall back to text-based detection if whisper didn't provide language
                if whisper_detected_lang:
                    detected_lang = whisper_detected_lang
                else:
                    detected_lang = self._detect_language_from_text(full_text)

            # Estimate confidence (whisper.cpp doesn't provide this directly)
            confidence = 0.9  # Default confidence

            return TranscriptionResult(
                text=full_text,
                language=detected_lang,
                confidence=confidence,
                segments=segments,
            )

        finally:
            # Clean up temporary file
            try:
                Path(temp_wav).unlink()
            except Exception:
                pass

    def detect_language(self, audio: np.ndarray) -> tuple[str, float]:
        """Detect the language of audio.

        Args:
            audio: Audio data as numpy array.

        Returns:
            Tuple of (language_code, probability).
        """
        # Transcribe with auto language detection
        result = self.transcribe(audio, language="auto")
        return result.language, result.confidence


# Helper function to create config from pipeline config
def create_whisper_cpp_config(
    model: str = "medium",
    language: Optional[str] = None,
    threads: int = 8,
) -> WhisperCppConfig:
    """Create WhisperCppConfig from pipeline settings.

    Args:
        model: Model size (medium, large-v3).
        language: Language code or None for auto-detect.
        threads: Number of threads.

    Returns:
        WhisperCppConfig instance.
    """
    # Map model names to GGML files
    model_files = {
        "medium": "~/.whisper-cpp-models/ggml-medium.bin",
        "large-v3": "~/.whisper-cpp-models/ggml-large-v3.bin",
        "large": "~/.whisper-cpp-models/ggml-large-v3.bin",
    }

    model_path = model_files.get(model, model_files["medium"])

    return WhisperCppConfig(
        model_path=model_path,
        language=language,
        threads=threads,
        use_gpu=True,
    )
