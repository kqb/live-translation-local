"""Speaker diarization module using pyannote.audio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class DiarizationConfig:
    """Configuration for speaker diarization."""

    enabled: bool = False  # Enable speaker diarization
    min_speakers: int = 1  # Minimum number of speakers
    max_speakers: int = 10  # Maximum number of speakers
    hf_token: Optional[str] = None  # Hugging Face token for model access


@dataclass
class SpeakerSegment:
    """A segment of speech from a specific speaker."""

    start: float  # Start time in seconds
    end: float  # End time in seconds
    speaker: str  # Speaker label (e.g., "SPEAKER_00", "SPEAKER_01")


class Diarizer:
    """Speaker diarization using pyannote.audio."""

    def __init__(self, config: DiarizationConfig):
        """Initialize the diarizer.

        Args:
            config: Diarization configuration.
        """
        self.config = config
        self._pipeline = None

    def _ensure_pipeline(self):
        """Ensure the diarization pipeline is loaded (lazy loading)."""
        if self._pipeline is None:
            try:
                from pyannote.audio import Pipeline

                # Load the speaker diarization pipeline
                # Note: Requires Hugging Face token and model acceptance
                if self.config.hf_token:
                    self._pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        token=self.config.hf_token,
                    )
                else:
                    # Try without token (may fail if model requires acceptance)
                    self._pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1"
                    )

                # Move to GPU if available
                if torch.cuda.is_available():
                    self._pipeline.to(torch.device("cuda"))

            except Exception as e:
                raise RuntimeError(
                    f"Failed to load diarization pipeline. "
                    f"Make sure you've accepted the model at "
                    f"https://huggingface.co/pyannote/speaker-diarization-3.1 "
                    f"and provided a valid HF token. Error: {e}"
                )

        return self._pipeline

    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> list[SpeakerSegment]:
        """Perform speaker diarization on audio.

        Args:
            audio: Audio data as numpy array (float32, mono).
            sample_rate: Sample rate of the audio (default: 16000).

        Returns:
            List of speaker segments with timestamps.
        """
        if not self.config.enabled:
            return []

        pipeline = self._ensure_pipeline()

        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Ensure audio is 1D
        if audio.ndim > 1:
            audio = audio.flatten()

        # Convert numpy array to torch tensor
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)

        # Create a dictionary with audio data for pyannote
        # pyannote expects dict with 'waveform' and 'sample_rate'
        audio_dict = {
            "waveform": audio_tensor,
            "sample_rate": sample_rate,
        }

        # Run diarization
        try:
            diarization = pipeline(
                audio_dict,
                min_speakers=self.config.min_speakers,
                max_speakers=self.config.max_speakers,
            )
        except Exception as e:
            # If diarization fails, return empty list
            print(f"Diarization failed: {e}")
            return []

        # Convert pyannote output to our format
        segments = []

        # Handle different pyannote API versions
        # Newer pyannote (4.x) returns DiarizeOutput with speaker_diarization attribute
        if hasattr(diarization, 'speaker_diarization'):
            # Use the Annotation object from DiarizeOutput
            annotation = diarization.speaker_diarization
            for segment, _, speaker in annotation.itertracks(yield_label=True):
                segments.append(
                    SpeakerSegment(
                        start=segment.start,
                        end=segment.end,
                        speaker=speaker,
                    )
                )
        elif hasattr(diarization, 'itertracks'):
            # Older API: diarization is directly an Annotation object
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(
                    SpeakerSegment(
                        start=turn.start,
                        end=turn.end,
                        speaker=speaker,
                    )
                )
        else:
            print(f"Unknown diarization output format: {type(diarization)}")

        return segments

    def assign_speaker_to_segments(
        self,
        transcription_segments: list[dict],
        speaker_segments: list[SpeakerSegment],
    ) -> list[dict]:
        """Assign speaker labels to transcription segments.

        Args:
            transcription_segments: List of transcription segments with 'start', 'end', 'text'.
            speaker_segments: List of speaker diarization segments.

        Returns:
            Transcription segments with added 'speaker' field.
        """
        if not speaker_segments:
            return transcription_segments

        # For each transcription segment, find the overlapping speaker
        for trans_seg in transcription_segments:
            trans_start = trans_seg["start"]
            trans_end = trans_seg["end"]
            trans_mid = (trans_start + trans_end) / 2

            # Find speaker segment that contains the midpoint
            best_speaker = "UNKNOWN"
            max_overlap = 0

            for speaker_seg in speaker_segments:
                # Calculate overlap
                overlap_start = max(trans_start, speaker_seg.start)
                overlap_end = min(trans_end, speaker_seg.end)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = speaker_seg.speaker

            trans_seg["speaker"] = best_speaker

        return transcription_segments

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        self._pipeline = None

    @property
    def is_loaded(self) -> bool:
        """Check if pipeline is loaded."""
        return self._pipeline is not None
