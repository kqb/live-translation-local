"""Session logging module for recording and analyzing live translation sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


@dataclass
class LogEntry:
    """A single log entry with audio and metadata."""

    timestamp: str  # ISO format timestamp
    chunk_index: int  # Chunk number in session
    audio_file: str  # Relative path to audio file
    transcription: str  # Original transcribed text
    translation: str  # Translated text
    language: str  # Detected language
    confidence: float  # Language confidence
    speaker_label: Optional[str] = None  # Diarization label (SPEAKER_00, etc.)
    speaker_name: Optional[str] = None  # Recognized speaker name
    duration: float = 0.0  # Audio duration in seconds

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "chunk_index": self.chunk_index,
            "audio_file": self.audio_file,
            "transcription": self.transcription,
            "translation": self.translation,
            "language": self.language,
            "confidence": self.confidence,
            "speaker_label": self.speaker_label,
            "speaker_name": self.speaker_name,
            "duration": self.duration,
        }


@dataclass
class SessionMetadata:
    """Metadata for a logging session."""

    session_id: str  # Unique session identifier (timestamp-based)
    start_time: str  # ISO format start time
    end_time: Optional[str] = None  # ISO format end time
    total_chunks: int = 0  # Total chunks logged
    total_duration: float = 0.0  # Total audio duration
    config: dict = field(default_factory=dict)  # Session configuration snapshot

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_chunks": self.total_chunks,
            "total_duration": self.total_duration,
            "config": self.config,
        }


class SessionLogger:
    """Logs live translation sessions with audio chunks and metadata."""

    def __init__(self, log_dir: Path, config: dict = None):
        """Initialize session logger.

        Args:
            log_dir: Base directory for session logs.
            config: Configuration snapshot to save with session.
        """
        self.log_dir = Path(log_dir)
        self.config = config or {}

        # Create session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Session state
        self.metadata = SessionMetadata(
            session_id=self.session_id,
            start_time=datetime.now().isoformat(),
            config=self.config,
        )
        self.log_entries: list[LogEntry] = []
        self.chunk_index = 0

    def log_chunk(
        self,
        audio: np.ndarray,
        transcription: str,
        translation: str,
        language: str,
        confidence: float,
        speaker_label: Optional[str] = None,
        speaker_name: Optional[str] = None,
        sample_rate: int = 16000,
    ) -> None:
        """Log an audio chunk with its metadata.

        Args:
            audio: Audio data as numpy array.
            transcription: Transcribed text.
            translation: Translated text.
            language: Detected language.
            confidence: Language detection confidence.
            speaker_label: Speaker label from diarization (e.g., "SPEAKER_00").
            speaker_name: Recognized speaker name (e.g., "Alice").
            sample_rate: Audio sample rate.
        """
        # Save audio file
        audio_filename = f"chunk_{self.chunk_index:06d}.wav"
        audio_path = self.session_dir / "audio" / audio_filename
        audio_path.parent.mkdir(exist_ok=True)

        try:
            # Ensure audio is float32 for soundfile
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Save as WAV file
            sf.write(str(audio_path), audio, sample_rate)

            # Calculate duration
            duration = len(audio) / sample_rate

            # Create log entry
            entry = LogEntry(
                timestamp=datetime.now().isoformat(),
                chunk_index=self.chunk_index,
                audio_file=f"audio/{audio_filename}",
                transcription=transcription,
                translation=translation,
                language=language,
                confidence=confidence,
                speaker_label=speaker_label,
                speaker_name=speaker_name,
                duration=duration,
            )

            self.log_entries.append(entry)
            self.chunk_index += 1

            # Update metadata
            self.metadata.total_chunks = len(self.log_entries)
            self.metadata.total_duration += duration

            # Save metadata incrementally (every 10 chunks to avoid too many writes)
            if self.chunk_index % 10 == 0:
                self._save_metadata()

        except Exception as e:
            print(f"Failed to log audio chunk: {e}")

    def _save_metadata(self) -> None:
        """Save session metadata and log entries to JSON."""
        metadata_path = self.session_dir / "session.json"

        data = {
            "metadata": self.metadata.to_dict(),
            "entries": [entry.to_dict() for entry in self.log_entries],
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def finalize(self) -> None:
        """Finalize the session and save all data."""
        self.metadata.end_time = datetime.now().isoformat()
        self._save_metadata()

    def get_session_summary(self) -> dict:
        """Get a summary of the current session.

        Returns:
            Dictionary with session statistics.
        """
        return {
            "session_id": self.session_id,
            "total_chunks": self.metadata.total_chunks,
            "total_duration": self.metadata.total_duration,
            "start_time": self.metadata.start_time,
            "session_dir": str(self.session_dir),
        }


def load_session(session_dir: Path) -> tuple[SessionMetadata, list[LogEntry]]:
    """Load a recorded session from disk.

    Args:
        session_dir: Path to session directory.

    Returns:
        Tuple of (metadata, log_entries).
    """
    metadata_path = session_dir / "session.json"

    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = SessionMetadata(**data["metadata"])
    entries = [LogEntry(**entry_data) for entry_data in data["entries"]]

    return metadata, entries


def list_sessions(log_dir: Path) -> list[dict]:
    """List all recorded sessions.

    Args:
        log_dir: Base directory for session logs.

    Returns:
        List of session summaries.
    """
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return []

    sessions = []
    for session_path in sorted(log_dir.iterdir(), reverse=True):
        if not session_path.is_dir():
            continue

        metadata_path = session_path / "session.json"
        if not metadata_path.exists():
            continue

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                metadata = data["metadata"]

            sessions.append({
                "session_id": metadata["session_id"],
                "start_time": metadata["start_time"],
                "end_time": metadata.get("end_time"),
                "total_chunks": metadata["total_chunks"],
                "total_duration": metadata["total_duration"],
                "path": str(session_path),
            })
        except Exception as e:
            print(f"Failed to load session {session_path.name}: {e}")
            continue

    return sessions
