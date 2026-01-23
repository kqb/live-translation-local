"""Audio capture module using sounddevice for real-time microphone input."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sounddevice as sd


@dataclass
class AudioConfig:
    """Configuration for audio capture."""

    device: Optional[int | str] = None  # None = default mic
    sample_rate: int = 16000  # 16kHz for Whisper
    chunk_duration: float = 3.0  # seconds per chunk
    channels: int = 1  # mono


class AudioCapture:
    """Thread-safe audio capture from microphone with chunking support."""

    def __init__(
        self,
        config: AudioConfig,
        callback: Optional[Callable[[np.ndarray], None]] = None,
    ):
        """Initialize audio capture.

        Args:
            config: Audio configuration.
            callback: Optional callback function called with each audio chunk.
        """
        self.config = config
        self.callback = callback

        # Calculate samples per chunk
        self.chunk_samples = int(config.sample_rate * config.chunk_duration)

        # Thread-safe queue for audio chunks
        self._queue: queue.Queue[np.ndarray] = queue.Queue()

        # Buffer for accumulating audio samples
        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._buffer_lock = threading.Lock()

        # Stream state
        self._stream: Optional[sd.InputStream] = None
        self._running = False

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Internal callback for sounddevice stream.

        Accumulates audio samples and emits chunks when threshold is reached.
        """
        if status:
            # Log status flags (overflows, underflows)
            pass

        # Copy the audio data
        audio_data = indata[:, 0].copy() if indata.ndim > 1 else indata.copy().flatten()

        with self._buffer_lock:
            self._buffer.append(audio_data)
            self._buffer_samples += len(audio_data)

            # Check if we have enough samples for a chunk
            if self._buffer_samples >= self.chunk_samples:
                # Concatenate all buffered audio
                full_audio = np.concatenate(self._buffer)

                # Extract the chunk
                chunk = full_audio[: self.chunk_samples]

                # Keep remaining samples in buffer
                remaining = full_audio[self.chunk_samples :]
                self._buffer = [remaining] if len(remaining) > 0 else []
                self._buffer_samples = len(remaining)

                # Put chunk in queue
                self._queue.put(chunk)

                # Call callback if provided
                if self.callback is not None:
                    try:
                        self.callback(chunk)
                    except Exception:
                        pass  # Don't crash on callback errors

    def start(self) -> None:
        """Start audio capture."""
        if self._running:
            return

        self._running = True

        # Create and start the input stream
        self._stream = sd.InputStream(
            device=self.config.device,
            channels=self.config.channels,
            samplerate=self.config.sample_rate,
            dtype=np.float32,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop audio capture."""
        self._running = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Clear the buffer
        with self._buffer_lock:
            self._buffer = []
            self._buffer_samples = 0

    def get_chunk(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Get the next audio chunk from the queue.

        Args:
            timeout: Maximum time to wait for a chunk (None = block forever).

        Returns:
            Audio chunk as numpy array, or None if timeout.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_chunk_nowait(self) -> Optional[np.ndarray]:
        """Get the next audio chunk without blocking.

        Returns:
            Audio chunk as numpy array, or None if no chunk available.
        """
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def flush_buffer(self) -> Optional[np.ndarray]:
        """Flush any remaining audio in the buffer.

        Returns:
            Remaining audio as numpy array, or None if buffer is empty.
        """
        with self._buffer_lock:
            if self._buffer_samples > 0:
                chunk = np.concatenate(self._buffer)
                self._buffer = []
                self._buffer_samples = 0
                return chunk
            return None

    @property
    def is_running(self) -> bool:
        """Check if audio capture is running."""
        return self._running

    def __enter__(self) -> "AudioCapture":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


def list_devices() -> list[dict]:
    """List available audio input devices.

    Returns:
        List of device info dictionaries.
    """
    devices = sd.query_devices()
    input_devices = []

    for i, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            input_devices.append(
                {
                    "index": i,
                    "name": device["name"],
                    "channels": device["max_input_channels"],
                    "sample_rate": device["default_samplerate"],
                    "is_default": i == sd.default.device[0],
                }
            )

    return input_devices


def get_default_device() -> Optional[int]:
    """Get the default input device index.

    Returns:
        Default input device index, or None if no default.
    """
    try:
        default = sd.default.device[0]
        return default if default >= 0 else None
    except Exception:
        return None
