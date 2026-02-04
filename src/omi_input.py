"""Omi wearable audio capture module using omi-sdk for BLE audio streaming."""

from __future__ import annotations

import asyncio
import queue
import struct
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
from omi import OmiOpusDecoder, listen_to_omi, print_devices as omi_print_devices
from rich.console import Console

console = Console()


# Omi BLE UUIDs
AUDIO_CHARACTERISTIC_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"


@dataclass
class OmiConfig:
    """Configuration for Omi audio capture."""

    mac_address: Optional[str] = None  # None = auto-discover
    auto_connect: bool = True
    auto_reconnect: bool = True
    reconnect_delay: float = 5.0
    battery_monitoring: bool = True
    sample_rate: int = 16000  # Fixed for Omi
    chunk_duration: float = 3.0  # seconds per chunk


class OmiAudioCapture:
    """Thread-safe audio capture from Omi wearable with BLE streaming."""

    def __init__(self, config: OmiConfig):
        """Initialize Omi audio capture.

        Args:
            config: Omi configuration.
        """
        self.config = config
        self.decoder = OmiOpusDecoder()

        # Calculate samples per chunk
        self.chunk_samples = int(config.sample_rate * config.chunk_duration)

        # Thread-safe queue for audio chunks
        self._queue: queue.Queue[np.ndarray] = queue.Queue()

        # Buffer for accumulating audio samples
        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._buffer_lock = threading.Lock()

        # BLE and threading state
        self._running = False
        self._ble_thread: Optional[threading.Thread] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._battery_level: int = 100  # Unknown initially

    def _audio_handler(self, _sender: int, data: bytes) -> None:
        """Handle incoming audio data from Omi BLE characteristic.

        Args:
            _sender: BLE characteristic sender (unused).
            data: Raw Opus-encoded audio packet.
        """
        try:
            # Decode Opus to PCM bytes (16-bit, 16kHz, mono)
            pcm_bytes = self.decoder.decode_packet(data)
            if not pcm_bytes:
                return

            # Convert PCM bytes (int16) to numpy float32 array
            # PCM is little-endian signed 16-bit integers
            pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            audio_data = pcm_int16.astype(np.float32) / 32768.0  # Normalize to [-1.0, 1.0]

            # Accumulate samples
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

        except Exception as e:
            console.print(f"[red]Error processing Omi audio packet: {e}[/red]")

    async def _connect_and_listen(self) -> None:
        """Async task to connect to Omi and listen for audio data."""
        mac_address = self.config.mac_address

        # Auto-discover if no MAC address provided
        if not mac_address:
            console.print("[yellow]No MAC address provided, scanning for Omi device...[/yellow]")
            # In a real implementation, we'd scan and find the first "Omi" device
            # For now, require explicit MAC address
            console.print("[red]Error: MAC address required. Use config.audio.omi.mac_address[/red]")
            return

        while self._running:
            try:
                console.print(f"[dim]→ Connecting to Omi device at {mac_address}...[/dim]")

                # Connect and listen (this blocks until disconnected)
                await listen_to_omi(
                    mac_address=mac_address,
                    char_uuid=AUDIO_CHARACTERISTIC_UUID,
                    data_handler=self._audio_handler,
                )

            except Exception as e:
                console.print(f"[red]Omi connection error: {e}[/red]")

                if not self.config.auto_reconnect or not self._running:
                    break

                console.print(
                    f"[yellow]Reconnecting in {self.config.reconnect_delay} seconds...[/yellow]"
                )
                await asyncio.sleep(self.config.reconnect_delay)

    def _run_event_loop(self) -> None:
        """Run the asyncio event loop in a separate thread."""
        self._event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._event_loop)

        try:
            self._event_loop.run_until_complete(self._connect_and_listen())
        except Exception as e:
            console.print(f"[red]Omi event loop error: {e}[/red]")
        finally:
            self._event_loop.close()

    def start(self) -> None:
        """Start Omi audio capture."""
        if self._running:
            return

        self._running = True

        # Start BLE connection in a separate thread with its own event loop
        self._ble_thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
            name="OmiAudioCapture-BLE",
        )
        self._ble_thread.start()

        console.print("[green]✓ Omi audio capture started[/green]")

    def stop(self) -> None:
        """Stop Omi audio capture."""
        self._running = False

        # Wait for BLE thread to finish
        if self._ble_thread is not None:
            self._ble_thread.join(timeout=2.0)
            self._ble_thread = None

        # Clear the buffer
        with self._buffer_lock:
            self._buffer = []
            self._buffer_samples = 0

        console.print("[yellow]Omi audio capture stopped[/yellow]")

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

    @property
    def battery_level(self) -> int:
        """Get current battery level (0-100)."""
        return self._battery_level

    def __enter__(self) -> "OmiAudioCapture":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


def list_omi_devices() -> None:
    """List available Omi devices via BLE scan."""
    console.print("[yellow]Scanning for BLE devices...[/yellow]")
    omi_print_devices()
