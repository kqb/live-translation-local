"""Backend detection and factory."""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING, Optional

from rich.console import Console

if TYPE_CHECKING:
    from ..transcriber import TranscriberConfig
    from .base import WhisperBackend

console = Console()


def detect_best_backend() -> str:
    """Auto-detect the best Whisper backend for current hardware.

    Returns:
        Backend name: "whisper-cpp" or "faster-whisper"
    """
    # 1. Check for Apple Silicon (M1/M2/M3/M4)
    if platform.machine() == "arm64" and platform.system() == "Darwin":
        console.print("[dim]→ Detected Apple Silicon - using whisper.cpp for optimal performance[/dim]")
        return "whisper-cpp"

    # 2. Check for CUDA GPU
    try:
        import torch

        if torch.cuda.is_available():
            console.print("[dim]→ Detected CUDA GPU - using faster-whisper[/dim]")
            return "faster-whisper"
    except ImportError:
        pass

    # 3. Default to faster-whisper for CPU
    console.print("[dim]→ Using faster-whisper (CPU mode)[/dim]")
    return "faster-whisper"


def get_backend(
    config: TranscriberConfig,
    backend_override: Optional[str] = None,
) -> WhisperBackend:
    """Get the appropriate Whisper backend.

    Args:
        config: Transcriber configuration.
        backend_override: Optional backend name to override auto-detection.

    Returns:
        WhisperBackend instance.

    Raises:
        ValueError: If backend_override is invalid.
        ImportError: If required backend package is not installed.
    """
    # Determine which backend to use
    backend_name = backend_override or detect_best_backend()

    if backend_name == "whisper-cpp":
        try:
            from .whisper_cpp import WhisperCppBackend

            return WhisperCppBackend(config)
        except ImportError as e:
            console.print(
                f"[yellow]Warning: Could not load whisper-cpp backend: {e}[/yellow]"
            )
            console.print("[yellow]Falling back to faster-whisper[/yellow]")
            backend_name = "faster-whisper"

    if backend_name == "faster-whisper":
        from .faster_whisper import FasterWhisperBackend

        return FasterWhisperBackend(config)

    raise ValueError(f"Unknown backend: {backend_name}")
