"""Main pipeline connecting audio capture, transcription, translation, and OBS output."""

from __future__ import annotations

import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .audio_capture import AudioCapture, AudioConfig
from .obs_output import OBSOutput, OutputConfig
from .transcriber import Transcriber, TranscriberConfig
from .translator import Translator, TranslatorConfig


console = Console()


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    transcriber: TranscriberConfig = field(default_factory=TranscriberConfig)
    translator: TranslatorConfig = field(default_factory=TranslatorConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        """Create config from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            PipelineConfig instance.
        """
        audio_data = data.get("audio", {})
        whisper_data = data.get("whisper", {})
        translation_data = data.get("translation", {})
        output_data = data.get("output", {})

        return cls(
            audio=AudioConfig(
                device=audio_data.get("device"),
                sample_rate=audio_data.get("sample_rate", 16000),
                chunk_duration=audio_data.get("chunk_duration", 3.0),
            ),
            transcriber=TranscriberConfig(
                model=whisper_data.get("model", "base"),
                device=whisper_data.get("device", "auto"),
                compute_type=whisper_data.get("compute_type", "int8"),
                language=whisper_data.get("language"),
            ),
            translator=TranslatorConfig(
                enabled=translation_data.get("enabled", True),
                source_lang=translation_data.get("source_lang"),
                target_lang=translation_data.get("target_lang", "es"),
            ),
            output=OutputConfig(
                text_file=output_data.get("text_file", "./subtitles.txt"),
                websocket_enabled=output_data.get("websocket_enabled", True),
                websocket_port=output_data.get("websocket_port", 8765),
                http_port=output_data.get("http_port", 8766),
                max_lines=output_data.get("max_lines", 2),
                clear_after=output_data.get("clear_after", 5.0),
            ),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load config from YAML file.

        Args:
            path: Path to YAML config file.

        Returns:
            PipelineConfig instance.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})


class Pipeline:
    """Main pipeline connecting all components for live translation."""

    def __init__(self, config: PipelineConfig):
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration.
        """
        self.config = config

        # Initialize components
        self.audio_capture = AudioCapture(config.audio)
        self.transcriber = Transcriber(config.transcriber)
        self.translator = Translator(config.translator)
        self.output = OBSOutput(config.output)

        # Pipeline state
        self._running = False
        self._processing_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Statistics
        self._stats = {
            "chunks_processed": 0,
            "transcriptions": 0,
            "translations": 0,
            "errors": 0,
        }

    def _process_audio_chunk(self, audio: np.ndarray) -> None:
        """Process an audio chunk through the pipeline.

        Args:
            audio: Audio chunk as numpy array.
        """
        try:
            self._stats["chunks_processed"] += 1

            # Transcribe
            transcription = self.transcriber.transcribe(audio)

            if not transcription.text.strip():
                return

            self._stats["transcriptions"] += 1

            # Determine source language
            source_lang = transcription.language
            if self.config.translator.source_lang:
                source_lang = self.config.translator.source_lang

            # Translate if enabled
            if self.config.translator.enabled:
                translation = self.translator.translate(
                    transcription.text,
                    source_lang=source_lang,
                )
                self._stats["translations"] += 1
                translated_text = translation.translated_text
            else:
                translated_text = ""

            # Output
            self.output.update(transcription.text, translated_text)

            # Display in console
            self._display_subtitle(transcription.text, translated_text, source_lang)

        except Exception as e:
            self._stats["errors"] += 1
            console.print(f"[red]Error processing audio: {e}[/red]")

    def _display_subtitle(
        self, original: str, translated: str, language: str
    ) -> None:
        """Display subtitle in console.

        Args:
            original: Original text.
            translated: Translated text.
            language: Detected language.
        """
        text = Text()
        text.append(f"[{language}] ", style="dim")
        text.append(original, style="white bold")

        if translated:
            text.append("\n")
            text.append(f"  → ", style="dim")
            text.append(translated, style="cyan italic")

        console.print(Panel(text, border_style="blue", padding=(0, 1)))

    def _processing_loop(self) -> None:
        """Main processing loop running in a separate thread."""
        while not self._stop_event.is_set():
            # Get audio chunk with timeout
            audio = self.audio_capture.get_chunk(timeout=0.5)

            if audio is not None:
                self._process_audio_chunk(audio)
            elif self.output.should_clear():
                self.output.clear()

    def start(self) -> None:
        """Start the pipeline."""
        if self._running:
            return

        console.print("[bold green]Starting Live Translator...[/bold green]")

        # Start components
        self.output.start()
        self.audio_capture.start()

        # Start processing thread
        self._running = True
        self._stop_event.clear()
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
        )
        self._processing_thread.start()

        # Display status
        if self.config.output.websocket_enabled:
            console.print(
                f"[blue]Browser source URL:[/blue] {self.output.overlay_url}"
            )
            console.print(
                f"[blue]WebSocket URL:[/blue] {self.output.websocket_url}"
            )
        console.print(
            f"[blue]Text file:[/blue] {self.config.output.text_file}"
        )
        console.print()
        console.print("[dim]Press Ctrl+C to stop[/dim]")
        console.print()

    def stop(self) -> None:
        """Stop the pipeline."""
        if not self._running:
            return

        console.print("\n[yellow]Stopping Live Translator...[/yellow]")

        # Signal processing thread to stop
        self._stop_event.set()

        # Stop components
        self.audio_capture.stop()
        self.output.stop()

        # Wait for processing thread
        if self._processing_thread is not None:
            self._processing_thread.join(timeout=2.0)
            self._processing_thread = None

        self._running = False

        # Display statistics
        console.print()
        console.print("[bold]Session Statistics:[/bold]")
        console.print(f"  Chunks processed: {self._stats['chunks_processed']}")
        console.print(f"  Transcriptions: {self._stats['transcriptions']}")
        console.print(f"  Translations: {self._stats['translations']}")
        console.print(f"  Errors: {self._stats['errors']}")

    def run(self) -> None:
        """Run the pipeline with graceful shutdown on Ctrl+C."""
        # Set up signal handlers
        def signal_handler(sig, frame):
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start pipeline
        self.start()

        # Keep main thread alive
        try:
            while self._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._running

    def __enter__(self) -> "Pipeline":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


def load_config(config_path: Optional[Path] = None) -> PipelineConfig:
    """Load pipeline configuration.

    Args:
        config_path: Optional path to config file.

    Returns:
        PipelineConfig instance.
    """
    # Default config locations
    default_paths = [
        Path("config.yaml"),
        Path("config.yml"),
        Path.home() / ".config" / "live-translator" / "config.yaml",
    ]

    if config_path:
        paths = [config_path]
    else:
        paths = default_paths

    for path in paths:
        if path.exists():
            console.print(f"[dim]Loading config from: {path}[/dim]")
            return PipelineConfig.from_yaml(path)

    # Return default config
    console.print("[dim]Using default configuration[/dim]")
    return PipelineConfig()


def run_pipeline(config_path: Optional[Path] = None) -> None:
    """Run the live translation pipeline.

    Args:
        config_path: Optional path to config file.
    """
    config = load_config(config_path)
    pipeline = Pipeline(config)
    pipeline.run()
