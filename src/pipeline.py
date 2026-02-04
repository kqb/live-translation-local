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
from .whisper_cpp_transcriber import WhisperCppTranscriber, create_whisper_cpp_config
from .g2_output import G2Output, G2Config
from .omi_input import OmiAudioCapture, OmiConfig


console = Console()


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    omi: OmiConfig = field(default_factory=OmiConfig)
    audio_source: str = "microphone"  # "microphone" or "omi"
    transcriber: TranscriberConfig = field(default_factory=TranscriberConfig)
    translator: TranslatorConfig = field(default_factory=TranslatorConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    g2: G2Config = field(default_factory=G2Config)
    backend: str = "whisper-cpp"  # "whisper-cpp" or "faster-whisper"
    translated_only: bool = False  # Only show translated text in console output

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        """Create config from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            PipelineConfig instance.
        """
        audio_data = data.get("audio", {})
        omi_data = audio_data.get("omi", {})
        whisper_data = data.get("whisper", {})
        translation_data = data.get("translation", {})
        output_data = data.get("output", {})
        g2_data = data.get("g2", {})

        return cls(
            audio_source=audio_data.get("source", "microphone"),
            audio=AudioConfig(
                device=audio_data.get("device"),
                sample_rate=audio_data.get("sample_rate", 16000),
                chunk_duration=audio_data.get("chunk_duration", 3.0),
            ),
            omi=OmiConfig(
                mac_address=omi_data.get("mac_address"),
                auto_connect=omi_data.get("auto_connect", True),
                auto_reconnect=omi_data.get("auto_reconnect", True),
                reconnect_delay=omi_data.get("reconnect_delay", 5.0),
                battery_monitoring=omi_data.get("battery_monitoring", True),
                chunk_duration=audio_data.get("chunk_duration", 3.0),
            ),
            transcriber=TranscriberConfig(
                model=whisper_data.get("model", "base"),
                device=whisper_data.get("device", "auto"),
                compute_type=whisper_data.get("compute_type", "int8"),
                language=whisper_data.get("language"),
                no_speech_threshold=whisper_data.get("no_speech_threshold", 0.6),
                logprob_threshold=whisper_data.get("logprob_threshold", -1.0),
                compression_ratio_threshold=whisper_data.get("compression_ratio_threshold", 2.4),
            ),
            translator=TranslatorConfig(
                enabled=translation_data.get("enabled", True),
                source_lang=translation_data.get("source_lang"),
                target_lang=translation_data.get("target_lang", "es"),
                glossary_path=translation_data.get("glossary_path"),
            ),
            output=OutputConfig(
                text_file=output_data.get("text_file", "./subtitles.txt"),
                websocket_enabled=output_data.get("websocket_enabled", True),
                websocket_port=output_data.get("websocket_port", 8765),
                http_port=output_data.get("http_port", 8766),
                max_lines=output_data.get("max_lines", 2),
                clear_after=output_data.get("clear_after", 5.0),
                scrolling_mode=output_data.get("scrolling_mode", True),
                history_lines=output_data.get("history_lines", 10),
                display_format=output_data.get("display_format", "both"),
            ),
            g2=G2Config(
                enabled=g2_data.get("enabled", False),
                mode=g2_data.get("mode", "teleprompter"),
                auto_connect=g2_data.get("auto_connect", True),
                use_right=g2_data.get("use_right", False),
                display_format=g2_data.get("display_format", "both"),
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

        # Initialize audio capture based on source
        if config.audio_source == "omi":
            console.print("[dim]→ Using Omi wearable as audio source[/dim]")
            self.audio_capture = OmiAudioCapture(config.omi)
        else:
            console.print("[dim]→ Using microphone as audio source[/dim]")
            self.audio_capture = AudioCapture(config.audio)

        # Choose transcriber backend
        if config.backend == "whisper-cpp":
            # Use whisper.cpp for Apple Silicon optimization
            whisper_cpp_config = create_whisper_cpp_config(
                model=config.transcriber.model,
                language=config.transcriber.language,
                threads=10,  # M4 Max has many cores
            )
            self.transcriber = WhisperCppTranscriber(whisper_cpp_config)
        else:
            # Use faster-whisper (default for CUDA/CPU)
            self.transcriber = Transcriber(config.transcriber)

        self.translator = Translator(config.translator)
        self.output = OBSOutput(config.output)

        # G2 smart glasses output (optional)
        self.g2_output: Optional[G2Output] = None
        if config.g2.enabled:
            try:
                self.g2_output = G2Output(config.g2)
            except ImportError as e:
                console.print(f"[yellow]G2 output disabled: {e}[/yellow]")

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

        # Sentence buffering for better translation context
        self._sentence_buffer = ""
        self._buffer_language = None
        self._last_update_time = time.time()

        # Context history for better translation (last 3 sentences)
        self._translation_context = []  # List of recent (original, translated) tuples
        self._max_context_sentences = 3

    def _build_translation_context(self, new_text: str) -> str:
        """Build translation context from recent sentences.

        Args:
            new_text: New sentence to translate.

        Returns:
            Text with context prepended (or just new_text if no context).
        """
        if not self._translation_context:
            return new_text

        # Include last N sentences as context
        context_parts = [orig for orig, _ in self._translation_context[-self._max_context_sentences:]]
        context_parts.append(new_text)
        return " ".join(context_parts)

    def _extract_new_translation(self, full_translation: str, original_text: str) -> str:
        """Extract the translation of the new sentence from full context translation.

        Args:
            full_translation: Translation that may include context.
            original_text: The new sentence that was being translated.

        Returns:
            Just the translation of the new sentence.
        """
        # Simple heuristic: If translation has multiple sentences, take the last one
        # This works because we append the new sentence at the end of context
        sentences = full_translation.split("。")  # Japanese sentence ending
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return full_translation

        # Return the last sentence (most likely the new one)
        return sentences[-1] + "。" if sentences[-1] else full_translation

    def _update_translation_context(self, original: str, translated: str) -> None:
        """Update the translation context with a new sentence pair.

        Args:
            original: Original English text.
            translated: Translated Japanese text.
        """
        self._translation_context.append((original, translated))
        # Keep only last N sentences
        if len(self._translation_context) > self._max_context_sentences:
            self._translation_context = self._translation_context[-self._max_context_sentences:]

    def _extract_complete_sentences(self, text: str, language: str) -> tuple[str, str]:
        """Extract complete sentences from text, leaving partial sentence in buffer.

        Args:
            text: Text to process.
            language: Language code (ja or en).

        Returns:
            Tuple of (complete_sentences, partial_sentence).
        """
        if not text:
            return "", ""

        # Sentence ending punctuation by language
        if language == "ja":
            sentence_ends = {"。", "！", "？", ".", "!", "?"}
        else:
            sentence_ends = {".", "!", "?"}

        # Find the last sentence boundary
        last_boundary = -1
        for i in range(len(text) - 1, -1, -1):
            if text[i] in sentence_ends:
                last_boundary = i
                break

        if last_boundary == -1:
            # No complete sentence found
            return "", text

        # Split into complete and partial
        complete = text[:last_boundary + 1].strip()
        partial = text[last_boundary + 1:].strip()

        return complete, partial

    def _process_audio_chunk(self, audio: np.ndarray) -> None:
        """Process an audio chunk through the pipeline.

        Args:
            audio: Audio chunk as numpy array.
        """
        try:
            self._stats["chunks_processed"] += 1
            console.print(f"[dim]Processing chunk {self._stats['chunks_processed']} ({len(audio)} samples)[/dim]")

            # Transcribe
            console.print("[dim]→ Transcribing...[/dim]")
            transcription = self.transcriber.transcribe(audio)

            if not transcription.text.strip():
                console.print("[dim]← No speech detected[/dim]")

                # Check if buffer has timed out (2 seconds of silence)
                if self._sentence_buffer and (time.time() - self._last_update_time) > 2.0:
                    console.print("[dim]→ Processing buffered partial sentence (timeout)...[/dim]")
                    self._process_complete_sentence(self._sentence_buffer, self._buffer_language)
                    self._sentence_buffer = ""
                    self._buffer_language = None
                return

            self._stats["transcriptions"] += 1
            if not self.config.translated_only:
                console.print(f"[dim]← Transcribed: \"{transcription.text}\" (lang: {transcription.language})[/dim]")

            # Skip non-English/non-Japanese languages (strictly EN→JA translation)
            if transcription.language not in ("en", "ja"):
                if not self.config.translated_only:
                    console.print(f"[yellow]⚠ Skipping non-EN/JA language: {transcription.language}[/yellow]")
                return

            # HYBRID APPROACH: Show transcription immediately (no translation yet)
            # Skip immediate display when translated_only is set
            if not self.config.translated_only:
                console.print("[dim]→ Showing immediate transcription...[/dim]")
                self.output.update(transcription.text, "")
                if self.g2_output:
                    self.g2_output.update(transcription.text, "")
                self._display_subtitle(transcription.text, "", transcription.language)

            # Determine source language
            source_lang = transcription.language
            if self.config.translator.source_lang:
                source_lang = self.config.translator.source_lang

            # Add to buffer for translation
            if self._buffer_language and self._buffer_language != source_lang:
                # Language changed, flush buffer first
                if self._sentence_buffer:
                    if not self.config.translated_only:
                        console.print("[dim]→ Processing buffered text (language change)...[/dim]")
                    self._process_complete_sentence(self._sentence_buffer, self._buffer_language)
                self._sentence_buffer = ""

            self._sentence_buffer += " " + transcription.text if self._sentence_buffer else transcription.text
            self._buffer_language = source_lang
            self._last_update_time = time.time()

            # Extract complete sentences
            complete_text, partial_text = self._extract_complete_sentences(self._sentence_buffer, source_lang)

            # Force flush if buffer is too long (prevent waiting forever)
            max_buffer_chars = 150  # ~2-3 sentences worth
            if len(self._sentence_buffer) > max_buffer_chars and not complete_text:
                if not self.config.translated_only:
                    console.print(f"[dim]→ Buffer size limit reached ({len(self._sentence_buffer)} chars), forcing translation...[/dim]")
                self._process_complete_sentence(self._sentence_buffer, source_lang)
                self._sentence_buffer = ""
            elif complete_text:
                if not self.config.translated_only:
                    console.print(f"[dim]→ Complete sentence detected: \"{complete_text}\"[/dim]")
                self._process_complete_sentence(complete_text, source_lang)
                self._sentence_buffer = partial_text
            else:
                # Translate buffered partial text (progressive translation)
                if not self.config.translated_only:
                    console.print(f"[dim]← Buffering partial: \"{self._sentence_buffer}\" ({len(self._sentence_buffer)} chars)[/dim]")

                # Skip progressive translation for SOV target languages (EN→JA, EN→KO, etc.)
                # SOV languages need the verb at the end, so partial translation is unreliable
                target_lang = self.config.translator.target_lang
                sov_languages = {"ja", "ko"}  # Japanese, Korean use SOV word order
                is_sov_translation = (source_lang == "en" and target_lang in sov_languages)

                # Only do progressive translation if buffer is substantial AND not translating to SOV
                min_progressive_chars = 80  # ~1-2 sentences minimum for SOV
                if (self._sentence_buffer and
                    source_lang == "en" and
                    self.config.translator.enabled and
                    (not is_sov_translation or len(self._sentence_buffer) >= min_progressive_chars)):

                    if not self.config.translated_only:
                        console.print(f"[dim]→ Translating partial buffer...[/dim]")

                    # Use context for partial translation too
                    context_text = self._build_translation_context(self._sentence_buffer)

                    partial_translation = self.translator.translate(
                        context_text,
                        source_lang=source_lang,
                        target_lang=target_lang,
                    )

                    # Extract new part
                    translated_partial = self._extract_new_translation(
                        partial_translation.translated_text,
                        self._sentence_buffer
                    )

                    if not self.config.translated_only:
                        console.print(f"[dim]← Partial translation: \"{translated_partial}\"[/dim]")
                    # Update output with partial translation
                    self.output.update(self._sentence_buffer, translated_partial)
                    if self.g2_output:
                        self.g2_output.update(self._sentence_buffer, translated_partial)
                elif is_sov_translation:
                    if not self.config.translated_only:
                        console.print(f"[dim]  (waiting for more context before translating to SOV language)[/dim]")

        except Exception as e:
            self._stats["errors"] += 1
            console.print(f"[red]Error processing audio: {e}[/red]")
            import traceback
            console.print(f"[red]{traceback.format_exc()}[/red]")

    def _process_complete_sentence(self, text: str, source_lang: str) -> None:
        """Process a complete sentence for translation and output.

        Args:
            text: Complete sentence text.
            source_lang: Source language code.
        """
        # Translate if enabled
        if self.config.translator.enabled and source_lang == "en":
            target_lang = self.config.translator.target_lang

            # Build context from recent translations
            context_text = self._build_translation_context(text)

            if not self.config.translated_only:
                console.print(f"[dim]→ Translating {source_lang} → {target_lang}...[/dim]")
                if context_text and context_text != text:
                    console.print(f"[dim]  (with {len(self._translation_context)} sentences of context)[/dim]")

            translation = self.translator.translate(
                context_text,  # Translate with context
                source_lang=source_lang,
                target_lang=target_lang,
            )
            self._stats["translations"] += 1

            # Extract only the new sentence translation (context might translate multi-sentence)
            translated_text = self._extract_new_translation(translation.translated_text, text)
            if not self.config.translated_only:
                console.print(f"[dim]← Translated: \"{translated_text}\"[/dim]")

            # Update context history
            self._update_translation_context(text, translated_text)
        else:
            # Skip translation for Japanese or other languages
            if source_lang != "en" and not self.config.translated_only:
                console.print(f"[dim]← Skipping translation (source is {source_lang}, not English)[/dim]")
            translated_text = ""

        # Output
        if not self.config.translated_only:
            console.print("[dim]→ Updating output...[/dim]")
        self.output.update(text, translated_text)
        if self.g2_output:
            self.g2_output.update(text, translated_text)

        # Display in console
        self._display_subtitle(text, translated_text, source_lang)

    def _display_subtitle(
        self, original: str, translated: str, language: str
    ) -> None:
        """Display subtitle in console.

        Args:
            original: Original text.
            translated: Translated text.
            language: Detected language.
        """
        # Skip empty display (immediate transcription already shown via output.update)
        if not translated:
            return

        # Display subtitle
        text = Text()
        if not self.config.translated_only:
            text.append(f"[{language}] ", style="dim")
            text.append(original, style="white")
            text.append("\n")
            text.append("→ ", style="dim")
        text.append(translated, style="cyan bold")
        console.print(Panel(text, border_style="cyan", padding=(0, 1)))
        console.print()  # Add spacing

    def _processing_loop(self) -> None:
        """Main processing loop running in a separate thread."""
        while not self._stop_event.is_set():
            # Get audio chunk with timeout
            audio = self.audio_capture.get_chunk(timeout=0.5)

            if audio is not None:
                self._process_audio_chunk(audio)
            elif self.output.should_clear():
                self.output.clear()

    def _preload_models(self) -> None:
        """Preload all models to avoid delays during processing."""
        import numpy as np

        console.print("[bold yellow]Preloading models...[/bold yellow]")
        console.print()

        # Preload Whisper model
        with console.status(f"[bold blue]Loading Whisper model ({self.config.transcriber.model})...", spinner="dots"):
            # Create dummy audio to trigger model loading
            dummy_audio = np.zeros(16000, dtype=np.float32)
            self.transcriber.transcribe(dummy_audio)
        console.print(f"[green]✓ Whisper model loaded ({self.config.transcriber.model})[/green]")

        # Preload translation model if enabled
        if self.config.translator.enabled:
            with console.status("[bold blue]Loading NLLB translation model...", spinner="dots"):
                # Trigger model loading with a test translation
                self.translator.translate(
                    "Hello",
                    source_lang="en",
                    target_lang=self.config.translator.target_lang,
                )
            console.print(f"[green]✓ Translation model loaded (en → {self.config.translator.target_lang})[/green]")

        console.print()
        console.print("[bold green]All models loaded and ready![/bold green]")
        console.print()

    def start(self) -> None:
        """Start the pipeline."""
        if self._running:
            return

        console.print("[bold green]Starting Live Translator...[/bold green]")
        console.print()

        # Preload models first
        self._preload_models()

        # Start components
        console.print("[dim]→ Starting output servers...[/dim]")
        self.output.start()
        if self.g2_output:
            console.print("[dim]→ Starting G2 glasses output...[/dim]")
            self.g2_output.start()
            console.print()
            console.print("[bold yellow]G2 glasses connected![/bold yellow]")
            console.print("[yellow]Activate Even AI on your glasses, then press Enter to start...[/yellow]")
            input()
        console.print("[dim]→ Starting audio capture...[/dim]")
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
        console.print("[bold green]✓ Ready - Listening for audio...[/bold green]")
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
        if self.g2_output:
            self.g2_output.stop()
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
