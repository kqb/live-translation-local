"""CLI module using Click for command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table


console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="live-translator")
def cli() -> None:
    """Live Translator for OBS Studio.

    Real-time speech transcription and translation for live streaming.
    """
    pass


@cli.command()
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file (default: config.yaml)",
)
@click.option(
    "-m",
    "--model",
    type=click.Choice(["tiny", "base", "small", "medium", "large-v2", "large-v3"]),
    help="Whisper model size",
)
@click.option(
    "-s",
    "--source",
    "source_lang",
    help="Source language code (e.g., en, es, fr)",
)
@click.option(
    "-t",
    "--target",
    "target_lang",
    help="Target language code (e.g., en, es, fr)",
)
@click.option(
    "-d",
    "--device",
    type=int,
    help="Audio input device index",
)
@click.option(
    "--no-translate",
    is_flag=True,
    help="Disable translation (transcription only)",
)
@click.option(
    "-b",
    "--backend",
    type=click.Choice(["whisper-cpp", "faster-whisper"]),
    default="whisper-cpp",
    help="Whisper backend (default: whisper-cpp for Apple Silicon)",
)
@click.option(
    "--translated-only",
    is_flag=True,
    help="Only show translated text in console (hide original)",
)
@click.option(
    "--audio-source",
    type=click.Choice(["microphone", "omi"]),
    help="Audio input source (default: from config)",
)
def run(
    config_path: Optional[Path],
    model: Optional[str],
    source_lang: Optional[str],
    target_lang: Optional[str],
    device: Optional[int],
    no_translate: bool,
    backend: str,
    translated_only: bool,
    audio_source: Optional[str],
) -> None:
    """Start the live translator.

    Examples:

        # Start with default config
        live-translator run

        # Use custom config file
        live-translator run -c myconfig.yaml

        # Override target language
        live-translator run -t fr

        # Use specific audio device
        live-translator run -d 2

        # Transcription only (no translation)
        live-translator run --no-translate
    """
    from .pipeline import PipelineConfig, Pipeline, load_config

    # Load base config
    config = load_config(config_path)

    # Apply command-line overrides
    if model:
        config.transcriber.model = model
    if source_lang:
        config.transcriber.language = source_lang
        config.translator.source_lang = source_lang
    if target_lang:
        config.translator.target_lang = target_lang
    if device is not None:
        config.audio.device = device
    if no_translate:
        config.translator.enabled = False
    if backend:
        config.backend = backend
    if translated_only:
        config.translated_only = True
    if audio_source:
        config.audio_source = audio_source

    # Display configuration
    console.print()
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Audio source: {config.audio_source}")
    if config.audio_source == "omi":
        console.print(f"  Omi MAC: {config.omi.mac_address or 'auto-discover'}")
    else:
        console.print(f"  Audio device: {config.audio.device or 'default'}")
    console.print(f"  Backend: {config.backend}")
    console.print(f"  Whisper model: {config.transcriber.model}")
    console.print(f"  Source language: {config.transcriber.language or 'auto-detect'}")
    console.print(f"  Target language: {config.translator.target_lang}")
    console.print(f"  Translation: {'enabled' if config.translator.enabled else 'disabled'}")
    console.print()

    # Run pipeline
    pipeline = Pipeline(config)
    pipeline.run()


@cli.command()
@click.option(
    "--ble",
    is_flag=True,
    help="Scan for BLE devices (Omi wearables)",
)
def devices(ble: bool) -> None:
    """List available audio input devices."""
    if ble:
        # Scan for BLE devices (Omi)
        from .omi_input import list_omi_devices

        console.print()
        console.print("[bold]Scanning for BLE devices (Omi wearables)...[/bold]")
        console.print()
        list_omi_devices()
        console.print()
        console.print("[dim]Copy the MAC address to config.yaml under audio.omi.mac_address[/dim]")
        console.print("[dim]Example: mac_address: \"XX:XX:XX:XX:XX:XX\"[/dim]")
        return

    # List microphone devices
    from .audio_capture import list_devices

    devices = list_devices()

    if not devices:
        console.print("[yellow]No audio input devices found.[/yellow]")
        return

    table = Table(title="Audio Input Devices")
    table.add_column("Index", style="cyan", justify="right")
    table.add_column("Name", style="white")
    table.add_column("Channels", justify="center")
    table.add_column("Sample Rate", justify="right")
    table.add_column("Default", justify="center")

    for device in devices:
        default_marker = "[green]✓[/green]" if device["is_default"] else ""
        table.add_row(
            str(device["index"]),
            device["name"],
            str(device["channels"]),
            f"{device['sample_rate']:.0f} Hz",
            default_marker,
        )

    console.print(table)
    console.print()
    console.print("[dim]Use -d/--device INDEX to select a device[/dim]")


@cli.command()
def languages() -> None:
    """List available translation language pairs."""
    from .translator import get_supported_languages, get_language_code_map

    languages = get_supported_languages()
    code_map = get_language_code_map()

    table = Table(title="Supported Languages")
    table.add_column("Code", style="cyan")
    table.add_column("Language", style="white")
    table.add_column("NLLB Code", style="dim")

    for code, name in sorted(languages.items(), key=lambda x: x[1]):
        nllb_code = code_map.get(code, "")
        table.add_row(code, name, nllb_code)

    console.print(table)
    console.print()
    console.print("[dim]Use -s/--source and -t/--target to specify languages[/dim]")
    console.print("[dim]Example: live-translator run -s en -t es[/dim]")


@cli.command()
@click.argument("source_lang")
@click.argument("target_lang")
def install(source_lang: str, target_lang: str) -> None:
    """Download and install translation model for a language pair.

    This downloads the NLLB-200 model and caches it locally.
    The model supports translation between any pair of 200+ languages.

    Examples:

        live-translator install en es

        live-translator install en fr
    """
    from .translator import Translator, TranslatorConfig, is_language_supported

    # Validate languages
    if not is_language_supported(source_lang):
        console.print(f"[red]Unsupported source language: {source_lang}[/red]")
        console.print("Run 'live-translator languages' to see supported languages.")
        return

    if not is_language_supported(target_lang):
        console.print(f"[red]Unsupported target language: {target_lang}[/red]")
        console.print("Run 'live-translator languages' to see supported languages.")
        return

    console.print(f"[bold]Installing translation model...[/bold]")
    console.print(f"  Source: {source_lang}")
    console.print(f"  Target: {target_lang}")
    console.print()

    with console.status("[bold blue]Downloading model...", spinner="dots"):
        try:
            config = TranslatorConfig(
                source_lang=source_lang,
                target_lang=target_lang,
            )
            translator = Translator(config)

            # Trigger model download by doing a test translation
            result = translator.translate(
                "Hello, world!",
                source_lang=source_lang,
                target_lang=target_lang,
            )

            console.print("[green]Model installed successfully![/green]")
            console.print()
            console.print("[dim]Test translation:[/dim]")
            console.print(f"  Original: Hello, world!")
            console.print(f"  Translated: {result.translated_text}")

        except Exception as e:
            console.print(f"[red]Error installing model: {e}[/red]")


@cli.command()
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
def test(config_path: Optional[Path]) -> None:
    """Test all components without starting the full pipeline.

    This command tests:
    - Audio capture (records a short clip)
    - Whisper transcription
    - NLLB translation
    - OBS output (file and WebSocket)
    """
    import time
    import numpy as np
    from .pipeline import load_config
    from .audio_capture import AudioCapture
    from .transcriber import Transcriber
    from .translator import Translator
    from .obs_output import OBSOutput

    config = load_config(config_path)

    console.print("[bold]Testing Live Translator Components[/bold]")
    console.print()

    # Test 1: Audio capture
    console.print("[cyan]1. Testing audio capture...[/cyan]")
    try:
        audio_capture = AudioCapture(config.audio)
        audio_capture.start()
        time.sleep(1.5)  # Record for 1.5 seconds
        audio_capture.stop()

        # Try to get any buffered audio
        audio = audio_capture.flush_buffer()
        if audio is not None and len(audio) > 0:
            console.print(f"   [green]✓[/green] Captured {len(audio)} samples ({len(audio)/config.audio.sample_rate:.2f}s)")
        else:
            console.print("   [yellow]⚠[/yellow] No audio captured (check microphone)")
    except Exception as e:
        console.print(f"   [red]✗[/red] Error: {e}")

    console.print()

    # Test 2: Transcriber
    console.print("[cyan]2. Testing Whisper transcriber...[/cyan]")
    try:
        with console.status("[dim]Loading model...", spinner="dots"):
            transcriber = Transcriber(config.transcriber)

            # Generate test audio (silence with some noise)
            test_audio = np.random.randn(16000 * 2).astype(np.float32) * 0.01

            result = transcriber.transcribe(test_audio)

        console.print(f"   [green]✓[/green] Model loaded: {config.transcriber.model}")
        console.print(f"   [green]✓[/green] Detected language: {result.language} ({result.confidence:.1%})")
    except Exception as e:
        console.print(f"   [red]✗[/red] Error: {e}")

    console.print()

    # Test 3: Translator
    console.print("[cyan]3. Testing NLLB translator...[/cyan]")
    try:
        with console.status("[dim]Loading model...", spinner="dots"):
            translator = Translator(config.translator)
            result = translator.translate(
                "Hello, how are you today?",
                source_lang="en",
                target_lang=config.translator.target_lang,
            )

        console.print(f"   [green]✓[/green] Model loaded")
        console.print(f"   [green]✓[/green] Translation: {result.original_text} → {result.translated_text}")
    except Exception as e:
        console.print(f"   [red]✗[/red] Error: {e}")

    console.print()

    # Test 4: OBS Output
    console.print("[cyan]4. Testing OBS output...[/cyan]")
    try:
        output = OBSOutput(config.output)
        output.start()

        # Write test subtitle
        output.update("Test original text", "Test translated text")
        time.sleep(0.5)

        output.stop()

        console.print(f"   [green]✓[/green] Text file: {config.output.text_file}")
        if config.output.websocket_enabled:
            console.print(f"   [green]✓[/green] WebSocket server: ws://localhost:{config.output.websocket_port}")
            console.print(f"   [green]✓[/green] HTTP overlay: http://localhost:{config.output.http_port}/overlay")
    except Exception as e:
        console.print(f"   [red]✗[/red] Error: {e}")

    console.print()
    console.print("[bold green]All tests completed![/bold green]")
    console.print()
    console.print("Run 'live-translator run' to start the live translator.")


@cli.command()
def config() -> None:
    """Show the default configuration template."""
    default_config = """# Live Translator Configuration

audio:
  device: null        # null = default mic, or specify device index/name
  sample_rate: 16000  # 16kHz for Whisper
  chunk_duration: 3.0 # seconds per audio chunk

whisper:
  model: "base"       # tiny, base, small, medium, large-v2, large-v3
  device: "auto"      # cpu, cuda, auto
  compute_type: "int8" # int8, float16, float32
  language: null      # null = auto-detect, or specify language code (en, es, fr, etc.)

  # Anti-hallucination settings
  no_speech_threshold: 0.6  # Higher = more aggressive filtering (0.0-1.0)
  logprob_threshold: -1.0   # Higher = filter low-confidence text
  compression_ratio_threshold: 2.4  # Lower = filter repetitive text

translation:
  enabled: true
  source_lang: null   # null = auto from whisper detection
  target_lang: "es"   # target language code (es, fr, de, ja, zh, etc.)

output:
  text_file: "./subtitles.txt"  # Path for OBS Text (GDI+) source
  websocket_enabled: true
  websocket_port: 8765
  http_port: 8766
  max_lines: 2        # Maximum lines to display (legacy mode)
  clear_after: 5.0    # Clear subtitles after N seconds of silence (legacy mode)

  # YouTube-style scrolling subtitles
  scrolling_mode: true  # Enable continuous scrolling subtitles
  history_lines: 10     # Number of subtitle lines to keep visible
"""
    console.print(default_config)
    console.print()
    console.print("[dim]Save this to 'config.yaml' and customize as needed.[/dim]")


if __name__ == "__main__":
    cli()
