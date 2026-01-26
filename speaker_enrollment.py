#!/usr/bin/env python3
"""Enroll speakers for persistent recognition across sessions.

This tool allows you to:
1. List enrolled speakers
2. Enroll a new speaker from an audio sample
3. Remove a speaker
4. Test speaker recognition on an audio file
"""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
import soundfile as sf
import yaml

from src.speaker_recognition import SpeakerRecognizer, SpeakerDatabase
from src.diarization import Diarizer, DiarizationConfig

console = Console()


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path("config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


@click.group()
def cli():
    """Speaker enrollment and recognition management."""
    pass


@cli.command()
def list_speakers():
    """List all enrolled speakers."""
    config = load_config()
    db_path = Path(config["speaker_recognition"]["database_path"])

    if not db_path.exists():
        console.print("[yellow]No speakers enrolled yet.[/yellow]")
        return

    db = SpeakerDatabase(db_path)
    speakers = db.list_speakers()

    if not speakers:
        console.print("[yellow]No speakers enrolled.[/yellow]")
        return

    table = Table(title="Enrolled Speakers")
    table.add_column("Name", style="cyan")
    table.add_column("Samples", style="green")

    for speaker in speakers:
        table.add_row(speaker["name"], str(speaker["sample_count"]))

    console.print(table)
    console.print(f"\n[bold]{len(speakers)} speakers enrolled[/bold]")


@cli.command()
@click.argument("name")
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
def enroll(name: str, audio_file: Path):
    """Enroll a new speaker or add sample to existing speaker.

    \b
    Arguments:
        NAME: Speaker name (e.g., "Alice", "Bob")
        AUDIO_FILE: Path to audio file with speaker's voice

    \b
    Example:
        speaker enroll Alice ~/voice_samples/alice.wav
    """
    console.print(f"\n[bold cyan]Enrolling speaker: {name}[/bold cyan]\n")

    # Load config
    config = load_config()

    # Initialize diarizer (needed for embedding extraction)
    console.print("[dim]Initializing diarization models...[/dim]")
    diarizer_config = DiarizationConfig(
        enabled=True,
        hf_token=config["diarization"]["hf_token"],
    )
    diarizer = Diarizer(diarizer_config)

    # Initialize speaker recognizer
    db_path = Path(config["speaker_recognition"]["database_path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)

    recognizer = SpeakerRecognizer(
        diarizer=diarizer,
        db_path=db_path,
        recognition_threshold=config["speaker_recognition"]["recognition_threshold"],
    )

    # Load audio
    console.print(f"[dim]Loading audio from {audio_file}...[/dim]")
    audio_data, sample_rate = sf.read(str(audio_file))

    # Convert to mono if stereo
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # Enroll speaker
    console.print("[dim]Extracting voice embedding...[/dim]")
    with console.status("[bold green]Enrolling speaker...", spinner="dots"):
        success = recognizer.enroll_speaker(name, audio_data, sample_rate)

    if success:
        console.print(f"\n[bold green]✓ Successfully enrolled {name}![/bold green]")

        # Show updated count
        speaker_info = [s for s in recognizer.db.list_speakers() if s["name"] == name]
        if speaker_info:
            console.print(f"[dim]Total samples for {name}: {speaker_info[0]['sample_count']}[/dim]")
    else:
        console.print(f"\n[bold red]✗ Failed to enroll {name}[/bold red]")
        console.print("[yellow]Make sure the audio file contains clear speech.[/yellow]")


@cli.command()
@click.argument("name")
@click.confirmation_option(prompt="Are you sure you want to remove this speaker?")
def remove(name: str):
    """Remove an enrolled speaker.

    \b
    Arguments:
        NAME: Speaker name to remove
    """
    config = load_config()
    db_path = Path(config["speaker_recognition"]["database_path"])

    if not db_path.exists():
        console.print("[yellow]No speakers database found.[/yellow]")
        return

    db = SpeakerDatabase(db_path)

    if db.remove_speaker(name):
        console.print(f"[bold green]✓ Removed speaker: {name}[/bold green]")
    else:
        console.print(f"[bold red]✗ Speaker not found: {name}[/bold red]")


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
def recognize(audio_file: Path):
    """Test speaker recognition on an audio file.

    \b
    Arguments:
        AUDIO_FILE: Path to audio file to test

    \b
    Example:
        speaker recognize ~/test.wav
    """
    console.print(f"\n[bold cyan]Testing Speaker Recognition[/bold cyan]\n")

    # Load config
    config = load_config()

    # Check if any speakers are enrolled
    db_path = Path(config["speaker_recognition"]["database_path"])
    if not db_path.exists():
        console.print("[yellow]No speakers enrolled yet. Use 'enroll' command first.[/yellow]")
        return

    db = SpeakerDatabase(db_path)
    if not db.speakers:
        console.print("[yellow]No speakers enrolled yet. Use 'enroll' command first.[/yellow]")
        return

    # Initialize diarizer
    console.print("[dim]Initializing models...[/dim]")
    diarizer_config = DiarizationConfig(
        enabled=True,
        hf_token=config["diarization"]["hf_token"],
    )
    diarizer = Diarizer(diarizer_config)

    # Initialize recognizer
    recognizer = SpeakerRecognizer(
        diarizer=diarizer,
        db_path=db_path,
        recognition_threshold=config["speaker_recognition"]["recognition_threshold"],
    )

    # Load audio
    console.print(f"[dim]Loading audio from {audio_file}...[/dim]")
    audio_data, sample_rate = sf.read(str(audio_file))

    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # Extract embedding and recognize
    console.print("[dim]Extracting voice embedding...[/dim]")
    with console.status("[bold green]Recognizing speaker...", spinner="dots"):
        embedding = recognizer.extract_embedding(audio_data, sample_rate)

        if embedding is None:
            console.print("[bold red]✗ Failed to extract embedding[/bold red]")
            return

        recognized_name = recognizer.recognize_speaker(embedding)

    # Show results
    if recognized_name:
        console.print(f"\n[bold green]✓ Recognized speaker: {recognized_name}[/bold green]")
    else:
        console.print(f"\n[bold yellow]? Unknown speaker[/bold yellow]")
        console.print("[dim]Speaker not in database or similarity below threshold[/dim]")

    # Show enrolled speakers for reference
    console.print(f"\n[dim]Enrolled speakers: {', '.join(s['name'] for s in recognizer.db.list_speakers())}[/dim]")


if __name__ == "__main__":
    cli()
