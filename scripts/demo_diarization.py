#!/usr/bin/env python3
"""Test speaker diarization on a single audio file."""

import sys
from pathlib import Path

import soundfile as sf
import yaml
from rich.console import Console
from rich.table import Table

from src.diarization import Diarizer, DiarizationConfig
from src.transcriber import Transcriber, TranscriberConfig

console = Console()


def test_diarization(audio_path: Path, config_path: Path):
    """Test diarization on a single audio file.

    Args:
        audio_path: Path to audio file
        config_path: Path to config.yaml
    """
    console.print(f"\n[bold cyan]Testing Speaker Diarization[/bold cyan]")
    console.print(f"Audio file: {audio_path}")
    console.print(f"File size: {audio_path.stat().st_size / 1024:.1f} KB\n")

    # Load config
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    # Load audio
    console.print("[dim]Loading audio file...[/dim]")
    audio_data, sample_rate = sf.read(str(audio_path))

    # Convert to mono if stereo
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    duration = len(audio_data) / sample_rate
    console.print(f"Duration: {duration:.1f} seconds\n")

    # Initialize diarizer
    console.print("[dim]Initializing diarizer (downloading models if needed)...[/dim]")
    diarizer_config = DiarizationConfig(
        enabled=True,
        min_speakers=config_data["diarization"]["min_speakers"],
        max_speakers=config_data["diarization"]["max_speakers"],
        hf_token=config_data["diarization"]["hf_token"],
    )
    diarizer = Diarizer(diarizer_config)

    # Run diarization
    console.print("[bold green]Running speaker diarization...[/bold green]")
    with console.status("[bold green]Detecting speakers...", spinner="dots"):
        speaker_segments = diarizer.diarize(audio_data, sample_rate)

    if not speaker_segments:
        console.print("[yellow]No speaker segments detected![/yellow]")
        return

    console.print(f"\n[bold]Found {len(speaker_segments)} speaker segments[/bold]\n")

    # Display speaker segments
    table = Table(title="Speaker Segments")
    table.add_column("Speaker", style="cyan")
    table.add_column("Start (s)", style="green")
    table.add_column("End (s)", style="green")
    table.add_column("Duration (s)", style="yellow")

    for seg in speaker_segments:
        duration_seg = seg.end - seg.start
        table.add_row(
            seg.speaker,
            f"{seg.start:.2f}",
            f"{seg.end:.2f}",
            f"{duration_seg:.2f}"
        )

    console.print(table)

    # Summary by speaker
    speakers = {}
    for seg in speaker_segments:
        if seg.speaker not in speakers:
            speakers[seg.speaker] = {"count": 0, "total_time": 0}
        speakers[seg.speaker]["count"] += 1
        speakers[seg.speaker]["total_time"] += seg.end - seg.start

    console.print("\n[bold]Speaker Summary:[/bold]")
    summary_table = Table()
    summary_table.add_column("Speaker", style="cyan")
    summary_table.add_column("Segments", style="green")
    summary_table.add_column("Total Time (s)", style="yellow")
    summary_table.add_column("Percentage", style="magenta")

    for speaker, stats in sorted(speakers.items()):
        percentage = (stats["total_time"] / duration) * 100
        summary_table.add_row(
            speaker,
            str(stats["count"]),
            f"{stats['total_time']:.2f}",
            f"{percentage:.1f}%"
        )

    console.print(summary_table)

    # Now test with transcription
    console.print("\n[bold cyan]Testing Transcription with Speaker Labels[/bold cyan]")

    transcriber_config = TranscriberConfig(
        model=config_data["whisper"]["model"],
        device="auto",
    )
    transcriber = Transcriber(transcriber_config)

    console.print("[dim]Transcribing audio...[/dim]")
    with console.status("[bold green]Transcribing...", spinner="dots"):
        transcription = transcriber.transcribe(audio_data)

    # Get segments with timestamps
    segments = transcription.segments

    if not segments:
        console.print("[yellow]No transcription segments found![/yellow]")
        return

    # Assign speakers to transcription segments
    console.print("[dim]Assigning speakers to transcription segments...[/dim]")
    segments_with_speakers = diarizer.assign_speaker_to_segments(
        segments, speaker_segments
    )

    # Display transcription with speakers
    console.print("\n[bold]Transcription with Speaker Labels:[/bold]\n")
    trans_table = Table(show_header=True)
    trans_table.add_column("Time", style="dim", width=12)
    trans_table.add_column("Speaker", style="cyan", width=12)
    trans_table.add_column("Text", style="white")

    for seg in segments_with_speakers[:10]:  # Show first 10 segments
        time_str = f"{seg['start']:.1f}-{seg['end']:.1f}s"
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg["text"].strip()
        trans_table.add_row(time_str, speaker, text)

    console.print(trans_table)

    if len(segments_with_speakers) > 10:
        console.print(f"\n[dim]... and {len(segments_with_speakers) - 10} more segments[/dim]")

    console.print(f"\n[bold green]✓ Diarization test complete![/bold green]")


if __name__ == "__main__":
    # Allow file path as command line argument
    import sys
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
    else:
        # Default test file
        test_file = Path("/Users/yuzucchi/Library/Application Support/Voxal/Recording/2021-06-25 19-09-38.wav")
    config_file = Path(__file__).parent / "config.yaml"

    if not test_file.exists():
        console.print(f"[red]Error: Test file not found: {test_file}[/red]")
        sys.exit(1)

    if not config_file.exists():
        console.print(f"[red]Error: Config file not found: {config_file}[/red]")
        sys.exit(1)

    test_diarization(test_file, config_file)
