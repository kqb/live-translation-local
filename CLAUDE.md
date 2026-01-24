# Live Translator for OBS Studio - Project Context

This document provides context for Claude Code when working with this project.

## Project Overview

**Live Translator for OBS Studio** is a real-time speech transcription and translation tool for live streaming. It captures microphone audio, transcribes it using Whisper, translates it with NLLB-200, and displays subtitles in OBS Studio.

**Key Technologies:**
- **faster-whisper**: CTranslate2-based Whisper implementation (4x faster than OpenAI Whisper)
- **NLLB-200**: Meta's "No Language Left Behind" translation model (200+ languages)
- **sounddevice**: Audio capture from microphone
- **websockets & aiohttp**: Real-time communication with OBS Browser Source
- **Click**: CLI framework with Rich for beautiful terminal output

**Project Type:** Python 3.10+ CLI application with async/await architecture

## Architecture

### Core Pipeline Flow

```
Audio Capture → Transcription → Translation → OBS Output
  (sounddevice)   (Whisper)      (NLLB-200)    (file/websocket/HTTP)
```

**Key Components:**

1. **AudioCapture** ([src/audio_capture.py](src/audio_capture.py))
   - Captures microphone audio in configurable chunks (default: 3 seconds)
   - Uses sounddevice with 16kHz sample rate for Whisper compatibility
   - Runs in separate thread, queues audio chunks

2. **Transcriber** ([src/transcriber.py](src/transcriber.py))
   - Uses faster-whisper with CTranslate2 backend
   - Supports GPU (CUDA) or CPU execution
   - Auto-detects language or uses specified language
   - Returns transcribed text with detected language

3. **Translator** ([src/translator.py](src/translator.py))
   - Uses Meta's NLLB-200 model via transformers library
   - Handles 200+ language pairs
   - Can be disabled for transcription-only mode
   - Caches model in memory for fast translation

4. **OBSOutput** ([src/obs_output.py](src/obs_output.py))
   - Multiple output methods: text file, WebSocket, HTTP server
   - Text file: Written for OBS Text (GDI+) source
   - WebSocket: Real-time updates to browser source (port 8765)
   - HTTP: Serves HTML overlay with styled subtitles (port 8766)
   - Auto-clears subtitles after configurable silence period

5. **Pipeline** ([src/pipeline.py](src/pipeline.py))
   - Orchestrates all components
   - Manages threading and async tasks
   - Handles graceful shutdown on Ctrl+C
   - Loads configuration from YAML

6. **CLI** ([src/cli.py](src/cli.py))
   - Click-based command interface
   - Commands: `run`, `devices`, `languages`, `install`, `test`, `config`
   - Rich terminal output for status and logs

### Configuration

Configuration via [config.yaml](config.yaml):
- **Audio**: Device selection, sample rate, chunk duration
- **Whisper**: Model size (tiny/base/small/medium/large-v2/large-v3), device (CPU/CUDA), compute type
- **Translation**: Enable/disable, source/target languages
- **Output**: Text file path, WebSocket/HTTP ports, max lines, clear timeout

## Code Conventions

### Python Style

**Formatting:**
- **Black**: Line length 100, target Python 3.10+
- **Ruff**: Linting with line length 100, target Python 3.10+
- Run `black .` before committing
- Run `ruff check . --fix` to auto-fix issues

**Type Hints:**
- Use modern type hints: `dict[str, Any]` instead of `Dict[str, Any]`
- Use `from __future__ import annotations` for forward references
- All public functions should have type hints

**Imports:**
- Group imports: standard library, third-party, local modules
- Use absolute imports from `src.*` package

**Docstrings:**
- Google-style docstrings for all public functions/classes
- Include Args, Returns, Raises sections where applicable

### Dataclasses

All configuration objects use `@dataclass` with:
- Type hints for all fields
- `default_factory` for mutable defaults
- `from_dict()` classmethod for YAML loading
- `to_dict()` method for serialization

Example pattern:
```python
@dataclass
class MyConfig:
    field1: str = "default"
    field2: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MyConfig":
        return cls(
            field1=data.get("field1", "default"),
            field2=data.get("field2", [])
        )
```

### Async/Await

- Use `async def` for I/O-bound operations (WebSocket, HTTP, file writes)
- Use `asyncio.Queue` for thread-safe communication
- Run async tasks with `asyncio.create_task()`
- Handle cleanup in try/finally blocks

### Error Handling

- Use Rich console for user-facing errors: `console.print("[red]Error:[/red] message")`
- Log technical details for debugging
- Graceful degradation where possible (e.g., translation failures)

### Threading

- Audio capture runs in `threading.Thread` (blocking sounddevice API)
- Main pipeline uses `asyncio` for I/O operations
- Use `threading.Event` for shutdown coordination
- Always set daemon threads appropriately

## Development Workflow

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (if using)
pre-commit install
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_transcriber.py -v
```

### Local Development

```bash
# Run from source
python -m src.cli run

# Or use installed entry point
live-translator run

# List available audio devices
live-translator devices

# Use custom config
live-translator run -c custom-config.yaml

# Other useful commands
live-translator languages        # List supported languages
live-translator test            # Test all components
live-translator config          # Show config template
```

### Common Tasks

**Adding a new Whisper model:**
1. Update `TranscriberConfig.model` type hint in [src/transcriber.py](src/transcriber.py)
2. Update CLI choices in [src/cli.py](src/cli.py) (line 37)
3. Test with: `live-translator run -m new-model`

**Adding a new output method:**
1. Add configuration to `OutputConfig` in [src/obs_output.py](src/obs_output.py)
2. Implement output handler method in `OBSOutput` class
3. Start/stop in `start()` and `stop()` methods
4. Update [config.yaml](config.yaml) with new options

**Optimizing performance:**
- **GPU**: Ensure CUDA 12+ and cuDNN 9+ installed for CTranslate2
- **Chunk duration**: Smaller chunks (1-2s) = lower latency but more processing
- **Model size**: `base` balances speed and accuracy; `tiny` for low-end hardware
- **Compute type**: `int8` for CPU, `float16` for GPU

## OBS Studio Integration

### Text File Source (Simple)

1. Add **Text (GDI+)** source in OBS
2. Check "Read from file"
3. Browse to `subtitles.txt` (or path in config)
4. Configure font, size, color

### Browser Source (Styled Overlay)

1. Add **Browser** source in OBS
2. URL: `http://localhost:8766`
3. Width: 1920, Height: 200 (adjust as needed)
4. Check "Shutdown source when not visible" for performance
5. Overlay includes fade animations and styling

### WebSocket (Advanced)

For custom HTML overlays:
1. Create HTML file with WebSocket client
2. Connect to `ws://localhost:8765`
3. Receive JSON: `{"text": "subtitle text", "timestamp": 123.45}`
4. Add as Browser source in OBS

## Dependencies

### Core Runtime
- **faster-whisper** (1.0.0+): Whisper transcription
- **transformers** (4.35.0+): NLLB-200 translation model loading
- **ctranslate2** (4.0.0+): Fast inference engine
- **sounddevice** (0.4.6+): Audio capture
- **numpy** (1.24.0+): Audio buffer manipulation
- **websockets** (12.0+): WebSocket server for OBS
- **aiohttp** (3.9.0+): HTTP server for browser overlay
- **click** (8.1.0+): CLI framework
- **pyyaml** (6.0.0+): Configuration loading
- **rich** (13.0.0+): Terminal output formatting

### Dev Dependencies
- **pytest** (7.0.0+): Testing framework
- **pytest-asyncio** (0.21.0+): Async test support
- **black** (23.0.0+): Code formatting
- **ruff** (0.1.0+): Fast Python linter

### Optional System Requirements
- **CUDA 12+ and cuDNN 9+**: For GPU acceleration (faster-whisper requirement)
- **PyTorch with CUDA**: For NLLB-200 GPU translation (optional)

## Known Issues & Limitations

1. **First run is slow**: Downloads Whisper model (~150MB for base) and NLLB-200 model (~1.2GB)
2. **Memory usage**: Large models (large-v3, NLLB-200) require 4-8GB RAM
3. **CUDA compatibility**: CTranslate2 requires CUDA 12+, older CUDA versions not supported
4. **Language detection**: Auto-detection may be inaccurate for short audio chunks
5. **Translation accuracy**: NLLB-200 quality varies by language pair

## Debugging Tips

### Audio Issues
```bash
# List audio devices and indices
live-translator devices

# Test with specific device
# Edit config.yaml: audio.device: <device_index>
# Or use CLI flag: live-translator run -d <device_index>
```

### Whisper Model Issues
```bash
# Test all components including model loading
live-translator test

# Force model download manually
python -c "from faster_whisper import WhisperModel; WhisperModel('base')"
```

### GPU Issues
```python
# Check CUDA availability for PyTorch (NLLB)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check CTranslate2 GPU support
python -c "import ctranslate2; print(ctranslate2.get_supported_compute_types('cuda'))"
```

### Logging
Enable debug logging by setting environment variable:
```bash
export LOG_LEVEL=DEBUG
live-translator run
```

## Performance Benchmarks

Approximate performance on different hardware (base model, 3s chunks):

| Hardware | Transcription Time | Translation Time | Total Latency |
|----------|-------------------|------------------|---------------|
| CPU (8 cores) | 1.5-2s | 0.5-1s | ~2-3s |
| GPU (RTX 3060) | 0.3-0.5s | 0.1-0.2s | ~0.5-1s |
| Apple M1 | 0.8-1.2s | 0.3-0.5s | ~1-2s |

**Optimization tips:**
- Use GPU for both Whisper and NLLB (10x faster)
- Reduce chunk duration for lower latency (increases CPU/GPU load)
- Use `int8` quantization on CPU, `float16` on GPU
- Disable translation if only transcription needed

## Future Roadmap

**Planned features:**
- [ ] Multiple simultaneous speakers (diarization)
- [ ] Custom vocabulary/terminology for domain-specific content
- [ ] Subtitle history/scrolling view
- [ ] Export transcripts to SRT/VTT files
- [ ] Real-time translation confidence scores
- [ ] Integration with YouTube/Twitch captions APIs
- [ ] macOS/Windows installers with GUI

## Related Resources

- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [NLLB-200 Model Card](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [OBS Studio Docs](https://obsproject.com/docs/)
- [CTranslate2 Documentation](https://opennmt.net/CTranslate2/)

---

**Last Updated:** 2026-01-23
**Project Version:** 0.1.0
**Python Version:** 3.10+
