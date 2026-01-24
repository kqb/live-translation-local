# Architecture Documentation

This document provides a detailed technical overview of the Live Translator for OBS Studio architecture.

## System Overview

Live Translator is a real-time audio processing pipeline that captures microphone input, transcribes it using Whisper, translates it with NLLB-200, and outputs synchronized subtitles to OBS Studio.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Live Translator Pipeline                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐      │
│  │    Audio     │      │              │      │              │      │
│  │   Capture    │─────▶│ Transcriber  │─────▶│  Translator  │      │
│  │ (sounddevice)│      │  (Whisper)   │      │  (NLLB-200)  │      │
│  └──────────────┘      └──────────────┘      └──────────────┘      │
│         │                      │                      │              │
│         │                      │                      │              │
│         │                      ▼                      ▼              │
│         │              ┌────────────────────────────────┐            │
│         └─────────────▶│        OBS Output              │            │
│                        │  - Text file (GDI+)            │            │
│                        │  - WebSocket (Browser Source)  │            │
│                        │  - HTTP Server (Overlay)       │            │
│                        └────────────────────────────────┘            │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                        ┌──────────────────────┐
                        │    OBS Studio        │
                        │  - Text Source       │
                        │  - Browser Source    │
                        └──────────────────────┘
```

## Component Architecture

### 1. Audio Capture ([src/audio_capture.py](src/audio_capture.py))

**Purpose:** Capture microphone audio in real-time and feed it to the transcription pipeline.

**Key Design Decisions:**
- Uses `sounddevice` library for cross-platform audio capture
- Runs in a separate thread (sounddevice uses blocking callbacks)
- Configurable chunk duration (default: 3 seconds) balances latency vs. accuracy
- 16kHz sample rate optimized for Whisper model

**Threading Model:**
```python
Main Thread                   Audio Thread
    │                             │
    │──── start() ───────────────▶│
    │                             │ sounddevice.InputStream
    │                             │    │
    │                             │    ├─ audio_callback()
    │                             │    │  └─ queue.put(audio_chunk)
    │                             │    │
    │◀────queue.get()─────────────┤    │
    │                             │    │
    │──── stop() ────────────────▶│    │
    │                             │◀───┘
```

**Data Flow:**
1. Microphone → sounddevice callback (every ~30ms)
2. Accumulate samples until chunk duration reached
3. Convert to numpy array (float32, mono)
4. Put in thread-safe queue
5. Main thread retrieves and processes

**Configuration:**
```yaml
audio:
  device: null          # null = default, or device index
  sample_rate: 16000    # Whisper-optimized
  chunk_duration: 3.0   # Seconds per chunk
```

**Error Handling:**
- Device not found → List available devices and exit
- Audio buffer overflow → Log warning, continue
- Invalid configuration → Raise ValueError with helpful message

### 2. Transcriber ([src/transcriber.py](src/transcriber.py))

**Purpose:** Convert audio chunks to text using faster-whisper (CTranslate2 backend).

**Key Design Decisions:**
- Uses `faster-whisper` instead of OpenAI Whisper (4x faster, lower memory)
- CTranslate2 backend enables GPU acceleration with CUDA
- Model loaded once at startup (expensive initialization)
- Supports multiple model sizes (tiny → large-v3)

**Model Selection Trade-offs:**

| Model | Size | Speed (CPU) | Speed (GPU) | Accuracy | Use Case |
|-------|------|-------------|-------------|----------|----------|
| tiny | ~40MB | Fast | Very fast | Low | Testing, low-end hardware |
| base | ~150MB | Medium | Fast | Good | **Recommended default** |
| small | ~500MB | Slow | Medium | Better | Higher accuracy needed |
| medium | ~1.5GB | Very slow | Medium | High | Professional transcription |
| large-v3 | ~3GB | Extremely slow | Slow | Highest | Best quality, powerful GPU |

**Inference Pipeline:**
```python
Audio (numpy.ndarray, 16kHz)
    │
    ├─ Pre-processing
    │  └─ Normalize audio levels
    │
    ├─ Whisper Model (CTranslate2)
    │  ├─ Encoder: audio → embeddings
    │  └─ Decoder: embeddings → tokens
    │
    ├─ Post-processing
    │  ├─ Decode tokens → text
    │  └─ Detect language (if auto-detect)
    │
    └─ Output: {"text": str, "language": str}
```

**GPU Acceleration:**
- Requires CUDA 12+ and cuDNN 9+ (CTranslate2 requirement)
- Automatically selected with `device: "auto"`
- Falls back to CPU if CUDA unavailable
- `compute_type`: `int8` (CPU), `float16` (GPU) for optimal speed

**Configuration:**
```yaml
whisper:
  model: "base"         # Model size
  device: "auto"        # cpu, cuda, auto
  compute_type: "int8"  # int8 (CPU), float16/float32 (GPU)
  language: null        # Auto-detect or specify (en, es, etc.)
```

**Language Detection:**
- Whisper can auto-detect language (30s+ audio recommended)
- Override with explicit language code for faster processing
- Detection result passed to translator for language pair selection

### 3. Translator ([src/translator.py](src/translator.py))

**Purpose:** Translate transcribed text using Meta's NLLB-200 model (200+ languages).

**Key Design Decisions:**
- Uses `transformers` library with NLLB-200-distilled-600M model
- Supports 200+ language pairs (all combinations)
- Can be disabled for transcription-only mode
- Model cached in memory after first load (~1.2GB)

**Translation Pipeline:**
```python
Text (str, source_lang)
    │
    ├─ Language Mapping
    │  └─ ISO code (es) → NLLB code (spa_Latn)
    │
    ├─ Tokenization
    │  └─ Text → token IDs
    │
    ├─ NLLB-200 Model
    │  ├─ Encoder: source tokens → embeddings
    │  └─ Decoder: embeddings → target tokens
    │
    ├─ Detokenization
    │  └─ Token IDs → text
    │
    └─ Output: translated text (target_lang)
```

**Language Code Mapping:**
NLLB uses specific codes (e.g., `spa_Latn` not `es`):
- Mapping table in `Translator._get_nllb_code()`
- Common languages pre-mapped
- Graceful fallback for unknown codes

**Performance:**
- First translation: ~2-3s (model load + inference)
- Subsequent: ~0.1-0.5s (CPU), ~0.05-0.1s (GPU)
- Translation can be disabled if only transcription needed

**Configuration:**
```yaml
translation:
  enabled: true
  source_lang: null     # null = auto from Whisper
  target_lang: "es"     # Target language code
```

**Optimization:**
- Batch processing possible but not implemented (single-user use case)
- GPU acceleration automatic if PyTorch + CUDA available
- Model quantization possible for lower memory usage

### 4. OBS Output ([src/obs_output.py](src/obs_output.py))

**Purpose:** Send subtitle text to OBS Studio via multiple output methods.

**Output Methods:**

#### 4.1 Text File (Simple, Lowest Latency)
- Writes to `subtitles.txt` on every update
- OBS Text (GDI+) source reads file periodically
- Pros: Simple, low overhead, no network
- Cons: File polling delay (~100ms), limited styling

```python
async def _update_text_file(self, text: str):
    with open(self.config.text_file, "w", encoding="utf-8") as f:
        f.write(text)
```

#### 4.2 WebSocket Server (Real-time, Programmatic)
- Runs WebSocket server on port 8765 (configurable)
- Broadcasts JSON messages to all connected clients
- OBS Browser Source connects via WebSocket
- Pros: Real-time, structured data, multiple clients
- Cons: Requires WebSocket client code in HTML

```python
async def _websocket_handler(websocket, path):
    # Register client
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        clients.remove(websocket)

async def _broadcast_text(text: str):
    message = json.dumps({"text": text, "timestamp": time.time()})
    await asyncio.gather(*[ws.send(message) for ws in clients])
```

**WebSocket Message Format:**
```json
{
  "text": "Translated subtitle text",
  "timestamp": 1234567890.123
}
```

#### 4.3 HTTP Server (Styled Overlay, User-friendly)
- Serves HTML overlay on port 8766 (configurable)
- Server-Sent Events (SSE) for real-time updates
- Pre-styled with CSS animations (fade in/out)
- Pros: Works out-of-box, professional styling, no client code
- Cons: Slightly higher overhead than WebSocket

**HTML Overlay Features:**
- Auto-fit text with word wrapping
- Fade-in animation on new text
- Auto-clear after silence period
- Configurable max lines (default: 2)
- Responsive sizing for different resolutions

```python
async def _serve_overlay(request):
    # Serve HTML with embedded CSS/JS
    return web.Response(text=HTML_TEMPLATE, content_type="text/html")

async def _serve_sse(request):
    # Server-Sent Events stream
    response = web.StreamResponse()
    await response.prepare(request)
    # Send updates as SSE messages
    await response.write(f"data: {text}\n\n".encode())
```

**Auto-clear Logic:**
- Timer starts on each subtitle update
- If no new text within `clear_after` seconds, clear display
- Prevents stale subtitles from lingering
- Configurable timeout (default: 5 seconds)

```python
async def _auto_clear_task(self):
    while self.running:
        await asyncio.sleep(0.1)
        if time.time() - self.last_update > self.config.clear_after:
            await self._update_all_outputs("")
```

**Configuration:**
```yaml
output:
  text_file: "./subtitles.txt"
  websocket_enabled: true
  websocket_port: 8765
  http_port: 8766
  max_lines: 2
  clear_after: 5.0
```

### 5. Pipeline ([src/pipeline.py](src/pipeline.py))

**Purpose:** Orchestrate all components and manage the processing pipeline.

**Concurrency Model:**

```
Main Thread (async)
├─ Audio Capture Thread
│  └─ Blocking sounddevice callbacks
│
├─ Transcription Task (async)
│  └─ Process audio chunks from queue
│
├─ Translation Task (async)
│  └─ Translate transcribed text
│
└─ OBS Output Tasks (async)
   ├─ WebSocket server
   ├─ HTTP server
   └─ Auto-clear timer
```

**Event Loop Architecture:**
```python
async def start(self):
    # Start all async components
    tasks = [
        asyncio.create_task(self._transcription_loop()),
        asyncio.create_task(self._obs_output.start()),
    ]

    # Start blocking audio in thread
    self._audio_capture.start()

    # Wait for shutdown signal
    await asyncio.gather(*tasks)
```

**Data Flow:**
```
Audio Thread ──▶ Queue ──▶ Transcription Loop
                              │
                              ├─ Transcriber.transcribe()
                              │         │
                              │         ▼
                              ├─ Translator.translate()
                              │         │
                              │         ▼
                              └─ OBSOutput.update_text()
                                         │
                                         ├─ Write file
                                         ├─ WebSocket broadcast
                                         └─ HTTP SSE update
```

**Graceful Shutdown:**
- Signal handler (Ctrl+C) sets shutdown flag
- All async tasks check flag and exit cleanly
- Audio thread stopped via event
- Resources cleaned up in `finally` blocks

```python
def _signal_handler(self, signum, frame):
    console.print("[yellow]Shutting down...[/yellow]")
    self.running = False
    self._shutdown_event.set()

async def stop(self):
    self.running = False
    self._audio_capture.stop()
    await self._obs_output.stop()
```

**Error Recovery:**
- Transcription errors: Log and skip chunk
- Translation errors: Use original text (no translation)
- Output errors: Log but continue processing
- Fatal errors: Clean shutdown with error message

### 6. CLI ([src/cli.py](src/cli.py))

**Purpose:** Command-line interface using Click framework.

**Commands:**

```bash
live-translator run            # Start the pipeline
live-translator list-devices   # List audio devices
live-translator list-models    # List Whisper models
```

**Configuration Priority:**
1. CLI flags (highest priority)
2. Config file specified with `-c`
3. Default `config.yaml`
4. Built-in defaults (lowest priority)

**Rich Output:**
- Colored status messages
- Tables for device/model lists
- Progress indicators during startup
- Formatted error messages

## Data Structures

### Configuration Classes

All configuration uses `@dataclass` with validation:

```python
@dataclass
class AudioConfig:
    device: Optional[int] = None
    sample_rate: int = 16000
    chunk_duration: float = 3.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioConfig":
        return cls(
            device=data.get("device"),
            sample_rate=data.get("sample_rate", 16000),
            chunk_duration=data.get("chunk_duration", 3.0)
        )
```

### Audio Data Format

```python
# Raw audio from sounddevice
audio_chunk: np.ndarray
# Shape: (num_samples,)  # Mono
# Dtype: float32
# Range: [-1.0, 1.0]
# Sample rate: 16000 Hz
```

### Transcription Result

```python
@dataclass
class TranscriptionResult:
    text: str              # Transcribed text
    language: str          # Detected/specified language code
    confidence: float      # Transcription confidence (0-1)
```

## Performance Characteristics

### Latency Breakdown

**Total latency** = Audio chunk + Transcription + Translation + Output

| Component | CPU | GPU |
|-----------|-----|-----|
| Audio chunk | 3.0s (configurable) | 3.0s |
| Transcription (base) | 1.5-2.0s | 0.3-0.5s |
| Translation | 0.5-1.0s | 0.1-0.2s |
| Output | <0.01s | <0.01s |
| **Total** | **5-6s** | **3.5-4s** |

**Optimization strategies:**
- Reduce chunk duration (1-2s) for lower latency
- Use GPU for 5-10x speedup
- Use smaller models (tiny/base) for faster processing
- Disable translation if not needed (saves 0.5-1s)

### Memory Usage

| Component | Memory |
|-----------|--------|
| Base Whisper model | ~150MB |
| NLLB-200 model | ~1.2GB |
| Audio buffers | ~10MB |
| Python runtime | ~50MB |
| **Total** | **~1.5GB** |

**Large models:**
- large-v3 Whisper: ~3GB
- Total with large models: ~5GB

### CPU/GPU Utilization

**CPU mode (8 cores):**
- Transcription: 80-100% (during processing)
- Translation: 60-80%
- Audio capture: <5%
- Output: <1%

**GPU mode (RTX 3060):**
- Transcription: 40-60% GPU
- Translation: 30-40% GPU
- CPU: 20-30% (preprocessing)

## Deployment Considerations

### System Requirements

**Minimum:**
- CPU: 4 cores, 2.0 GHz
- RAM: 4GB
- Python 3.10+
- Disk: 2GB (models)

**Recommended:**
- CPU: 8 cores, 3.0 GHz or GPU (NVIDIA RTX 2060+)
- RAM: 8GB
- Python 3.11+
- Disk: 5GB (all models)

**GPU Acceleration:**
- NVIDIA GPU with CUDA 12+ support
- cuDNN 9+ installed
- 4GB+ VRAM

### Scaling Limitations

**Current design:**
- Single user, single microphone
- No multi-speaker diarization
- No concurrent transcription jobs
- Text-only output (no audio)

**Future scaling possibilities:**
- Multi-microphone support (separate pipelines)
- Distributed processing (transcription/translation workers)
- Cloud deployment (API service)
- WebRTC for remote audio capture

## Security Considerations

1. **Local processing**: No audio sent to external services
2. **Network exposure**: HTTP/WebSocket servers on localhost only
3. **File permissions**: Output files readable by OBS user
4. **Model integrity**: Verify model checksums (not implemented)
5. **Input validation**: Audio device indices validated

## Testing Strategy

### Unit Tests
- Component initialization
- Configuration loading/validation
- Audio processing functions
- Translation language code mapping

### Integration Tests
- Audio capture → transcription flow
- Transcription → translation flow
- Translation → output flow
- Full pipeline end-to-end

### Performance Tests
- Latency benchmarks
- Memory usage profiling
- GPU utilization monitoring
- Long-running stability

### Manual Tests
- OBS integration (all output methods)
- Different audio devices
- Various model sizes
- Multiple languages

## Future Architecture Improvements

1. **Plugin system**: Extensible output methods
2. **Model caching**: Faster startup with cached models
3. **Distributed processing**: Separate transcription/translation workers
4. **Real-time metrics**: Latency, accuracy, resource usage dashboards
5. **Multi-language UI**: Internationalization support
6. **Cloud backend**: Optional cloud API for heavy models
7. **Streaming protocol**: Replace chunking with streaming transcription

---

**Last Updated:** 2026-01-23
**Architecture Version:** 1.0
