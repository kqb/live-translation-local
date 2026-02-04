# Multi-Modal Conversation Intelligence System - Project Context

This document provides context for Claude Code when working with this project.

## Project Overview

**Multi-Modal Conversation Intelligence System** is a comprehensive platform for capturing, processing, and analyzing conversations in real-time. It goes far beyond simple transcription—combining speaker recognition, translation, AR display, semantic memory, and multi-output streaming into a unified pipeline.

**What it does:**
- 🎤 **Real-time transcription** using Whisper with hallucination filtering
- 🌍 **200+ language translation** via NLLB-200
- 👥 **Speaker diarization + recognition** (pyannote.audio + persistent voice profiles)
- 🥽 **AR glasses output** (Even Realities G2 BLE integration)
- 🧠 **Semantic memory capture** (Exocortex/Qdrant vector database)
- 📼 **Session logging** (audio archiving + metadata)
- 🎬 **Multi-output streaming** (OBS, WebSocket, HTTP, AR glasses, files)

**Evolution:** Started as "Live Translator for OBS Studio" → evolved into full conversation intelligence platform

**Key Technologies:**
- **faster-whisper**: CTranslate2-based Whisper (4x faster)
- **NLLB-200**: Meta's multilingual translation (200+ languages)
- **pyannote.audio**: State-of-the-art speaker diarization
- **sentence-transformers**: Voice embeddings for speaker recognition
- **Qdrant**: Vector database for semantic memory (Exocortex)
- **Even G2 BLE protocol**: AR glasses integration (reverse-engineered)
- **sounddevice**: Audio capture from microphone
- **websockets & aiohttp**: Real-time OBS integration
- **Click + Rich**: Beautiful CLI

**Project Type:** Python 3.11+ CLI application with async/await architecture

---

## Architecture

### Expanded Pipeline Flow

```
┌────────────────────┐
│   Audio Inputs     │
│ • Microphone       │
│ • Audio Files      │
│ • Voice Memos      │  (PLANNED: Omi wearable)
└─────────┬──────────┘
          │
          ▼
┌──────────────────────────────────────────────────────┐
│             PROCESSING PIPELINE                       │
├──────────────────────────────────────────────────────┤
│  1. Transcription (faster-whisper)                   │
│     ├─ Hallucination filtering                       │
│     ├─ Language detection                            │
│     └─ Segment timing                                │
│                                                        │
│  2. Speaker Diarization (pyannote.audio 3.1)         │
│     ├─ Multi-speaker detection                       │
│     ├─ Segment assignment (SPEAKER_00, etc.)         │
│     └─ Speaking time calculation                     │
│                                                        │
│  3. Speaker Recognition (sentence-transformers)       │
│     ├─ Voice embedding extraction                    │
│     ├─ Cosine similarity matching                    │
│     ├─ Persistent profile database                   │
│     └─ Name assignment (Alice, Bob, etc.)            │
│                                                        │
│  4. Translation (NLLB-200)                           │
│     ├─ 200+ language pairs                           │
│     ├─ Custom glossary support                       │
│     └─ CTranslate2 backend                           │
│                                                        │
│  5. Semantic Memory (Exocortex)                      │
│     ├─ Vector embeddings (sentence-transformers)     │
│     ├─ Qdrant storage                                │
│     └─ Metadata indexing                             │
└─────────┬────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────┐
│                OUTPUT LAYER                           │
├──────────────────────────────────────────────────────┤
│  • OBS Studio (file/WebSocket/HTTP)                  │
│  • Even G2 Smart Glasses (BLE notifications)         │
│  • Session Logging (WAV chunks + JSON metadata)      │
│  • Exocortex Memory (Qdrant vector DB)               │
└──────────────────────────────────────────────────────┘
```

### Core Components

#### 1. **AudioCapture** ([src/audio_capture.py](src/audio_capture.py))
- Captures microphone audio in configurable chunks (default: 3 seconds)
- Uses sounddevice with 16kHz sample rate for Whisper
- Thread-based with queue for async processing
- Supports device selection by index

#### 2. **Transcriber** ([src/transcriber.py](src/transcriber.py))
- faster-whisper with CTranslate2 backend
- GPU (CUDA) or CPU execution
- Hallucination filtering (no_speech_threshold, logprob, compression_ratio)
- Auto-detects language or uses specified language
- Returns: text + detected language + confidence

#### 3. **Diarization** ([src/diarization.py](src/diarization.py))
- pyannote.audio 3.1 for speaker detection
- Detects 1-10 speakers per chunk (configurable)
- Assigns SPEAKER_00, SPEAKER_01, etc. labels
- Calculates speaking time per speaker
- Requires HuggingFace token for model access

#### 4. **SpeakerRecognition** ([src/speaker_recognition.py](src/speaker_recognition.py))
- Extracts voice embeddings using sentence-transformers
- Cosine similarity matching against enrolled profiles
- Persistent JSON database of speaker profiles
- Configurable threshold (0.0-1.0, default: 0.75)
- CLI enrollment: `live-translator enroll <name>`

#### 5. **Translator** ([src/translator.py](src/translator.py))
- NLLB-200 model via transformers library
- 200+ language pairs
- Custom glossary support
- Can be disabled for transcription-only
- GPU acceleration automatic if available

#### 6. **G2Output** ([src/g2_output.py](src/g2_output.py))
- BLE connection to Even Realities G2 smart glasses
- Reverse-engineered protocol (3-packet auth handshake)
- Dual-eye support (LEFT + RIGHT lenses)
- Notification mode (push translations as AR notifications)
- Display formats: original, translated, or both
- Speaker names included in AR display

#### 7. **SessionLogger** ([src/session_logger.py](src/session_logger.py))
- Records audio chunks as WAV files (lossless)
- JSON metadata: transcription, translation, speaker data
- Incremental saves every 10 chunks
- Session structure: `data/sessions/<timestamp>/`
- CLI replay: `live-translator sessions replay <id>`

#### 8. **ExocortexMemory** ([src/exocortex/memory.py](src/exocortex/memory.py))
- Stores conversations in Qdrant vector database
- sentence-transformers for embeddings
- Rich metadata: speaker, language, timestamp, confidence
- Semantic search via `exo query "<text>"`
- Integration: automatic capture when enabled

#### 9. **OBSOutput** ([src/obs_output.py](src/obs_output.py))
- Multiple output methods: text file, WebSocket, HTTP
- Text file: OBS Text (GDI+) source
- WebSocket: Real-time browser source (port 8765)
- HTTP: Styled HTML overlay (port 8766)
- YouTube-style scrolling subtitles
- Auto-clear after silence period

#### 10. **Pipeline** ([src/pipeline.py](src/pipeline.py))
- Orchestrates all components
- Async/await architecture
- Graceful shutdown (Ctrl+C)
- Error recovery and degradation
- Configuration loading from YAML

#### 11. **CLI** ([src/cli.py](src/cli.py))
- Click-based command interface
- Commands: `run`, `devices`, `languages`, `enroll`, `sessions`, etc.
- Rich terminal output
- Progress indicators for ingestion tasks

---

## Configuration

Configuration via [config.yaml](config.yaml):

```yaml
audio:
  device: null        # null = default, or device index
  sample_rate: 16000
  chunk_duration: 3.0

whisper:
  model: "medium"     # tiny/base/small/medium/large-v2/large-v3
  device: "auto"      # cpu, cuda, auto
  compute_type: "int8" # int8 (CPU), float16 (GPU)
  language: null      # null = auto-detect
  no_speech_threshold: 0.6
  logprob_threshold: -1.0
  compression_ratio_threshold: 2.4

translation:
  enabled: true
  source_lang: null   # null = auto from Whisper
  target_lang: "es"
  glossary_path: null

diarization:
  enabled: true       # Speaker detection
  min_speakers: 1
  max_speakers: 10
  hf_token: "hf_xxx"  # Required: HuggingFace token

speaker_recognition:
  enabled: true       # Persistent speaker names
  database_path: "./data/speakers.json"
  recognition_threshold: 0.75

g2:
  enabled: true       # Even G2 glasses
  mode: "notification"
  auto_connect: true
  use_right: false    # Use left eye (default)
  display_format: "both"  # original/translated/both

logging:
  enabled: true       # Session recording
  log_dir: "./data/sessions"
  save_audio: true
  incremental_save: true

exocortex:
  enabled: true
  qdrant_url: "http://localhost:6333"
  collection_name: "conversations"

output:
  text_file: "./subtitles.txt"
  websocket_enabled: true
  websocket_port: 8765
  http_port: 8766
  scrolling_mode: true
  history_lines: 10
  max_lines: 2
  clear_after: 5.0
```

---

## Code Conventions

### Python Style

**Formatting:**
- **Black**: Line length 100, target Python 3.11+
- **Ruff**: Linting with line length 100
- Run `black .` before committing
- Run `ruff check . --fix` to auto-fix

**Type Hints:**
- Use modern syntax: `dict[str, Any]` not `Dict[str, Any]`
- `from __future__ import annotations` for forward refs
- All public functions have type hints

**Imports:**
- Group: standard library → third-party → local
- Use absolute imports: `from src.transcriber import ...`

**Docstrings:**
- Google-style for all public functions/classes
- Include Args, Returns, Raises sections

### Dataclasses

All config objects use `@dataclass`:
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

- Use `async def` for I/O-bound ops
- `asyncio.Queue` for thread-safe communication
- `asyncio.create_task()` for concurrent tasks
- Cleanup in try/finally blocks

### Error Handling

- Rich console for user errors: `console.print("[red]Error:[/red] message")`
- Log technical details for debugging
- Graceful degradation (e.g., translation failures use original text)

### Threading

- Audio capture in `threading.Thread` (blocking sounddevice)
- Main pipeline uses `asyncio`
- `threading.Event` for shutdown
- Always set daemon appropriately

---

## Development Workflow

### Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### Testing

```bash
pytest                              # All tests
pytest --cov=src --cov-report=html  # With coverage
pytest tests/test_transcriber.py -v # Specific test
```

### Local Development

```bash
live-translator run                 # Start pipeline
live-translator devices             # List audio devices
live-translator languages           # List supported languages
live-translator enroll Alice        # Enroll speaker
live-translator sessions list       # List recorded sessions
live-translator test                # Test all components
exo query "what did Alice say"      # Search semantic memory
```

### Common Tasks

**Adding a feature:**
1. Update relevant component in `src/`
2. Add configuration to `config.yaml` template
3. Update CLI if needed
4. Add tests in `tests/`
5. Update README.md and this doc

**Optimizing performance:**
- **GPU**: CUDA 12+ and cuDNN 9+ for faster-whisper
- **Chunk duration**: Smaller (1-2s) = lower latency, more processing
- **Model size**: `base` = balanced, `tiny` = low-end hardware
- **Compute type**: `int8` (CPU), `float16` (GPU)

---

## Speaker System

### Diarization (pyannote.audio)

**Purpose:** Detect multiple speakers in audio chunks

**Setup:**
1. Get HuggingFace token: https://huggingface.co/settings/tokens
2. Accept model license: https://huggingface.co/pyannote/speaker-diarization-3.1
3. Add token to config: `diarization.hf_token: "hf_xxx"`

**Output:**
- Assigns labels: SPEAKER_00, SPEAKER_01, etc.
- Returns segments with speaker labels + timestamps
- Calculates total speaking time per speaker

**Limitations:**
- Accuracy drops for <2s chunks
- May merge/split speakers incorrectly
- Requires good audio quality (minimal background noise)

### Speaker Recognition (sentence-transformers)

**Purpose:** Map SPEAKER_XX labels to real names

**Enrollment:**
```bash
live-translator enroll Alice
# Speak for 5 seconds...
# Profile saved to data/speakers.json
```

**Matching:**
- Extracts voice embedding from audio
- Computes cosine similarity vs enrolled profiles
- Matches if similarity > threshold (default: 0.75)
- Falls back to SPEAKER_XX if no match

**Database:**
```json
{
  "Alice": {
    "embedding": [0.123, -0.456, ...],  # 512-dim vector
    "created_at": "2026-01-25T14:30:52",
    "sample_count": 1
  }
}
```

See [SPEAKER_ASSIGNMENT_GUIDE.md](SPEAKER_ASSIGNMENT_GUIDE.md) for full details.

---

## Even G2 Smart Glasses

### BLE Protocol

**Reverse-engineered from:**
- https://github.com/i-soxi/even-g2-protocol
- Community docs: https://discord.gg/arDkX3pr

**Connection Flow:**
1. Scan for `G2_L_XXXX` and `G2_R_XXXX`
2. Connect to both (LEFT + RIGHT)
3. 3-packet authentication handshake:
   - Packet 1: InitAuth
   - Packet 2: AuthResponse
   - Packet 3: CompleteAuth
4. Send notifications via characteristic `0000fd03-...`

**Notification Format:**
```json
{
  "appId": "com.example.app",
  "appName": "Live Translator",
  "layout": "Conversation",
  "text": "[Alice] Hello → Hola",
  "timestamp": "2026-01-25T14:30:52"
}
```

**Display Modes:**
- **Notification**: Push updates as AR notifications
- **Teleprompter** (planned): Scrolling long-form text

**Limitations:**
- Max 234 bytes per packet (multi-packet for longer messages)
- ~0.2-0.5s BLE latency
- Battery drain on glasses (~2-3h with frequent updates)

See [G2_INTEGRATION.md](G2_INTEGRATION.md) for setup guide.

---

## Session Logging

### What Gets Recorded

**Audio:**
- WAV files (16kHz, mono, lossless)
- Saved as `chunk_XXXXXX.wav`

**Metadata (JSON):**
```json
{
  "session_id": "20260125_143052",
  "started_at": "2026-01-25T14:30:52",
  "config": { ... },  // Snapshot of config
  "entries": [
    {
      "chunk_id": 0,
      "timestamp": "2026-01-25T14:30:55",
      "transcription": {
        "text": "Hello everyone",
        "language": "en",
        "confidence": 0.92
      },
      "translation": {
        "text": "Hola a todos",
        "language": "es"
      },
      "speakers": [
        {
          "speaker_id": "SPEAKER_00",
          "name": "Alice",
          "start": 0.5,
          "end": 2.3
        }
      ]
    }
  ]
}
```

### CLI

```bash
live-translator sessions list                    # List all sessions
live-translator sessions replay 20260125_143052  # Replay (audio + transcript)
live-translator sessions export 20260125_143052 --format srt  # Export subtitles
```

---

## Exocortex Semantic Memory

### Architecture

```
Conversation → Embeddings → Qdrant → Semantic Search
             (sentence-transformers)
```

**What's Stored:**
- Transcription text (original language)
- Translation (target language)
- Speaker name/ID
- Timestamp
- Language
- Confidence scores
- Session ID

**Vector Embeddings:**
- Model: `all-MiniLM-L6-v2` (sentence-transformers)
- Dimension: 384
- Metric: Cosine similarity

**Qdrant Schema:**
```python
{
  "id": "uuid",
  "vector": [0.123, -0.456, ...],  # 384-dim
  "payload": {
    "text": "Hello everyone",
    "translation": "Hola a todos",
    "speaker": "Alice",
    "language": "en",
    "timestamp": "2026-01-25T14:30:55",
    "session_id": "20260125_143052",
    "confidence": 0.92
  }
}
```

### CLI

```bash
exo query "what did Alice say about the project?"  # Semantic search
exo stats                                          # Show statistics
exo clear                                          # Clear all memories
```

See [EXOCORTEX_INTEGRATION.md](EXOCORTEX_INTEGRATION.md) for full details.

---

## OBS Studio Integration

### Text File Source

1. Add **Text (GDI+)** source
2. "Read from file" → `subtitles.txt`
3. Configure font/color

### Browser Source (Styled)

1. Add **Browser** source
2. URL: `http://localhost:8766`
3. Width: 1920, Height: 200
4. Includes fade animations

### WebSocket (Custom)

```html
<script>
const ws = new WebSocket('ws://localhost:8765');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // data.text, data.timestamp, data.speaker
};
</script>
```

---

## Dependencies

### Core Runtime

- **faster-whisper** (1.0.0+) — Whisper transcription
- **transformers** (4.35.0+) — NLLB-200 model
- **ctranslate2** (4.0.0+) — Fast inference
- **sounddevice** (0.4.6+) — Audio capture
- **pyannote.audio** (3.1+) — Speaker diarization
- **sentence-transformers** (2.2.0+) — Voice embeddings
- **qdrant-client** (1.7.0+) — Vector database
- **bleak** (0.21.0+) — BLE for G2 glasses
- **websockets** (12.0+) — WebSocket server
- **aiohttp** (3.9.0+) — HTTP server
- **click** (8.1.0+) — CLI framework
- **pyyaml** (6.0.0+) — Config loading
- **rich** (13.0.0+) — Terminal formatting

### Dev Dependencies

- **pytest** (7.0.0+)
- **pytest-asyncio** (0.21.0+)
- **black** (23.0.0+)
- **ruff** (0.1.0+)

### Optional System Requirements

- **CUDA 12+ and cuDNN 9+** — GPU acceleration (CTranslate2)
- **PyTorch with CUDA** — NLLB-200 GPU
- **Qdrant server** — Semantic memory (Docker or native)

---

## Performance Characteristics

### Latency Breakdown

**Total = Audio chunk + Transcription + Diarization + Recognition + Translation + Output**

| Component | CPU (8 cores) | GPU (RTX 3060) |
|-----------|---------------|----------------|
| Audio chunk | 3.0s | 3.0s |
| Transcription | 1.5-2.0s | 0.3-0.5s |
| Diarization | 2.0-3.0s | 1.0-1.5s |
| Recognition | 0.1-0.2s | 0.05-0.1s |
| Translation | 0.5-1.0s | 0.1-0.2s |
| Output | <0.01s | <0.01s |
| **Total** | **7-9s** | **4.5-5.5s** |

**Optimization:**
- Disable diarization/recognition → Save 2-3s
- Use smaller models → Save 1-2s
- Reduce chunk duration → Lower base latency

### Memory Usage

| Component | Memory |
|-----------|--------|
| Whisper (medium) | ~770MB |
| NLLB-200 | ~1.2GB |
| pyannote.audio | ~300MB |
| Qdrant embeddings | ~50MB |
| Runtime | ~100MB |
| **Total** | **~2.5GB** |

---

## Known Issues & Limitations

1. **First run slow**: Downloads models (Whisper ~770MB, NLLB ~1.2GB, pyannote ~300MB)
2. **Memory intensive**: Requires 4-8GB RAM with all features enabled
3. **CUDA 12+ required**: Older CUDA versions not supported by CTranslate2
4. **Diarization accuracy**: Drops for <2s chunks or noisy audio
5. **G2 battery drain**: Frequent updates drain glasses battery (~2-3h)
6. **Voice memo ingestion**: 40/62 files remaining (last worked: Jan 27)

---

## Future Roadmap

**In Progress:**
- [ ] **Omi wearable integration** — BLE input device support (research complete)
- [ ] **Voice memo batch ingestion** — Process remaining 40/62 files

**Planned:**
- [ ] G2 teleprompter mode (scrolling long-form text)
- [ ] Multi-language simultaneous translation
- [ ] YouTube/Twitch caption API integration
- [ ] Offline translation support
- [ ] Mobile app (iOS/Android)
- [ ] Web dashboard for session management
- [ ] Custom model fine-tuning interface

---

## Related Resources

### Core Technologies

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [Qdrant](https://qdrant.tech/)

### Community & Protocols

- [even-g2-protocol](https://github.com/i-soxi/even-g2-protocol)
- [EvenRealities Discord](https://discord.gg/arDkX3pr)
- [Omi GitHub](https://github.com/BasedHardware/omi)
- [Omi Docs](https://docs.omi.me)

---

**Last Updated:** 2026-02-04  
**Project Version:** 2.0 (Conversation Intelligence System)  
**Python Version:** 3.11+
