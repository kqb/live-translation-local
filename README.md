# Multi-Modal Conversation Intelligence System

Real-time transcription, translation, speaker recognition, AR glasses output, and semantic memory capture—all in one unified system.

**Transform conversations into actionable intelligence** with state-of-the-art AI models, persistent speaker profiles, AR display capabilities, and comprehensive session logging.

## What Makes This Special

This isn't just a translator—it's a complete conversation intelligence platform that combines:

- **🎯 Production-Grade Transcription**: Whisper with advanced hallucination filtering
- **🌍 200+ Language Translation**: NLLB-200 with custom glossary support
- **👥 Speaker Intelligence**: Automatic diarization + persistent voice recognition
- **🥽 AR Glasses Output**: Real-time display on Even Realities G2 smart glasses
- **🧠 Semantic Memory**: Capture conversations into vector-searchable knowledge base
- **📼 Session Recording**: Archive audio chunks with rich metadata for analysis
- **🎬 Multi-Output**: OBS streaming, WebSocket, HTTP, AR glasses, file logging

## Key Features

### Real-Time Transcription & Translation
- **faster-whisper** with CTranslate2 backend (4x faster than OpenAI Whisper)
- **NLLB-200** translation supporting 200+ language pairs
- Advanced anti-hallucination filtering (no_speech_threshold, logprob, compression_ratio)
- Auto language detection or manual specification
- Custom glossary support for domain-specific terminology

### Speaker Intelligence
- **Automatic Diarization**: Detect and label multiple speakers (pyannote.audio 3.1)
- **Persistent Recognition**: Enroll voice profiles ("Alice", "Bob") that persist across sessions
- **Cosine Similarity Matching**: Identify speakers by voice characteristics
- **Speaker Database**: JSON-based storage with configurable thresholds

### Even G2 Smart Glasses Integration
- **BLE Protocol**: Direct connection to Even Realities G2 AR glasses
- **Dual Eye Support**: Automatically connects to both left and right lenses
- **Notification Mode**: Push translations as AR notifications
- **Flexible Display**: Show original, translated, or both with speaker names
- **Teleprompter Mode**: (Coming soon) Scrolling display for long-form content

### Exocortex Memory System
- **Semantic Capture**: Store conversations in Qdrant vector database
- **Embedding Models**: sentence-transformers for semantic similarity
- **Rich Metadata**: Timestamp, speaker, language, confidence scores
- **CLI Integration**: Query and analyze your conversation history

### Session Logging & Recording
- **Audio Archiving**: Save WAV chunks with lossless quality
- **Metadata Tracking**: JSON logs with transcription, translation, speaker data
- **Incremental Saves**: Automatic saves every 10 chunks
- **Session Summaries**: Duration, chunk count, speaker distribution

### Streaming & Output
- **OBS Studio**: Text file and browser source overlay with styled subtitles
- **WebSocket**: Real-time updates (port 8765) for custom integrations
- **HTTP Server**: Standalone HTML overlay (port 8766) with fade animations
- **YouTube-Style Scrolling**: Continuous subtitle history with configurable lines

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/kqb/live-translation-local.git
cd live-translation-local

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate on Windows

# Install dependencies
pip install -e .
```

### 2. Basic Transcription & Translation

```bash
# List available audio devices
live-translator devices

# Start with default config (auto-detect → Spanish)
live-translator run

# Specify languages
live-translator run -s en -t ja

# Use larger model for better accuracy
live-translator run -m medium
```

### 3. Enable Speaker Recognition

```bash
# Edit config.yaml
diarization:
  enabled: true
  hf_token: "hf_xxxxx"  # Get from https://huggingface.co/settings/tokens

speaker_recognition:
  enabled: true

# Enroll speakers (speak for 5 seconds)
live-translator enroll Alice
live-translator enroll Bob

# Run with speaker names
live-translator run
```

### 4. Connect Even G2 Smart Glasses

```bash
# Edit config.yaml
g2:
  enabled: true
  mode: "notification"
  auto_connect: true
  display_format: "both"  # Show original + translation

# Make sure glasses are powered on
live-translator run
# Automatically scans, connects, and authenticates via BLE
```

### 5. Enable Session Logging

```bash
# Edit config.yaml
logging:
  enabled: true
  log_dir: "./data/sessions"
  save_audio: true

# Run and record everything
live-translator run

# List recorded sessions
live-translator sessions list

# Replay session
live-translator sessions replay <session_id>
```

## Use Cases

### 🎤 Live Event Translation
Stream real-time translations for international audiences using OBS Studio integration.

### 🥽 AR-Assisted Conversations
Wear G2 smart glasses and see translations directly in your field of view during face-to-face conversations.

### 🗣️ Meeting Transcription
Record multi-speaker meetings with automatic speaker labeling and searchable archives.

### 🧠 Personal Knowledge Base
Capture conversations into your Exocortex semantic memory for future reference and analysis.

### 📺 Content Creation
Add professional multi-language subtitles to streams with YouTube-style scrolling display.

### 🔬 Research & Analysis
Archive audio chunks with metadata for linguistic research or conversation pattern analysis.

## Configuration

See [config.yaml](config.yaml) for the complete configuration template.

### Core Settings

```yaml
audio:
  device: null        # null = default mic, or device index
  sample_rate: 16000  # 16kHz for Whisper
  chunk_duration: 3.0 # seconds per chunk

whisper:
  model: "medium"     # tiny, base, small, medium, large-v2, large-v3
  device: "auto"      # cpu, cuda, auto
  compute_type: "int8" # int8, float16, float32
  language: null      # null = auto-detect

  # Anti-hallucination settings
  no_speech_threshold: 0.6  # Higher = more aggressive filtering
  logprob_threshold: -1.0   # Filter low-confidence text
  compression_ratio_threshold: 2.4  # Filter repetitive text

translation:
  enabled: true
  source_lang: null   # null = auto from Whisper
  target_lang: "es"   # Target language code
  glossary_path: null # Optional custom glossary
```

### Speaker System

```yaml
diarization:
  enabled: true       # Enable speaker detection
  min_speakers: 1
  max_speakers: 10
  hf_token: "hf_xxx"  # Required: HuggingFace token

speaker_recognition:
  enabled: true       # Enable persistent speaker names
  database_path: "./data/speakers.json"
  recognition_threshold: 0.75  # Similarity threshold (0.0-1.0)
```

### Even G2 Smart Glasses

```yaml
g2:
  enabled: true       # Enable G2 output
  mode: "notification"  # Display mode: notification or teleprompter
  auto_connect: true  # Auto-connect on startup
  use_right: false    # Use left eye (default)
  display_format: "both"  # "original", "translated", or "both"
```

### Session Logging

```yaml
logging:
  enabled: true       # Enable session recording
  log_dir: "./data/sessions"
  save_audio: true    # Save audio chunks
  incremental_save: true  # Save metadata every 10 chunks
```

### Output Options

```yaml
output:
  text_file: "./subtitles.txt"  # For OBS Text (GDI+)
  websocket_enabled: true
  websocket_port: 8765
  http_port: 8766
  max_lines: 2        # Legacy mode
  clear_after: 5.0    # Clear after N seconds silence

  # YouTube-style scrolling subtitles
  scrolling_mode: true  # Enable continuous scrolling
  history_lines: 10     # Number of lines visible
```

## CLI Reference

### Core Commands

| Command | Description |
|---------|-------------|
| `live-translator run` | Start the live translator |
| `live-translator devices` | List audio input devices |
| `live-translator languages` | List supported languages |
| `live-translator test` | Test all components |
| `live-translator config` | Show configuration template |

### Speaker Management

| Command | Description |
|---------|-------------|
| `live-translator enroll <name>` | Enroll new speaker profile |
| `live-translator speakers list` | List enrolled speakers |
| `live-translator speakers delete <name>` | Remove speaker profile |

### Session Management

| Command | Description |
|---------|-------------|
| `live-translator sessions list` | List recorded sessions |
| `live-translator sessions replay <id>` | Replay session |
| `live-translator sessions export <id>` | Export to SRT/VTT |

### Exocortex Integration

| Command | Description |
|---------|-------------|
| `exo query <text>` | Search semantic memory |
| `exo stats` | Show memory statistics |
| `exo clear` | Clear all memories |

### Run Options

```
-c, --config PATH    Configuration file path
-m, --model MODEL    Whisper model (tiny/base/small/medium/large-v2/large-v3)
-s, --source LANG    Source language code
-t, --target LANG    Target language code
-d, --device INDEX   Audio input device index
--no-translate       Disable translation (transcription only)
--backend BACKEND    Whisper backend (whisper-cpp or faster-whisper)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Microphone  │  │  Audio File  │  │  Video File  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         └──────────────────┴──────────────────┘              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    PROCESSING LAYER                          │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Transcription (faster-whisper)                        │ │
│  │  • Whisper models (tiny → large-v3)                   │ │
│  │  • Hallucination filtering                            │ │
│  │  • Language detection                                 │ │
│  └──────────────────────┬─────────────────────────────────┘ │
│                         │                                    │
│                         ▼                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Diarization (pyannote.audio 3.1)                     │ │
│  │  • Speaker detection & labeling                       │ │
│  │  • Segment assignment                                 │ │
│  └──────────────────────┬─────────────────────────────────┘ │
│                         │                                    │
│                         ▼                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Speaker Recognition (sentence-transformers)          │ │
│  │  • Voice embedding extraction                         │ │
│  │  • Cosine similarity matching                         │ │
│  │  • Persistent profile database                        │ │
│  └──────────────────────┬─────────────────────────────────┘ │
│                         │                                    │
│                         ▼                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Translation (NLLB-200)                               │ │
│  │  • 200+ language pairs                                │ │
│  │  • Custom glossary support                            │ │
│  │  • CTranslate2 backend                                │ │
│  └──────────────────────┬─────────────────────────────────┘ │
└─────────────────────────┼───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                       OUTPUT LAYER                           │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  OBS Studio  │  │   G2 Glasses │  │   Session    │      │
│  │              │  │              │  │   Logging    │      │
│  │ • Text File  │  │ • BLE Notify │  │              │      │
│  │ • WebSocket  │  │ • Left/Right │  │ • WAV Chunks │      │
│  │ • HTTP       │  │ • Both Eyes  │  │ • JSON Meta  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Exocortex Memory (Qdrant)                          │   │
│  │  • Vector embeddings                                │   │
│  │  • Semantic search                                  │   │
│  │  • Metadata storage                                 │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Advanced Features

### Speaker Diarization & Recognition

**Automatic Speaker Detection**:
- Uses pyannote.audio 3.1 for state-of-the-art diarization
- Detects multiple speakers automatically (configurable 1-10 speakers)
- Assigns SPEAKER_00, SPEAKER_01, etc. labels to segments

**Persistent Speaker Profiles**:
- Enroll speakers with 5-second voice samples
- Voice embeddings stored in JSON database
- Cosine similarity matching (configurable threshold 0.0-1.0)
- Speakers recognized across sessions

**Usage**:
```bash
# Enroll speakers
live-translator enroll Alice
# Speak for 5 seconds...

live-translator enroll Bob
# Speak for 5 seconds...

# Run with speaker names
live-translator run
# Output: [Alice] Hello everyone → Hola a todos
#         [Bob] How's it going? → ¿Cómo va todo?
```

See [SPEAKER_ASSIGNMENT_GUIDE.md](SPEAKER_ASSIGNMENT_GUIDE.md) for comprehensive documentation.

### Even G2 Smart Glasses

**Real-Time AR Translation**:
- Direct BLE connection to Even Realities G2 glasses
- Automatic discovery and authentication
- Dual-eye support (LEFT and RIGHT lenses)
- Push notifications with speaker names and translations

**Display Formats**:
- **both**: Show original + translation with speaker name
- **original**: Show only transcribed text
- **translated**: Show only translation

**Architecture**:
- Uses reverse-engineered BLE protocol from [even-g2-protocol](https://github.com/i-soxi/even-g2-protocol)
- 3-packet authentication handshake
- JSON payloads with CRC32C checksums
- Multi-packet support for messages >234 bytes

**Performance**:
- Total latency: ~1-3 seconds end-to-end
- Transcription: ~0.5-2s (Whisper)
- Translation: ~0.1-0.5s (NLLB-200)
- G2 Display: ~0.2-0.5s (BLE)

See [G2_INTEGRATION.md](G2_INTEGRATION.md) for setup guide.

### Session Logging & Recording

**What Gets Recorded**:
- **Audio Chunks**: WAV files (16kHz, lossless)
- **Transcriptions**: Original text with language detection
- **Translations**: Translated text with confidence scores
- **Speaker Data**: Diarization labels + recognized names
- **Timestamps**: ISO format for precise timing
- **Metadata**: Session duration, chunk count, configuration snapshot

**Session Structure**:
```
data/sessions/20260125_143052/
├── session.json          # Metadata + all log entries
└── audio/
    ├── chunk_000000.wav
    ├── chunk_000001.wav
    └── ...
```

**CLI**:
```bash
# List sessions
live-translator sessions list

# Replay session (play audio + show transcript)
live-translator sessions replay 20260125_143052

# Export to SRT/VTT subtitles
live-translator sessions export 20260125_143052 --format srt
```

### Exocortex Semantic Memory

**Semantic Conversation Storage**:
- Stores transcriptions as vector embeddings in Qdrant
- sentence-transformers for semantic similarity
- Rich metadata: speaker, language, timestamp, confidence
- Query conversations by meaning, not keywords

**Integration**:
```bash
# Configure in config.yaml
exocortex:
  enabled: true
  qdrant_url: "http://localhost:6333"
  collection_name: "conversations"

# Query memory
exo query "what did Alice say about the project timeline?"

# Statistics
exo stats
# Output: 1,247 memories | 5 speakers | 12 sessions
```

See [EXOCORTEX_INTEGRATION.md](EXOCORTEX_INTEGRATION.md) for architecture details.

## OBS Studio Integration

### Text Source (Simple)

1. Add **Text (GDI+)** source in OBS
2. Check "Read from file"
3. Browse to `subtitles.txt`
4. Configure font, size, color

### Browser Source (Styled Overlay)

1. Add **Browser** source in OBS
2. URL: `http://localhost:8766`
3. Width: 1920, Height: 200
4. Includes fade animations and professional styling

### WebSocket (Custom Integrations)

Create custom HTML overlays:
```html
<script>
const ws = new WebSocket('ws://localhost:8765');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.text, data.timestamp, data.speaker);
};
</script>
```

## Supported Languages

**200+ languages supported** via NLLB-200. Common pairs:

| Code | Language | Code | Language |
|------|----------|------|----------|
| en | English | ja | Japanese |
| es | Spanish | ko | Korean |
| fr | French | zh | Chinese (Simplified) |
| de | German | ar | Arabic |
| it | Italian | hi | Hindi |
| pt | Portuguese | ru | Russian |
| nl | Dutch | th | Thai |
| pl | Polish | vi | Vietnamese |

Run `live-translator languages` for the complete list.

## Whisper Models

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| tiny | 39M | Fastest | Basic | Testing, low-end hardware |
| base | 74M | Fast | Good | Real-time streaming |
| small | 244M | Medium | Better | Quality streaming |
| medium | 769M | Slow | Great | High-quality recording |
| large-v2 | 1.5G | Slowest | Best | Maximum accuracy |
| large-v3 | 1.5G | Slowest | Best | Maximum accuracy (latest) |

**Recommendation**: Use `base` for live streaming, `medium` for recording.

## Performance Optimization

### GPU Acceleration

**For Whisper (faster-whisper)**:
```bash
# Requires CUDA 12+ and cuDNN 9+
# Install CUDA Toolkit from NVIDIA
# Install cuDNN from NVIDIA

# Verify CUDA support
python -c "import ctranslate2; print(ctranslate2.get_supported_compute_types('cuda'))"
```

**For NLLB-200 (PyTorch)**:
```bash
# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Latency Tuning

**Lower Latency**:
- Reduce `chunk_duration` (minimum ~1.5s for Whisper)
- Use smaller model (`tiny` or `base`)
- Disable translation if not needed
- Disable speaker recognition if not needed

**Higher Quality**:
- Use larger model (`medium` or `large-v3`)
- Increase `chunk_duration` (3-5s)
- Enable speaker diarization
- Lower hallucination thresholds

### Benchmarks

Approximate performance (base model, 3s chunks):

| Hardware | Transcription | Translation | Total Latency |
|----------|--------------|-------------|---------------|
| CPU (8 cores) | 1.5-2s | 0.5-1s | ~2-3s |
| GPU (RTX 3060) | 0.3-0.5s | 0.1-0.2s | ~0.5-1s |
| Apple M1 | 0.8-1.2s | 0.3-0.5s | ~1-2s |

## Troubleshooting

### Audio Issues

**No audio captured**:
```bash
# List devices
live-translator devices

# Test specific device
live-translator run -d 2

# Check permissions (macOS)
System Preferences → Security & Privacy → Microphone
```

### Speaker Recognition Issues

**Diarization not working**:
- Verify HuggingFace token is valid
- Accept pyannote model license at https://huggingface.co/pyannote/speaker-diarization-3.1
- Check token has model access permissions

**Speaker names not showing**:
- Ensure `speaker_recognition.enabled: true`
- Enroll speakers with `live-translator enroll <name>`
- Check recognition threshold (lower = more matches, higher = stricter)

### G2 Glasses Issues

**Glasses not found**:
- Power on glasses (show as G2_L_XXXX and G2_R_XXXX in Bluetooth)
- Disconnect from Even app first
- Move glasses closer to computer
- Check Bluetooth is enabled

**Notifications not appearing**:
- Wake glasses display (tap side button)
- Check battery level
- Verify `g2.enabled: true` in config
- Look for "Connected and authenticated!" in logs

### Model Download Issues

**Slow download**:
- Models download on first use (Whisper ~150MB, NLLB-200 ~1.2GB)
- Pre-download: `live-translator install en es`

**CUDA errors**:
- Verify CUDA 12+ installed
- Check cuDNN 9+ installed
- Verify GPU compatibility: `nvidia-smi`

### WebSocket/HTTP Issues

**Browser source not updating**:
- Check port 8765 (WebSocket) not blocked by firewall
- Verify URL: `http://localhost:8766`
- Try refreshing browser source in OBS
- Check `websocket_enabled: true` in config

## Development

### Project Structure

```
live-translation-local/
├── src/
│   ├── cli.py                  # CLI entry point
│   ├── pipeline.py             # Main orchestration
│   ├── audio_capture.py        # Microphone input
│   ├── transcriber.py          # Whisper transcription
│   ├── translator.py           # NLLB-200 translation
│   ├── obs_output.py           # OBS outputs (file/WS/HTTP)
│   ├── diarization.py          # Speaker diarization
│   ├── speaker_recognition.py  # Persistent speaker profiles
│   ├── g2_output.py            # Even G2 BLE protocol
│   ├── session_logger.py       # Session recording
│   └── exocortex/
│       ├── cli.py              # Exocortex CLI
│       └── memory.py           # Semantic memory
├── config.yaml                 # Configuration
├── pyproject.toml              # Package metadata
└── requirements.txt            # Dependencies
```

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_transcriber.py -v
```

### Code Style

```bash
# Format code
black . --line-length 100

# Lint code
ruff check . --fix
```

## Known Limitations

1. **First Run**: Model downloads take time (Whisper ~150MB, NLLB-200 ~1.2GB)
2. **Memory Usage**: Large models require 4-8GB RAM
3. **CUDA**: Requires CUDA 12+, older versions not supported
4. **Language Detection**: May be inaccurate for short chunks (<2s)
5. **Translation Quality**: NLLB-200 quality varies by language pair
6. **G2 Teleprompter**: Not yet implemented (notification mode only)
7. **Batch Audio Processing**: Coming soon for audio file ingestion

## Roadmap

**Planned Features**:
- [x] Speaker diarization
- [x] Persistent speaker recognition
- [x] Even G2 smart glasses output
- [x] Session logging
- [x] Exocortex semantic memory
- [ ] Batch audio file processing
- [ ] Video file ingestion with visual context
- [ ] G2 teleprompter mode
- [ ] Multi-language simultaneous translation
- [ ] YouTube/Twitch caption API integration
- [ ] Offline translation support
- [ ] Mobile app (iOS/Android)
- [ ] Web dashboard

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

**Core Technologies**:
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - CTranslate2 Whisper implementation
- [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) - Meta's No Language Left Behind
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [sentence-transformers](https://www.sbert.net/) - Voice embeddings
- [Qdrant](https://qdrant.tech/) - Vector database for semantic memory

**Community**:
- [even-g2-protocol](https://github.com/i-soxi/even-g2-protocol) - G2 BLE protocol reverse engineering
- [EvenRealities Discord](https://discord.gg/arDkX3pr) - G2 protocol community

**Infrastructure**:
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) - Fast inference engine
- [sounddevice](https://python-sounddevice.readthedocs.io/) - Audio I/O
- [Click](https://click.palletsprojects.com/) - CLI framework
- [Rich](https://rich.readthedocs.io/) - Terminal formatting

---

**Built with ❤️ for multilingual communication and conversation intelligence**

For issues, feature requests, or questions, please open an issue on GitHub.
