# Live Translator for OBS Studio

Real-time speech transcription and translation for live streaming. Capture microphone audio, transcribe with Whisper, translate with NLLB-200, and display in OBS Studio.

## Features

- **Real-time transcription** using faster-whisper (CTranslate2-based Whisper)
- **200+ language translation** using Meta's NLLB-200 model
- **OBS Studio integration** via text file and browser source overlay
- **Auto language detection** or manual language specification
- **Low latency** with configurable chunk duration
- **Styled overlay** with fade animations for professional streaming

## Requirements

- Python 3.10+
- CUDA (optional, for GPU acceleration)
- Working microphone
- OBS Studio (for streaming integration)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/live-translator/live-translator-obs.git
cd live-translator-obs
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

### 4. Download models (first run)

The models will be downloaded automatically on first use, or you can pre-download them:

```bash
live-translator install en es
```

## Quick Start

### 1. List audio devices

```bash
live-translator devices
```

### 2. Start the translator

```bash
# Default configuration (auto-detect language, translate to Spanish)
live-translator run

# Specify source and target languages
live-translator run -s en -t fr

# Use a specific audio device
live-translator run -d 2

# Use a larger Whisper model for better accuracy
live-translator run -m medium

# Custom configuration file
live-translator run -c myconfig.yaml
```

### 3. Configure OBS Studio

#### Option A: Text Source (GDI+)

1. Add a **Text (GDI+)** source in OBS
2. Check **Read from file**
3. Select `subtitles.txt` in your working directory
4. Style the text as desired

#### Option B: Browser Source (Recommended)

1. Add a **Browser** source in OBS
2. Set URL to `http://localhost:8766/overlay`
3. Set width to 1920 and height to 1080
4. The overlay has a transparent background and styled subtitles

## Configuration

Create a `config.yaml` file to customize settings:

```yaml
audio:
  device: null        # null = default mic, or device index
  sample_rate: 16000  # 16kHz for Whisper
  chunk_duration: 3.0 # seconds per audio chunk

whisper:
  model: "base"       # tiny, base, small, medium, large-v2, large-v3
  device: "auto"      # cpu, cuda, auto
  compute_type: "int8" # int8, float16, float32
  language: null      # null = auto-detect

translation:
  enabled: true
  source_lang: null   # null = auto from whisper
  target_lang: "es"   # target language code

output:
  text_file: "./subtitles.txt"
  websocket_enabled: true
  websocket_port: 8765
  http_port: 8766
  max_lines: 2
  clear_after: 5.0    # seconds
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `live-translator run` | Start the live translator |
| `live-translator devices` | List audio input devices |
| `live-translator languages` | List supported languages |
| `live-translator install <from> <to>` | Download translation model |
| `live-translator test` | Test all components |
| `live-translator config` | Show default configuration template |

### Run Options

```
-c, --config PATH    Configuration file path
-m, --model MODEL    Whisper model (tiny/base/small/medium/large-v2/large-v3)
-s, --source LANG    Source language code
-t, --target LANG    Target language code
-d, --device INDEX   Audio input device index
--no-translate       Disable translation (transcription only)
```

## Supported Languages

The translator supports 200+ languages. Common ones include:

| Code | Language | Code | Language |
|------|----------|------|----------|
| en | English | ja | Japanese |
| es | Spanish | ko | Korean |
| fr | French | zh | Chinese |
| de | German | ar | Arabic |
| it | Italian | hi | Hindi |
| pt | Portuguese | ru | Russian |

Run `live-translator languages` for the full list.

## Whisper Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | 39M | Fastest | Basic |
| base | 74M | Fast | Good |
| small | 244M | Medium | Better |
| medium | 769M | Slow | Great |
| large-v2 | 1.5G | Slowest | Best |
| large-v3 | 1.5G | Slowest | Best |

For live streaming, `base` or `small` models are recommended for the best balance of speed and accuracy.

## GPU Acceleration

The translator automatically uses CUDA if available. For best performance:

1. Install CUDA Toolkit 11.x or 12.x
2. Install cuDNN
3. Install PyTorch with CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Architecture

```
┌─────────────────┐
│   Microphone    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Audio Capture  │  sounddevice
│   (chunking)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Transcriber   │  faster-whisper
│  (Whisper ASR)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Translator    │  CTranslate2 + NLLB-200
│  (200+ langs)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   OBS Output    │
│ ┌─────────────┐ │
│ │  Text File  │ │  → OBS Text (GDI+)
│ ├─────────────┤ │
│ │  WebSocket  │ │  → Browser Source
│ │  + HTTP     │ │
│ └─────────────┘ │
└─────────────────┘
```

## Troubleshooting

### No audio captured

- Run `live-translator devices` to list available devices
- Specify the correct device with `-d INDEX`
- Check microphone permissions

### High latency

- Reduce `chunk_duration` in config (minimum ~2 seconds)
- Use a smaller Whisper model (`tiny` or `base`)
- Use GPU acceleration if available

### Translation errors

- Run `live-translator install <from> <to>` to ensure models are downloaded
- Check internet connection for initial model download

### OBS browser source not updating

- Ensure WebSocket port (8765) is not blocked
- Check browser source URL: `http://localhost:8766/overlay`
- Try refreshing the browser source in OBS

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - CTranslate2 implementation of Whisper
- [NLLB-200](https://github.com/facebookresearch/fairseq/tree/nllb) - Meta's No Language Left Behind
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) - Fast inference engine
- [sounddevice](https://python-sounddevice.readthedocs.io/) - Audio I/O library
