# Speaker Assignment Guide

Complete guide for assigning speakers to your exocortex memories.

## Overview

Your exocortex now supports **two-tier speaker identification**:

1. **Speaker Diarization** - Automatically detects and labels speakers as SPEAKER_00, SPEAKER_01, etc.
2. **Speaker Recognition** (Optional) - Maps those labels to actual names like "Alice", "Bob", etc.

## Current Setup

### Configuration ([config.yaml](config.yaml))

```yaml
diarization:
  enabled: true       # ✓ Enabled
  min_speakers: 1
  max_speakers: 10
  hf_token: "hf_SiX..."  # ✓ Configured

speaker_recognition:
  enabled: false      # Not yet enabled (optional)
  database_path: "./data/speakers.json"
  recognition_threshold: 0.75
```

## Usage

### 1. Ingest New Audio Files with Speaker Diarization

```bash
# Single file
python3 -m src.exocortex.cli ingest path/to/audio.wav

# Multiple files
python3 -m src.exocortex.cli ingest file1.wav file2.m4a file3.mp3

# Entire directory (recursive)
python3 -m src.exocortex.cli ingest ~/Voice\ Memos/ --recursive

# Voice Memos folder
python3 -m src.exocortex.cli ingest \
  "/Users/yuzucchi/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings" \
  --recursive
```

**What happens:**
- Audio is transcribed with Whisper
- Speakers are detected and labeled (SPEAKER_00, SPEAKER_01, etc.)
- Each transcription segment gets a speaker label
- Memories are indexed in the exocortex with speaker information

### 2. Re-process Existing Memories

Re-process your existing 1,796 memory segments from ~550 unique audio files:

```bash
# Test with 5 files first
python3 reprocess_optimized.py 5

# Process all files
python3 reprocess_optimized.py
```

**What happens:**
- Groups memories by audio file
- Runs diarization on each unique audio file once
- Updates all memories from that file with speaker labels
- Re-indexes in the database

**Estimated time:** ~2-5 minutes per audio file (depends on length and hardware)

### 3. Search Memories with Speaker Info

```bash
# Search all memories
python3 -m src.exocortex.cli recall "Bitcoin"

# Check stats
python3 -m src.exocortex.cli stats
```

Results now include speaker labels in the output.

## Speaker Recognition (Optional Advanced Feature)

To map SPEAKER_00/01 to actual names across sessions:

### Enable in config.yaml

```yaml
speaker_recognition:
  enabled: true  # Change to true
  database_path: "./data/speakers.json"
  recognition_threshold: 0.75
```

### Enroll Speakers

```bash
# List enrolled speakers
python3 speaker_enrollment.py list-speakers

# Enroll a speaker from audio sample
python3 speaker_enrollment.py enroll Alice ~/audio/alice_voice.wav

# Enroll more samples for better accuracy
python3 speaker_enrollment.py enroll Alice ~/audio/alice_voice2.wav

# Test recognition
python3 speaker_enrollment.py recognize ~/audio/test.wav
```

**Via exocortex CLI:**

```bash
# List speakers
python3 -m src.exocortex.cli speakers list

# Enroll speaker
python3 -m src.exocortex.cli speakers enroll Alice ~/audio/alice.wav

# Test recognition
python3 -m src.exocortex.cli speakers recognize ~/audio/test.wav
```

### How It Works

1. **Enrollment**: Extracts voice embeddings from audio samples
2. **Recognition**: Matches new audio against stored embeddings
3. **Mapping**: SPEAKER_00 → "Alice", SPEAKER_01 → "Bob", etc.

### Best Practices

- **Enroll multiple samples** (3-5) per speaker for better accuracy
- Use **clean audio** with minimal background noise
- **10-30 seconds** of speech per sample is ideal
- Samples should contain **natural speech** (not reading)

## Database Schema

Each memory in the exocortex database contains:

```python
{
  "memory_id": "mem_abc123",
  "text": "transcribed text",
  "language": "en",
  "speaker_label": "SPEAKER_00",  # From diarization
  "speaker_name": "Alice",        # From recognition (optional)
  "metadata": {
    "timestamp": "2023-03-16T22:19:50",
    "source_type": "audio",
    "source_id": "/path/to/audio.m4a"
  }
}
```

## Troubleshooting

### "No speaker segments detected"

**Cause:** Audio too short or no speech detected
**Solution:**
- Use audio files with at least 2-3 seconds of speech
- Ensure audio quality is good (not too noisy)
- Check that `min_speakers` in config isn't too high

### "Failed to load diarization pipeline"

**Cause:** Missing HuggingFace token or model not accepted
**Solution:**
1. Get token at: https://huggingface.co/settings/tokens
2. Accept models:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Update `hf_token` in config.yaml

### "Speaker recognition threshold too strict"

**Cause:** `recognition_threshold` set too high (default 0.75)
**Solution:** Lower it in config.yaml (e.g., 0.65 for more lenient matching)

## Performance Tips

### CPU vs GPU

**CPU (Apple Silicon M1/M2):**
- ~2-5 seconds per minute of audio
- Use `compute_type: "int8"` in config

**CUDA GPU:**
- ~0.5-1 second per minute of audio
- Use `compute_type: "float16"` in config

### Batch Processing

Process files in batches to avoid memory issues:

```bash
# Process in smaller batches
python3 reprocess_optimized.py 10  # First 10 files
python3 reprocess_optimized.py 20  # First 20 files
# etc.
```

## Example Workflow

### Complete Setup for New Project

```bash
# 1. Ingest new recordings with diarization
python3 -m src.exocortex.cli ingest ~/Recordings/*.m4a

# 2. Re-process existing memories
python3 reprocess_optimized.py

# 3. (Optional) Enroll known speakers
python3 speaker_enrollment.py enroll Alice ~/alice_sample.wav
python3 speaker_enrollment.py enroll Bob ~/bob_sample.wav

# 4. Enable speaker recognition in config.yaml
# (set speaker_recognition.enabled to true)

# 5. Search your memories
python3 -m src.exocortex.cli recall "machine learning"
```

## Files Created

- `test_diarization.py` - Test script for speaker diarization
- `reprocess_optimized.py` - Batch re-processing with diarization
- `speaker_enrollment.py` - Speaker recognition management
- `SPEAKER_ASSIGNMENT_GUIDE.md` - This guide

## Technical Details

### Models Used

- **Whisper**: Speech-to-text transcription (medium model)
- **Pyannote Diarization 3.1**: Speaker segmentation
- **Pyannote Segmentation 3.0**: Voice activity detection
- **Sentence Transformers**: Text embeddings for search (MiniLM-L6-v2)

### Storage

- **Qdrant**: Vector database for semantic search
- **SQLite**: Metadata and full memory storage
- **JSON**: Speaker profiles database

### Memory Structure

```
~/.exocortex/memories/
├── qdrant/           # Vector embeddings
├── metadata.db       # SQLite database
└── (future)
    └── speakers.json # Speaker profiles
```

## Next Steps

1. ✅ **Test complete** - Verify diarization works on sample files
2. ⏳ **Re-process existing** - Run `reprocess_optimized.py` on your 550 files
3. ✅ **Voice Memos ingestion** - Test with Voice Memos folder
4. 📋 **Optional**: Set up speaker recognition for persistent names

---

**Last Updated:** 2026-01-25
**Version:** 1.0.0
