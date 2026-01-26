# Exocortex Integration with Live Translator

## Overview

The exocortex memory system is now integrated with the live translation pipeline. When enabled, it automatically captures and indexes English transcriptions for later semantic search and recall.

## Features

- **Automatic Memory Capture**: English transcriptions are automatically indexed as memories
- **Speaker Information**: Captures speaker labels and names when diarization/recognition is enabled
- **Semantic Search**: Use the `exo` CLI to search your conversation history
- **Phase 1**: English-only memory capture (per design document)

## Configuration

Edit `config.yaml` to enable the exocortex:

```yaml
exocortex:
  enabled: true  # Set to true to enable memory indexing
  storage_path: "./data/memories"  # Where to store memories
  auto_index: true  # Auto-index English transcriptions
```

## Usage

### 1. Enable Exocortex

Edit `config.yaml` and set `exocortex.enabled: true`

### 2. Run Live Translator

```bash
live-translator run
```

The pipeline will:
- Start capturing audio as usual
- Transcribe and translate as normal
- **Automatically** index English transcriptions as memories

### 3. Speak and Stop

Speak some English text, then stop with `Ctrl+C`. The exocortex will save all captured memories.

### 4. Search Your Memories

Use the `exo` CLI to search your memories:

```bash
# View statistics
exo stats

# Search for specific content
exo recall "machine learning"

# View recent memories
exo recent
```

## Architecture

### Memory Flow

```
Audio → Transcription → [Memory Capture] → Exocortex Index
                      ↓
                  Translation → Output
```

### Components

1. **Pipeline Integration** (`src/pipeline.py`):
   - `ExocortexConfig`: Configuration dataclass
   - `_capture_memory()`: Captures transcriptions as memories
   - Initializes `MemoryIndexer` when enabled
   - Proper cleanup on shutdown

2. **Exocortex Core** (`src/exocortex/`):
   - `memory.py`: Memory model and metadata
   - `indexer.py`: Memory indexing logic
   - `storage.py`: Qdrant vector storage + SQLite metadata
   - `embedder.py`: Sentence embeddings (all-MiniLM-L6-v2)
   - `cli.py`: CLI commands for memory management

### Memory Metadata

Each memory includes:
- `memory_id`: Unique identifier
- `timestamp`: When it was captured
- `source_type`: "audio" (from live transcription)
- `source_id`: Session identifier
- `text`: Transcribed text
- `language`: "en" (Phase 1 only)
- `speaker_label`: SPEAKER_00, SPEAKER_01, etc. (if diarization enabled)
- `speaker_name`: Recognized speaker name (if recognition enabled)

## Phase 1 Limitations

- **English only**: Only English transcriptions are indexed
- **No Japanese**: Japanese audio is not indexed (per design doc)
- **Audio source only**: Only live transcription memories (no manual capture yet)

## Testing

Run the integration test:

```bash
python3 test_exocortex_integration.py
```

Expected output:
```
✓ Config loading test passed
✓ All exocortex imports successful
✓ Pipeline config test passed
✓ All tests passed!
```

## Performance

- **Memory overhead**: ~200MB for embedding model (first run downloads)
- **Indexing speed**: ~100-200ms per memory (includes embedding generation)
- **Storage**: ~1KB per memory (text + metadata + embedding)
- **Search speed**: <50ms for typical queries

## Troubleshooting

### Exocortex not indexing

Check that:
1. `exocortex.enabled: true` in `config.yaml`
2. Speaking **English** (only English is indexed in Phase 1)
3. Check console output for "Indexed memory: ..." messages

### Import errors

If you see import errors:
```bash
pip install sentence-transformers qdrant-client
```

### Storage path issues

Default storage path is `./data/memories`. Ensure:
- Directory is writable
- Sufficient disk space
- Not using network/cloud storage (Qdrant requires local disk)

## Future Enhancements (Phase 2+)

- Japanese memory capture
- Multi-language semantic search
- Manual memory creation via CLI
- Memory tags and categories
- Memory export/import
- Context injection into translations

## Related Files

- `/Users/yuzucchi/Documents/live-translation-local/config.yaml` - Configuration
- `/Users/yuzucchi/Documents/live-translation-local/src/pipeline.py` - Pipeline integration
- `/Users/yuzucchi/Documents/live-translation-local/src/exocortex/` - Exocortex core
- `/Users/yuzucchi/Documents/live-translation-local/test_exocortex_integration.py` - Integration tests
