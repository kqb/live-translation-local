# Even G2 Smart Glasses Integration

Your Live Translator now supports **Even Realities G2 smart glasses** for real-time translation display directly on your AR glasses!

## What's New

Instead of relying on the limited built-in Even AI translation, you now have:

- ✅ **High-Quality Translation**: NLLB-200 with 200+ language support
- ✅ **Advanced Transcription**: Whisper with hallucination filtering
- ✅ **Speaker Recognition**: Persistent voice profiles (Alice, Bob, etc.)
- ✅ **Session Logging**: Record all conversations for later analysis
- ✅ **Dual Output**: Glasses + OBS streaming simultaneously

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/yuzucchi/Documents/live-translation-local
pip install bleak soundfile pyannote.audio
```

### 2. Configure G2 Output

Edit [config.yaml](config.yaml):

```yaml
g2:
  enabled: true       # Enable G2 output
  mode: "notification"  # Use notification display
  auto_connect: true  # Auto-connect on startup
  use_right: false    # Use left eye (default)
  display_format: "both"  # Show both original and translation
```

### 3. Enable Features (Optional)

```yaml
# Speaker diarization (detect who's speaking)
diarization:
  enabled: true
  hf_token: "hf_xxxxxxxxxxxxx"  # From https://huggingface.co/settings/tokens

# Speaker recognition (persistent names)
speaker_recognition:
  enabled: true
  database_path: "./data/speakers.json"

# Session logging
logging:
  enabled: true
  log_dir: "./data/sessions"
```

### 4. Run

```bash
# Make sure G2 glasses are powered on and nearby
live-translator run
```

The translator will:
1. Scan for your G2 glasses (LEFT and RIGHT eyes)
2. Connect via Bluetooth
3. Authenticate
4. Start displaying translations in real-time!

## Display Formats

Control what appears on your glasses with `display_format`:

### `"both"` (Default)
```
[Alice]
Hello, how are you?
→ こんにちは、お元気ですか？
```

### `"original"`
```
[Speech]
Hello, how are you?
```

### `"translated"`
```
[Translation]
こんにちは、お元気ですか？
```

## Output Modes

### Notification Mode (Recommended)
- **Mode**: `notification`
- **Display**: Push notifications that appear at the top
- **Best for**: Conversations, live translation
- **Updates**: Every new utterance

### Teleprompter Mode (Future)
- **Mode**: `teleprompter`
- **Display**: Scrolling text display
- **Best for**: Long-form content, speeches
- **Status**: Coming soon

## Speaker Recognition Setup

Make your glasses show names instead of "SPEAKER_00":

```bash
# 1. Enable diarization first (in config.yaml)
diarization:
  enabled: true
  hf_token: "your_hf_token"

# 2. Enable speaker recognition
speaker_recognition:
  enabled: true

# 3. Enroll speakers
live-translator enroll Alice
# Speak for 5 seconds...

live-translator enroll Bob
# Speak for 5 seconds...

# 4. List enrolled speakers
live-translator speakers list

# 5. Run with persistent speakers!
live-translator run
```

Now your glasses will show:
```
[Alice]
Hello everyone
→ みなさん、こんにちは

[Bob]
How's it going?
→ 調子はどうですか？
```

## Architecture

```
Microphone → Whisper (transcribe) → Diarization (who?) → Recognition (name?)
                                                              ↓
                                                         NLLB-200 (translate)
                                                              ↓
                                        ┌─────────────────────┴──────────────┐
                                        ↓                                    ↓
                                   G2 Glasses                           OBS Output
                            (BLE notifications)                    (text/websocket)
```

## Troubleshooting

### G2 glasses not found

```
[G2] ERROR: Could not find both G2 eyes!
```

**Solution:**
- Make sure glasses are powered on
- Glasses should show in Bluetooth settings as "G2_L_XXXX" and "G2_R_XXXX"
- Try increasing scan timeout in code
- Make sure glasses aren't already connected to phone

### Connection timeout

```
[G2] Connection failed: timeout
```

**Solution:**
- Move glasses closer to computer
- Restart glasses (power off/on)
- Check Bluetooth is enabled on computer
- Disconnect glasses from Even app first

### Notifications not appearing

**Checklist:**
- G2 output enabled in config: `g2.enabled: true`
- Connection successful: Look for `[G2] Connected and authenticated!`
- Glasses awake: Tap side button to wake display
- Check battery: Low battery may affect display

### Speaker recognition not working

**Requirements:**
1. Diarization enabled with valid HF token
2. Speaker database exists at configured path
3. Speakers enrolled with `live-translator enroll NAME`
4. Good quality audio (clear voice samples)

## Performance

### Latency
- **Transcription**: ~0.5-2s (depends on Whisper model)
- **Translation**: ~0.1-0.5s (NLLB-200)
- **G2 Display**: ~0.2-0.5s (BLE notification)
- **Total**: ~1-3s end-to-end

### Battery Impact
- G2 glasses: Moderate (notifications use less power than constant display)
- Computer: Minimal (BLE is low-power)

### Tips for Lower Latency
- Use smaller Whisper model: `model: "base"` instead of `"medium"`
- Disable speaker recognition if not needed
- Disable session logging if not needed
- Use `backend: "whisper-cpp"` on Apple Silicon

## Integration with Even G2 Protocol

This integration uses the reverse-engineered Even G2 BLE protocol from:
https://github.com/i-soxi/even-g2-protocol

### Key Components
- **BLE Connection**: Standard Bluetooth Low Energy
- **Authentication**: 3-packet handshake sequence
- **Notifications**: JSON payload with CRC32C checksum
- **Multi-packet**: Supports messages >234 bytes

### Protocol Files
The integration cloned the protocol repository to:
`/Users/yuzucchi/Documents/live-translation-local/even-g2-protocol/`

You can explore the protocol documentation there.

## Future Enhancements

Potential improvements:

- [ ] Teleprompter mode for long-form content
- [ ] Customizable notification styling
- [ ] Offline translation support
- [ ] Multi-language simultaneous translation
- [ ] Voice activity detection for lower latency
- [ ] Gesture controls (if G2 supports it)

## Credits

- **Even G2 Protocol**: Reverse engineered by the EvenRealities community
- **Discord**: Join [EvenRealities Discord](https://discord.gg/arDkX3pr) for protocol updates
- **Live Translator**: Your advanced translation pipeline

## Comparison: Built-in Even AI vs Live Translator

| Feature | Even AI (Built-in) | Live Translator + G2 |
|---------|-------------------|---------------------|
| Translation Quality | Basic | NLLB-200 (production-grade) |
| Languages | Limited | 200+ languages |
| Speaker Recognition | No | Yes (with names) |
| Hallucination Filtering | No | Yes (advanced) |
| Session Logging | No | Yes (audio + metadata) |
| OBS Integration | No | Yes (streaming support) |
| Customization | None | Full control |
| Offline Mode | N/A | Coming soon |

---

**Enjoy your enhanced G2 translation experience!** 🎉

For issues or questions, check the main [README.md](README.md) or the Even G2 protocol documentation.
