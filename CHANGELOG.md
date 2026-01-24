# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project documentation (CLAUDE.md, CONTRIBUTING.md, ARCHITECTURE.md)
- Claude Code configuration (.claude/settings.json)
- Auto-formatting hook with Black
- Auto-linting hook with Ruff

## [0.1.0] - 2026-01-23

### Added
- Real-time audio capture with sounddevice
- Whisper-based transcription using faster-whisper
- NLLB-200 based translation (200+ languages)
- Multiple OBS output methods:
  - Text file for OBS Text (GDI+) source
  - WebSocket server for programmatic access
  - HTTP server with styled overlay
- Auto-clear subtitles after silence
- CLI with Click framework:
  - `start` - Start live transcription/translation
  - `list-devices` - List audio devices
  - `list-models` - List Whisper models
- YAML configuration support
- Rich terminal output
- GPU acceleration support (CUDA 12+)
- Auto language detection
- Configurable chunk duration for latency tuning

### Development
- Black code formatting (line-length: 100)
- Ruff linting
- pytest test framework
- Type hints throughout codebase
- Dataclass-based configuration

[Unreleased]: https://github.com/kqb/live-translation-local/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kqb/live-translation-local/releases/tag/v0.1.0
