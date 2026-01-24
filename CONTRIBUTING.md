# Contributing to Live Translator for OBS Studio

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other contributors

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv, conda, etc.)
- (Optional) CUDA 12+ for GPU acceleration

### Development Setup

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/live-translation-local.git
cd live-translation-local
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install development dependencies**

```bash
pip install -e ".[dev]"
```

4. **Verify installation**

```bash
live-translator --version
live-translator list-devices
```

## Development Workflow

### Branch Naming

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/improvements

### Making Changes

1. **Create a feature branch**

```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**
   - Write clean, readable code
   - Follow the code style guide (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Format and lint your code**

```bash
# Auto-format with Black
black .

# Check and fix linting issues
ruff check . --fix

# Verify no issues remain
ruff check .
```

4. **Run tests**

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_transcriber.py -v
```

5. **Commit your changes**

```bash
git add .
git commit -m "Add feature: description of what you added"
```

Use clear, descriptive commit messages:
- Start with a verb (Add, Fix, Update, Refactor, etc.)
- Keep the first line under 72 characters
- Add detailed description in the body if needed

6. **Push and create a pull request**

```bash
git push origin feature/your-feature-name
```

Then open a pull request on GitHub with:
- Clear description of the changes
- Reference any related issues
- Screenshots/videos for UI changes
- Test results if applicable

## Code Style Guide

### Python Code Standards

**Follow PEP 8 with project-specific conventions:**

- **Line length**: 100 characters (Black default for this project)
- **Formatting**: Use Black (automatic formatting)
- **Linting**: Use Ruff (catches common errors and style issues)
- **Type hints**: Required for all public functions
- **Docstrings**: Google-style for all public classes and functions

### Type Hints

Use modern Python 3.10+ type hints:

```python
# Good
def process_audio(data: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Process audio data."""
    pass

# Avoid (old-style)
from typing import Dict, Any
def process_audio(data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    pass
```

### Imports

Organize imports in three groups:

```python
# Standard library
import sys
from pathlib import Path
from typing import Optional

# Third-party packages
import numpy as np
from rich.console import Console

# Local modules
from .audio_capture import AudioCapture
from .transcriber import Transcriber
```

### Docstrings

Use Google-style docstrings:

```python
def transcribe_audio(audio: np.ndarray, language: Optional[str] = None) -> str:
    """Transcribe audio to text using Whisper.

    Args:
        audio: Audio data as numpy array.
        language: Optional language code (e.g., 'en', 'es'). Auto-detect if None.

    Returns:
        Transcribed text.

    Raises:
        TranscriptionError: If transcription fails.
    """
    pass
```

### Dataclasses

Use dataclasses for configuration and data structures:

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class MyConfig:
    """Configuration for my component."""

    required_field: str
    optional_field: int = 42
    list_field: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MyConfig":
        """Create config from dictionary."""
        return cls(
            required_field=data["required_field"],
            optional_field=data.get("optional_field", 42),
            list_field=data.get("list_field", [])
        )
```

### Error Handling

- Use specific exception types
- Provide helpful error messages
- Use Rich console for user-facing errors

```python
from rich.console import Console

console = Console()

try:
    result = risky_operation()
except ValueError as e:
    console.print(f"[red]Error:[/red] Invalid value: {e}")
    raise
```

### Async/Await

- Use `async def` for I/O-bound operations
- Use `asyncio.create_task()` for concurrent tasks
- Always clean up resources in `finally` blocks

```python
async def process_stream():
    """Process audio stream."""
    try:
        while running:
            data = await queue.get()
            await process_data(data)
    finally:
        await cleanup()
```

## Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Mirror source structure: `tests/test_transcriber.py` for `src/transcriber.py`
- Use pytest fixtures for common setup
- Test both success and failure cases
- Use meaningful test names

```python
import pytest
from src.transcriber import Transcriber

def test_transcriber_basic_audio():
    """Test transcription with basic audio input."""
    transcriber = Transcriber()
    result = transcriber.transcribe(sample_audio)
    assert result.text is not None
    assert len(result.text) > 0

def test_transcriber_invalid_audio():
    """Test transcriber handles invalid audio gracefully."""
    transcriber = Transcriber()
    with pytest.raises(ValueError):
        transcriber.transcribe(invalid_audio)
```

### Async Tests

Use `pytest-asyncio` for async tests:

```python
import pytest

@pytest.mark.asyncio
async def test_websocket_server():
    """Test WebSocket server communication."""
    server = await start_websocket_server()
    # Test logic here
    await server.close()
```

### Test Coverage

- Aim for >80% code coverage
- Focus on critical paths and edge cases
- Don't sacrifice readability for coverage

## Adding New Features

### Before You Start

1. **Check existing issues** - Someone might already be working on it
2. **Open an issue** - Discuss the feature before implementing
3. **Get feedback** - Ensure the feature aligns with project goals

### Feature Checklist

When adding a new feature:

- [ ] Add implementation code
- [ ] Add configuration options (if needed)
- [ ] Update `config.yaml` with new settings
- [ ] Add tests for the feature
- [ ] Update documentation (README, CLAUDE.md, etc.)
- [ ] Add docstrings to new functions/classes
- [ ] Format code with Black
- [ ] Run linter (Ruff) and fix issues
- [ ] Verify tests pass
- [ ] Update CHANGELOG (if exists)

### Example: Adding a New Output Method

1. **Add configuration**

```python
# src/obs_output.py
@dataclass
class OutputConfig:
    # Existing fields...
    my_new_output_enabled: bool = False
    my_new_output_port: int = 9999
```

2. **Implement the feature**

```python
class OBSOutput:
    async def _start_my_new_output(self):
        """Start my new output method."""
        # Implementation here
        pass

    async def start(self):
        # Existing code...
        if self.config.my_new_output_enabled:
            await self._start_my_new_output()
```

3. **Add tests**

```python
# tests/test_obs_output.py
@pytest.mark.asyncio
async def test_my_new_output():
    """Test my new output method."""
    config = OutputConfig(my_new_output_enabled=True)
    output = OBSOutput(config)
    await output.start()
    # Assertions here
```

4. **Update documentation**

- Add section to README.md
- Update CLAUDE.md with technical details
- Add example to config.yaml

## Reporting Bugs

### Before Reporting

1. **Search existing issues** - The bug might already be reported
2. **Update to latest version** - It might be fixed already
3. **Test in a clean environment** - Rule out local configuration issues

### Bug Report Template

Include:

- **Environment**: OS, Python version, GPU/CPU
- **Steps to reproduce**: Exact commands and configuration
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Logs**: Any error messages or stack traces
- **Configuration**: Relevant parts of `config.yaml`

Example:

```markdown
**Environment:**
- OS: Ubuntu 22.04
- Python: 3.11.5
- GPU: NVIDIA RTX 3060 (CUDA 12.1)

**Steps to reproduce:**
1. Install with `pip install -e .`
2. Run `live-translator run -m base`
3. Speak into microphone

**Expected:** Subtitles appear in subtitles.txt
**Actual:** Error: "CUDA out of memory"

**Logs:**
[paste error message]

**Config:**
[paste relevant config.yaml sections]
```

## Documentation

### What to Document

- New features and configuration options
- Breaking changes
- Performance considerations
- Examples and use cases

### Where to Document

- **README.md**: User-facing features, installation, quick start
- **CLAUDE.md**: Technical context for Claude Code, architecture details
- **Code comments**: Complex logic, non-obvious decisions
- **Docstrings**: All public functions and classes

## Performance Considerations

When adding features:

- **Profile before optimizing** - Don't guess, measure
- **Consider memory usage** - Large models can use 4-8GB RAM
- **Test on low-end hardware** - Not everyone has a GPU
- **Benchmark latency** - Real-time transcription requires low latency

## Questions?

- **General questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Security issues**: Email maintainers privately
- **Feature requests**: Open a GitHub Issue with [Feature Request] tag

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md (when created)
- Mentioned in release notes
- Credited in commit messages (Co-authored-by)

Thank you for contributing to Live Translator for OBS Studio! 🎙️✨
