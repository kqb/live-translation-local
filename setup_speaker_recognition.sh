#!/bin/bash
# Setup script for speaker recognition with sample enrollments

echo "🎙️  Speaker Recognition Setup"
echo "================================"
echo ""

# Enable speaker recognition in config
echo "📝 Enabling speaker recognition in config.yaml..."
sed -i '' 's/enabled: false      # Enable persistent speaker recognition/enabled: true       # Enable persistent speaker recognition/' config.yaml

echo "✓ Speaker recognition enabled"
echo ""

# Create data directory
mkdir -p data
echo "✓ Created data directory for speaker database"
echo ""

echo "📚 Speaker Enrollment Guide"
echo "============================"
echo ""
echo "To enroll speakers, you need audio samples with clean single-speaker speech."
echo ""
echo "Commands:"
echo ""
echo "  # List enrolled speakers"
echo "  python3 speaker_enrollment.py list-speakers"
echo ""
echo "  # Enroll a speaker (use 10-30 second audio clips with clear speech)"
echo "  python3 speaker_enrollment.py enroll \"Your Name\" ~/audio/sample.wav"
echo ""
echo "  # Add more samples for better accuracy (recommended: 3-5 samples per speaker)"
echo "  python3 speaker_enrollment.py enroll \"Your Name\" ~/audio/sample2.wav"
echo ""
echo "  # Test recognition on any audio file"
echo "  python3 speaker_enrollment.py recognize ~/audio/test.wav"
echo ""
echo "Via exocortex CLI:"
echo "  python3 -m src.exocortex.cli speakers list"
echo "  python3 -m src.exocortex.cli speakers enroll \"Name\" audio.wav"
echo "  python3 -m src.exocortex.cli speakers recognize audio.wav"
echo ""
echo "💡 Tips for Best Results:"
echo "  • Use 3-5 samples per speaker for better accuracy"
echo "  • Each sample should be 10-30 seconds of natural speech"
echo "  • Use clean audio with minimal background noise"
echo "  • Samples can be from different recordings/times"
echo ""
echo "✅ Setup complete! Ready to enroll speakers."
