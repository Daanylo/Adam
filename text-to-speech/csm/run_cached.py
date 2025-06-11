#!/usr/bin/env python3
"""
Quick TTS script with natural caching - improved version of run.py
"""

from generator import load_csm_1b
import torchaudio

print("🔄 Loading model...")
generator = load_csm_1b(device="cuda")

print("🎤 Generating audio...")
audio = generator.generate(
    text="Hello from Sesame.",
    speaker=0,
    context=[],
    max_audio_length_ms=5_000,
)

print("💾 Saving audio...")
torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
print("✅ Done! Audio saved to audio.wav")
