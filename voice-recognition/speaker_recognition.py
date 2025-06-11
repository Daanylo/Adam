"""
Speaker Recognition Script

This script uses the 'resemblyzer' library for speaker embedding and recognition.
It can enroll new speakers and recognize who is speaking from a short audio clip.

Dependencies:
- resemblyzer
- sounddevice
- numpy
- scipy

Install with:
    pip install resemblyzer sounddevice numpy scipy

Usage:
- Run the script and follow the prompts to enroll a new speaker or recognize a speaker from microphone input.
- Audio is recorded from your default microphone.
"""

import os
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
import sounddevice as sd
from scipy.io.wavfile import write, read
import time
import pickle

SPEAKER_DB = "speakers.pkl"
SAMPLE_RATE = 16000
RECORD_SECONDS = 4

encoder = VoiceEncoder()

# Load or initialize speaker database
def load_speakers():
    if os.path.exists(SPEAKER_DB):
        with open(SPEAKER_DB, "rb") as f:
            return pickle.load(f)
    return {}

def save_speakers(speakers):
    with open(SPEAKER_DB, "wb") as f:
        pickle.dump(speakers, f)

def record_audio(filename, seconds=RECORD_SECONDS, sample_rate=SAMPLE_RATE):
    print(f"Recording {seconds} seconds of audio...")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    write(filename, sample_rate, audio)
    print(f"Saved recording to {filename}")
    return filename

def enroll_speaker(name, speakers):
    filename = f"{name}_enroll.wav"
    record_audio(filename)
    wav = preprocess_wav(filename)
    embed = encoder.embed_utterance(wav)
    speakers[name] = embed
    save_speakers(speakers)
    print(f"Enrolled speaker: {name}")


def recognize_speaker(speakers):
    filename = "test_speaker.wav"
    record_audio(filename)
    wav = preprocess_wav(filename)
    embed = encoder.embed_utterance(wav)
    if not speakers:
        print("No speakers enrolled yet.")
        return None
    # Compare to all enrolled speakers
    best_name = None
    best_score = -1
    for name, ref_embed in speakers.items():
        score = np.dot(embed, ref_embed) / (np.linalg.norm(embed) * np.linalg.norm(ref_embed))
        print(f"Similarity to {name}: {score:.3f}")
        if score > best_score:
            best_score = score
            best_name = name
    if best_score > 0.75:
        print(f"Recognized as: {best_name} (score: {best_score:.3f})")
        return best_name
    else:
        print("Speaker not recognized.")
        return None

def main():
    speakers = load_speakers()
    while True:
        print("\nSpeaker Recognition Menu:")
        print("1. Enroll new speaker")
        print("2. Recognize speaker")
        print("3. List enrolled speakers")
        print("4. Exit")
        choice = input("Select an option: ").strip()
        if choice == "1":
            name = input("Enter speaker name: ").strip()
            if name:
                enroll_speaker(name, speakers)
        elif choice == "2":
            recognize_speaker(speakers)
        elif choice == "3":
            print("Enrolled speakers:", list(speakers.keys()))
        elif choice == "4":
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
