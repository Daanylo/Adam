#!/usr/bin/env python3
"""
Interactive TTS Server for Sesame CSM Model

This server keeps the model loaded in memory and provides an interactive
command-line interface for generating audio without reloading the model.
"""

import os
import sys
import time
from typing import List, Optional
import torch
import torchaudio
from generator import load_csm_1b, Segment


class TTSServer:
    def __init__(self, device: str = "cuda", cache_dir: Optional[str] = None):
        """
        Initialize the TTS server.
        
        Args:
            device: Device to run the model on ('cuda', 'cpu', 'mps')
            cache_dir: Directory to cache downloaded models
        """
        self.device = device
        self.generator = None
        self.context: List[Segment] = []
        
        # Note: Cache environment variables should be set BEFORE importing
        # the modules, so we don't set them here - they should be set by the launcher script
        
        self._load_model()
    
    def _load_model(self):
        """Load the model once and keep it in memory."""
        print("🔄 Loading CSM model... This may take a moment.")
        start_time = time.time()
        
        try:
            self.generator = load_csm_1b(device=self.device)
            load_time = time.time() - start_time
            print(f"✅ Model loaded successfully in {load_time:.2f} seconds!")
            print(f"📱 Device: {self.device}")
            print(f"🎵 Sample rate: {self.generator.sample_rate} Hz")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def generate_audio(
        self, 
        text: str, 
        speaker: int = 0, 
        max_audio_length_ms: float = 5000,
        output_file: str = "output.wav"
    ) -> str:
        """
        Generate audio from text.
        
        Args:
            text: Text to synthesize
            speaker: Speaker ID (default: 0)
            max_audio_length_ms: Maximum audio length in milliseconds
            output_file: Output file path
            
        Returns:
            Path to the generated audio file
        """
        if not self.generator:
            raise RuntimeError("Model not loaded")
        
        print(f"🎤 Generating: '{text}'")
        start_time = time.time()
        
        try:
            audio = self.generator.generate(
                text=text,
                speaker=speaker,
                context=self.context,
                max_audio_length_ms=max_audio_length_ms,
            )
            
            # Save audio
            torchaudio.save(output_file, audio.unsqueeze(0).cpu(), self.generator.sample_rate)
            
            generation_time = time.time() - start_time
            duration = len(audio) / self.generator.sample_rate
            
            print(f"✅ Generated {duration:.2f}s audio in {generation_time:.2f}s")
            print(f"💾 Saved to: {output_file}")
            
            return output_file
            
        except Exception as e:
            print(f"❌ Error generating audio: {e}")
            return None
    
    def add_to_context(self, text: str, audio_file: str, speaker: int = 0):
        """Add a segment to the conversation context."""
        try:
            audio, sr = torchaudio.load(audio_file)
            if sr != self.generator.sample_rate:
                audio = torchaudio.functional.resample(
                    audio, orig_freq=sr, new_freq=self.generator.sample_rate
                )
            
            segment = Segment(
                speaker=speaker,
                text=text,
                audio=audio.squeeze(0)
            )
            self.context.append(segment)
            print(f"📝 Added to context: '{text}'")
            
        except Exception as e:
            print(f"❌ Error adding to context: {e}")
    
    def clear_context(self):
        """Clear the conversation context."""
        self.context.clear()
        print("🗑️ Context cleared")
    
    def show_context(self):
        """Show current context."""
        if not self.context:
            print("📝 Context is empty")
        else:
            print(f"📝 Context ({len(self.context)} segments):")
            for i, segment in enumerate(self.context):
                print(f"  {i+1}. Speaker {segment.speaker}: '{segment.text}'")
    
    def run_interactive(self):
        """Run the interactive command-line interface."""
        print("\n🎙️  Sesame CSM Interactive TTS Server")
        print("=" * 50)
        print("Commands:")
        print("  generate <text>     - Generate audio from text")
        print("  speaker <id>        - Set speaker ID (default: 0)")
        print("  length <ms>         - Set max audio length in ms (default: 5000)")
        print("  output <filename>   - Set output filename (default: output.wav)")
        print("  context             - Show current context")
        print("  clear               - Clear context")
        print("  help                - Show this help")
        print("  quit/exit           - Exit server")
        print("=" * 50)
        
        # Default settings
        current_speaker = 0
        current_length = 5000
        current_output = "output.wav"
        
        while True:
            try:
                user_input = input("\n🎤 > ").strip()
                
                if not user_input:
                    continue
                
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if command in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif command == 'help':
                    print("\nCommands:")
                    print("  generate <text>     - Generate audio from text")
                    print("  speaker <id>        - Set speaker ID")
                    print("  length <ms>         - Set max audio length in ms")
                    print("  output <filename>   - Set output filename")
                    print("  context             - Show current context")
                    print("  clear               - Clear context")
                    print("  help                - Show this help")
                    print("  quit/exit           - Exit server")
                
                elif command == 'generate':
                    if not args:
                        print("❌ Please provide text to generate")
                        continue
                    
                    self.generate_audio(
                        text=args,
                        speaker=current_speaker,
                        max_audio_length_ms=current_length,
                        output_file=current_output
                    )
                
                elif command == 'speaker':
                    try:
                        current_speaker = int(args)
                        print(f"🎭 Speaker set to: {current_speaker}")
                    except ValueError:
                        print("❌ Invalid speaker ID. Please provide a number.")
                
                elif command == 'length':
                    try:
                        current_length = float(args)
                        print(f"⏱️ Max length set to: {current_length}ms")
                    except ValueError:
                        print("❌ Invalid length. Please provide a number.")
                
                elif command == 'output':
                    if args:
                        current_output = args
                        print(f"💾 Output file set to: {current_output}")
                    else:
                        print("❌ Please provide a filename")
                
                elif command == 'context':
                    self.show_context()
                
                elif command == 'clear':
                    self.clear_context()
                
                else:
                    # Treat unknown commands as text to generate
                    self.generate_audio(
                        text=user_input,
                        speaker=current_speaker,
                        max_audio_length_ms=current_length,
                        output_file=current_output
                    )
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive TTS Server for Sesame CSM")
    parser.add_argument(
        "--device", 
        default="cuda", 
        choices=["cuda", "cpu", "mps"],
        help="Device to run the model on"
    )
    parser.add_argument(
        "--cache-dir",
        default="./model_cache",
        help="Directory to cache downloaded models"
    )
    parser.add_argument(
        "--text",
        help="Generate audio for this text and exit (non-interactive mode)"
    )
    parser.add_argument(
        "--output",
        default="output.wav",
        help="Output file for non-interactive mode"
    )
    
    args = parser.parse_args()
    
    # Initialize server
    server = TTSServer(device=args.device, cache_dir=args.cache_dir)
    
    if args.text:
        # Non-interactive mode
        server.generate_audio(args.text, output_file=args.output)
    else:
        # Interactive mode
        server.run_interactive()


if __name__ == "__main__":
    main()
