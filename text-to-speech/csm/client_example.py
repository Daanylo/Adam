#!/usr/bin/env python3
"""
Example client for the TTS API

This shows how to use the TTS API from Python code or LLM applications.
"""

import requests
import json
import time
from pathlib import Path

class TTSClient:
    def __init__(self, base_url: str = "http://127.0.0.1:5000"):
        """
        Initialize TTS API client.
        
        Args:
            base_url: Base URL of the TTS API server
        """
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> dict:
        """Check if the API server is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def is_ready(self) -> bool:
        """Check if the model is ready for generation."""
        try:
            response = requests.get(f"{self.base_url}/ready")
            data = response.json()
            return data.get("ready", False)
        except requests.RequestException:
            return False
    
    def wait_until_ready(self, timeout: int = 60) -> bool:
        """Wait until the model is ready, with timeout."""
        print("⏳ Waiting for model to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_ready():
                print("✅ Model is ready!")
                return True
            print("🔄 Model still loading...")
            time.sleep(2)
        
        print("❌ Timeout waiting for model to be ready")
        return False
    
    def generate_speech(
        self, 
        text: str, 
        speaker: int = 0, 
        max_length_ms: int = 10000,
        download: bool = True,
        save_path: str = None
    ) -> dict:
        """
        Generate speech from text.
        
        Args:
            text: Text to synthesize
            speaker: Speaker ID (0, 1, 2, etc.)
            max_length_ms: Maximum audio length in milliseconds
            download: Whether to download the audio file
            save_path: Local path to save the audio file
            
        Returns:
            Dictionary with generation results
        """
        try:
            payload = {
                "text": text,
                "speaker": speaker,
                "max_length_ms": max_length_ms,
                "return_audio": True
            }
            
            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                return {"error": f"API error: {response.status_code}", "details": response.text}
            
            result = response.json()
            
            # Download audio file if requested
            if download and "filename" in result:
                filename = result["filename"]
                audio_url = f"{self.base_url}/audio/{filename}"
                
                audio_response = requests.get(audio_url)
                if audio_response.status_code == 200:
                    # Determine save path
                    if save_path:
                        local_path = Path(save_path)
                    else:
                        local_path = Path(filename)
                    
                    # Save audio file
                    with open(local_path, 'wb') as f:
                        f.write(audio_response.content)
                    
                    result["local_path"] = str(local_path)
                    print(f"💾 Audio saved to: {local_path}")
                else:
                    result["download_error"] = f"Failed to download audio: {audio_response.status_code}"
            
            return result
            
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def list_files(self) -> dict:
        """List all generated audio files."""
        try:
            response = requests.get(f"{self.base_url}/files")
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def get_info(self) -> dict:
        """Get server information."""
        try:
            response = requests.get(f"{self.base_url}/info")
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

def main():
    """Example usage of the TTS client."""
    
    # Initialize client
    client = TTSClient("http://127.0.0.1:5000")
    
    # Check health
    print("🏥 Checking server health...")
    health = client.health_check()
    print(f"Health: {health}")
    
    # Wait for model to be ready
    if not client.wait_until_ready():
        print("❌ Model not ready, exiting")
        return
    
    # Generate speech
    print("\n🎤 Generating speech...")
    result = client.generate_speech(
        text="Hello! This is a test of the TTS API. How does it sound?",
        speaker=0,
        max_length_ms=15000
    )
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Generated audio:")
        print(f"   Duration: {result['duration_seconds']} seconds")
        print(f"   Generation time: {result['generation_time_seconds']} seconds")
        if "local_path" in result:
            print(f"   Saved to: {result['local_path']}")
    
    # List files
    print("\n📁 Listing audio files...")
    files = client.list_files()
    if "files" in files:
        print(f"Found {files['total_count']} files:")
        for file_info in files['files'][:5]:  # Show first 5
            print(f"   {file_info['filename']} ({file_info['size_bytes']} bytes)")
    
    # Get server info
    print("\n📊 Server information...")
    info = client.get_info()
    print(f"Device: {info.get('device', 'unknown')}")
    print(f"Model loaded: {info.get('model_loaded', False)}")
    if "sample_rate" in info:
        print(f"Sample rate: {info['sample_rate']} Hz")

# Example for LLM integration
def llm_speak(text: str, voice_id: int = 0) -> str:
    """
    Simple function that an LLM can call to speak text.
    
    Args:
        text: Text to speak
        voice_id: Voice/speaker ID
        
    Returns:
        Path to the generated audio file, or error message
    """
    client = TTSClient()
    
    # Quick ready check
    if not client.is_ready():
        return "Error: TTS model not ready"
    
    # Generate speech
    result = client.generate_speech(text, speaker=voice_id)
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    return result.get("local_path", result.get("filename", "Audio generated successfully"))

if __name__ == "__main__":
    main()
