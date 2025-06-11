#!/usr/bin/env python3
"""
REST API Server for Sesame CSM TTS

This creates a REST API that can be called by LLMs or other applications
to generate speech from text.
"""

import os
import sys
import time
import uuid
from typing import List, Optional
from pathlib import Path
import torch
import torchaudio
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import threading
import logging

# Import our TTS components
from generator import load_csm_1b, Segment

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class TTSAPIServer:
    def __init__(self, device: str = "cuda", output_dir: str = "./audio_output"):
        """
        Initialize the TTS API server.
        
        Args:
            device: Device to run the model on ('cuda', 'cpu', 'mps')
            output_dir: Directory to store generated audio files
        """
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.generator = None
        self.model_loaded = False
        self.loading_lock = threading.Lock()
        
        # Load model in background
        threading.Thread(target=self._load_model_background, daemon=True).start()
    
    def _load_model_background(self):
        """Load the model in the background."""
        with self.loading_lock:
            if self.model_loaded:
                return
                
            logger.info("🔄 Loading CSM model...")
            start_time = time.time()
            
            try:
                self.generator = load_csm_1b(device=self.device)
                load_time = time.time() - start_time
                logger.info(f"✅ Model loaded successfully in {load_time:.2f} seconds!")
                logger.info(f"📱 Device: {self.device}")
                logger.info(f"🎵 Sample rate: {self.generator.sample_rate} Hz")
                self.model_loaded = True
            except Exception as e:
                logger.error(f"❌ Error loading model: {e}")
                sys.exit(1)
    
    def is_ready(self):
        """Check if the model is loaded and ready."""
        return self.model_loaded
    
    def generate_audio(
        self, 
        text: str, 
        speaker: int = 0, 
        max_audio_length_ms: float = 10000,
        return_file: bool = True
    ) -> dict:
        """
        Generate audio from text.
        
        Args:
            text: Text to synthesize
            speaker: Speaker ID (default: 0)
            max_audio_length_ms: Maximum audio length in milliseconds
            return_file: Whether to save and return file path
            
        Returns:
            Dictionary with generation results
        """
        if not self.model_loaded:
            return {
                "error": "Model not loaded yet. Please wait and try again.",
                "status": "loading"
            }
        
        try:
            logger.info(f"🎤 Generating audio for: '{text}'")
            start_time = time.time()
            
            audio = self.generator.generate(
                text=text,
                speaker=speaker,
                context=[],  # No context for API calls
                max_audio_length_ms=max_audio_length_ms,
            )
            
            generation_time = time.time() - start_time
            duration = len(audio) / self.generator.sample_rate
            
            result = {
                "status": "success",
                "text": text,
                "speaker": speaker,
                "duration_seconds": round(duration, 2),
                "generation_time_seconds": round(generation_time, 2),
                "sample_rate": self.generator.sample_rate
            }
            
            if return_file:
                # Generate unique filename
                filename = f"tts_{uuid.uuid4().hex[:8]}_{int(time.time())}.wav"
                filepath = (self.output_dir / filename).resolve()
                
                # Save audio file
                torchaudio.save(str(filepath), audio.unsqueeze(0).cpu(), self.generator.sample_rate)
                logger.info(f"[DEBUG] Absolute save path: {filepath}")
                result["filename"] = filename
                result["filepath"] = str(filepath)
                
                logger.info(f"✅ Generated {duration:.2f}s audio in {generation_time:.2f}s")
                logger.info(f"💾 Saved to: {filename}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error generating audio: {e}")
            return {
                "error": str(e),
                "status": "error"
            }

# Global TTS server instance
tts_server = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    global tts_server
    return jsonify({
        "status": "healthy",
        "model_loaded": tts_server.is_ready() if tts_server else False,
        "device": tts_server.device if tts_server else "unknown"
    })

@app.route('/ready', methods=['GET'])
def ready_check():
    """Check if the model is ready for generation."""
    global tts_server
    if not tts_server:
        return jsonify({"ready": False, "message": "Server not initialized"}), 503
    
    return jsonify({
        "ready": tts_server.is_ready(),
        "message": "Model ready" if tts_server.is_ready() else "Model still loading"
    })

@app.route('/generate', methods=['POST'])
def generate_speech():
    """
    Generate speech from text.
    
    Expected JSON payload:
    {
        "text": "Hello world",
        "speaker": 0,  // optional, default 0
        "max_length_ms": 10000,  // optional, default 10000
        "return_audio": true  // optional, default true
    }
    """
    global tts_server
    
    if not tts_server:
        return jsonify({"error": "Server not initialized"}), 500
    
    if not tts_server.is_ready():
        return jsonify({
            "error": "Model not ready yet", 
            "status": "loading",
            "message": "Please wait for the model to finish loading"
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = data['text']
        speaker = data.get('speaker', 0)
        max_length_ms = data.get('max_length_ms', 10000)
        return_audio = data.get('return_audio', True)
        
        # Validate inputs
        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({"error": "Text must be a non-empty string"}), 400
        
        if not isinstance(speaker, int) or speaker < 0:
            return jsonify({"error": "Speaker must be a non-negative integer"}), 400
        
        if not isinstance(max_length_ms, (int, float)) or max_length_ms <= 0:
            return jsonify({"error": "max_length_ms must be a positive number"}), 400
        
        # Generate audio
        result = tts_server.generate_audio(
            text=text,
            speaker=speaker,
            max_audio_length_ms=max_length_ms,
            return_file=return_audio
        )
        
        if result.get("status") == "error":
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in generate_speech: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/audio/<filename>', methods=['GET'])
def get_audio_file(filename):
    """
    Serve audio files.
    """
    global tts_server
    
    if not tts_server:
        return jsonify({"error": "Server not initialized"}), 500
    
    try:
        filepath = (tts_server.output_dir / filename).resolve()
        logger.info(f"[DEBUG] Attempting to serve file: {filepath}")
        if not filepath.exists():
            logger.error(f"[DEBUG] File not found: {filepath}")
            # List files in the directory for debugging
            files = list(tts_server.output_dir.glob("*.wav"))
            logger.error(f"[DEBUG] Files in output dir: {[str(f) for f in files]}")
            return jsonify({"error": "Audio file not found"}), 404
        
        return send_file(
            str(filepath),
            mimetype='audio/wav',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Error serving audio file: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/files', methods=['GET'])
def list_audio_files():
    """
    List all generated audio files.
    """
    global tts_server
    
    if not tts_server:
        return jsonify({"error": "Server not initialized"}), 500
    
    try:
        files = []
        for filepath in tts_server.output_dir.glob("*.wav"):
            stat = filepath.stat()
            files.append({
                "filename": filepath.name,
                "size_bytes": stat.st_size,
                "created_timestamp": stat.st_ctime,
                "download_url": f"/audio/{filepath.name}"
            })
        
        # Sort by creation time (newest first)
        files.sort(key=lambda x: x['created_timestamp'], reverse=True)
        
        return jsonify({
            "files": files,
            "total_count": len(files)
        })
        
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/info', methods=['GET'])
def get_server_info():
    """
    Get server information.
    """
    global tts_server
    
    if not tts_server:
        return jsonify({"error": "Server not initialized"}), 500
    
    info = {
        "model_loaded": tts_server.is_ready(),
        "device": tts_server.device,
        "output_directory": str(tts_server.output_dir),
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "generate": "/generate (POST)",
            "audio": "/audio/<filename> (GET)",
            "files": "/files (GET)",
            "info": "/info (GET)"
        }
    }
    
    if tts_server.is_ready():
        info["sample_rate"] = tts_server.generator.sample_rate
    
    return jsonify(info)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="TTS API Server for Sesame CSM")
    parser.add_argument(
        "--device", 
        default="cuda", 
        choices=["cuda", "cpu", "mps"],
        help="Device to run the model on"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the server on"
    )
    parser.add_argument(
        "--output-dir",
        default="./audio_output",
        help="Directory to store generated audio files"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )
    
    args = parser.parse_args()
    
    # Initialize TTS server
    global tts_server
    tts_server = TTSAPIServer(device=args.device, output_dir=args.output_dir)
    
    logger.info(f"🚀 Starting TTS API server on {args.host}:{args.port}")
    logger.info(f"📱 Device: {args.device}")
    logger.info(f"📁 Output directory: {args.output_dir}")
    
    # Start Flask app
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )

if __name__ == "__main__":
    main()
