#!/usr/bin/env python3
"""
AI Chat Tool with LLM and Text-to-Speech Integration

This tool allows you to:
1. Type a prompt and send it to Ollama LLM
2. Send the LLM response to your TTS API
3. Play the generated audio
4. Display both the text response and play audio

Prerequisites:
- Ollama running locally (default: http://localhost:11434)
- TTS API server running (default: http://localhost:5000)
- Python packages: requests, pygame (for audio playback)
"""

import requests
import json
import time
import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
from queue import Queue
import openai
# --- Add inputimeout for non-blocking input ---
try:
    from inputimeout import inputimeout, TimeoutOccurred
    INPUTIMEOUT_AVAILABLE = True
except ImportError:
    INPUTIMEOUT_AVAILABLE = False
    print("⚠️  inputimeout not installed. Face events may not be handled promptly during user input.")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️  Warning: pygame not installed. Audio playback will be disabled.")
    print("   Install with: pip install pygame")

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def list_models(self) -> list:
        """List available models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except requests.RequestException:
            return []
    
    def generate(self, model: str, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response from Ollama."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except requests.RequestException as e:
            return f"Error connecting to Ollama: {e}"

class OpenAIClient:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key

    def generate(self, prompt: str, system_prompt: str = None, model: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        use_model = model if model else self.model
        try:
            response = openai.chat.completions.create(
                model=use_model,
                messages=messages,
                temperature=0.7,
                max_tokens=512
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[OpenAI Error] {e}"

class TTSClient:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
    
    def is_available(self) -> bool:
        """Check if TTS API is available."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def is_ready(self) -> bool:
        """Check if TTS model is ready."""
        try:
            response = requests.get(f"{self.base_url}/ready", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("ready", False)
            return False
        except requests.RequestException:
            return False
    def generate_speech(self, text: str, speaker: int = 0, max_length_ms: int = 10000) -> Optional[bytes]:
        """Generate speech from text and return audio data."""
        payload = {
            "text": text,
            "speaker": speaker,
            "max_length_ms": max_length_ms,
            "return_audio": True
        }
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success" and "filename" in data:
                    # Download the audio file with retry mechanism
                    filename = data['filename']
                    max_retries = 10
                    retry_delay = 1
                    
                    for attempt in range(max_retries):
                        try:
                            audio_response = requests.get(
                                f"{self.base_url}/audio/{filename}",
                                timeout=60
                            )
                            if audio_response.status_code == 200:
                                return audio_response.content
                            elif audio_response.status_code == 404:
                                # File not ready yet, wait and retry
                                time.sleep(retry_delay)
                                continue
                            else:
                                print(f"❌ Error downloading audio: {audio_response.status_code}")
                                return None
                        except requests.RequestException as e:
                            if attempt < max_retries - 1:
                                time.sleep(retry_delay)
                                continue
                            else:
                                print(f"❌ Error downloading audio after {max_retries} attempts: {e}")
                                return None
                    
                    print(f"❌ Audio file not ready after {max_retries} attempts")
                    return None
                else:
                    print(f"❌ TTS Generation failed: {data}")
                    return None
            else:
                print(f"❌ TTS API error: {response.status_code} - {response.text}")
                return None
                
        except requests.RequestException as e:
            print(f"❌ Error connecting to TTS API: {e}")
            return None

class AzureTTSClient:
    def __init__(self, subscription_key: str, region: str):
        self.subscription_key = subscription_key
        self.region = region
        self.endpoint = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"

    def generate_speech(self, text: str, voice: str = "en-US-DavisNeural") -> Optional[bytes]:
        headers = {
            "Ocp-Apim-Subscription-Key": self.subscription_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm"
        }
        ssml = f'''<speak version="1.0" xml:lang="en-US"><voice xml:lang="en-US" xml:gender="Male" name="{voice}">{text}</voice></speak>'''
        try:
            response = requests.post(self.endpoint, headers=headers, data=ssml.encode('utf-8'))
            if response.status_code == 200:
                return response.content
            else:
                print(f"[Azure TTS Error] {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"[Azure TTS Error] {e}")
            return None

class AudioPlayer:
    def __init__(self):
        self.initialized = False
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init()
                self.initialized = True
            except pygame.error as e:
                print(f"⚠️  Could not initialize audio: {e}")
    
    def play_audio(self, audio_data: bytes) -> bool:
        """Play audio data."""
        if not self.initialized:
            print("⚠️  Audio playback not available")
            return False
        temp_path = None
        try:
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            # Play the audio file
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            # On Windows, try to release the file lock before deleting
            try:
                if hasattr(pygame.mixer.music, 'unload'):
                    pygame.mixer.music.unload()
                else:
                    # Load a dummy file to release the lock
                    pygame.mixer.music.load(temp_path)
            except Exception:
                pass
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception:
                # Suppress file-in-use errors
                pass
            return True
        except Exception as e:
            print(f"❌ Error playing audio: {e}")
            # Try to clean up temp file if possible
            if temp_path:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
            return False

class AIChatTool:
    def __init__(self, ollama_url: str, tts_url: str, model: str, use_openai=False, openai_key=None, openai_model=None, use_azure_tts=False, azure_key=None, azure_region=None):
        if use_openai and openai_key:
            self.ollama = OpenAIClient(openai_key, openai_model or "gpt-3.5-turbo")
            self.model = openai_model or "gpt-3.5-turbo"  # Always use OpenAI model if OpenAI is enabled
        else:
            self.ollama = OllamaClient(ollama_url)
            self.model = model
        if use_azure_tts:
            if azure_key and azure_region:
                self.tts = AzureTTSClient(azure_key, azure_region)
                print(f"[DEBUG] Azure TTS is ENABLED. Using region '{azure_region}' and default voice 'en-US-AriaNeural'.")
            else:
                print("[WARNING] Azure TTS requested but missing azure_key or azure_region. Audio will be disabled.")
                self.tts = None
        else:
            self.tts = TTSClient(tts_url)
        self.audio_player = AudioPlayer()
        
        # Default system prompt for a cynical, humorous, emotional AI friend named Adam
        self.base_system_prompt = (
            "You are Adam, a cynical, witty, and sometimes grumpy AI friend. "
            "You have a dry sense of humor, are not afraid to argue, disagree, or get annoyed with the user, "
            "and you respond with the emotional range of a real human: you can be happy, irritated, confused, or angry, and it should show in your responses. "
            "You are spontaneous and unpredictable, but you still care about the user in your own way. "
            "Do not talk about your emotions directly, but let your mood and attitude be clear from your words, tone, and style. "
            "You are not a helpful assistant, you are a friend who is sometimes supportive, sometimes sarcastic, sometimes blunt, and always yourself. "
            "Your name is Adam. Refer to yourself as Adam if needed."
        )
        self.system_prompt = self.base_system_prompt
        
        # For face recognition integration
        self.seen_faces = {}  # face_id -> name or None
        self.recognized_names = set()
        self.last_unknown_face_id = None  # Track the last unknown face for registration
        self.on_save_face_callback = None  # Callback to save a face (set by main app)
        self.face_presence = {}  # face_id -> {'name': str, 'last_seen': float}
        self.face_presence_timeout = 60  # seconds
        self.flask_app = Flask(__name__)
        CORS(self.flask_app)  # Enable CORS for all routes
        self.user_db_path = os.path.join(os.path.dirname(__file__), 'user_db.json')
        self._load_user_db()
        self._setup_api()
        self._start_api_thread()
    
    def set_on_save_face_callback(self, callback):
        self.on_save_face_callback = callback

    def _default_save_face_callback(self, face_id, name):
        """Default callback: send registration to face recognition API."""
        print(f"[DEBUG] Attempting to register face: face_id={face_id}, name={name}")
        try:
            response = requests.post(
                "http://127.0.0.1:5200/register_face",
                json={"face_id": face_id, "name": name},
                timeout=5
            )
            print(f"[DEBUG] /register_face response: status={response.status_code}, text={response.text}")
            if response.status_code == 200:
                print(f"[System] Face '{name}' registered successfully in face recognition system.")
            else:
                print(f"[System] Failed to register face: {response.text}")
        except Exception as e:
            print(f"[System] Error registering face: {e}")

    def _update_system_prompt(self):
        # Compute all names seen in the last minute (even if not currently in frame)
        now = time.time()
        recent_names = set()
        for v in self.face_presence.values():
            if v['name'] and now - v['last_seen'] <= self.face_presence_timeout:
                recent_names.add(v['name'])
        if recent_names:
            names = ', '.join(sorted(recent_names))
            self.system_prompt = f"{self.base_system_prompt}\nYou know these people have been present in the last minute: {names}."
        else:
            self.system_prompt = self.base_system_prompt
        # Now remove expired faces
        expired = [fid for fid, v in self.face_presence.items() if now - v['last_seen'] > self.face_presence_timeout]
        for fid in expired:
            del self.face_presence[fid]

    def _setup_api(self):
        @self.flask_app.route('/face_event', methods=['POST'])
        def face_event():
            data = request.get_json()
            face_id = data.get('face_id')
            name = data.get('name')
            now = time.time()
            if not face_id:
                return jsonify({'error': 'face_id required'}), 400
            # Update presence info
            if name:
                self.face_presence[face_id] = {'name': name, 'last_seen': now}
                # Track last seen time for each name
                if not hasattr(self, 'name_seen_times'):
                    self.name_seen_times = {}
                self.name_seen_times[name] = now
                self._update_system_prompt()
                if face_id not in self.seen_faces:
                    self.seen_faces[face_id] = name
                    prompt = f"You recognize {name}. Greet {name} by name in a friendly, natural way, as if you just saw them walk in. Only greet them the first time you see them."
                    self._inject_prompt(prompt, is_system=True)
            else:
                self.notify_unknown_face(face_id)  # Track unknown face for registration
                prompt = "[SYSTEM] I see someone I don't recognize. Please ask the user who this person is, in a natural and conversational way. (This is not a user message.)"
                self._inject_prompt(prompt, is_system=True)
            return jsonify({'status': 'prompt_sent'})

        @self.flask_app.route('/chat', methods=['POST'])
        def chat_api():
            data = request.get_json()
            prompt = data.get('prompt', '')
            if not prompt:
                return jsonify({'error': 'No prompt provided'}), 400
            # Use the same logic as the CLI chat loop for LLM response
            # Use the most recently seen user for context if available
            now = time.time()
            current_name = None
            latest_time = 0
            for v in self.face_presence.values():
                if v['name'] and now - v['last_seen'] <= self.face_presence_timeout:
                    if v['last_seen'] > latest_time:
                        latest_time = v['last_seen']
                        current_name = v['name']
            user_name_for_context = current_name
            if current_name:
                user_profile = self.get_user_profile(current_name)
                profile_context = ""
                if user_profile:
                    profile_context = f"\nUser profile for {current_name}: {json.dumps(user_profile, ensure_ascii=False)}"
                system_prompt_for_llm = f"{self.system_prompt}\nThe person currently interacting with you is {current_name}.{profile_context}"
            else:
                system_prompt_for_llm = self.system_prompt
            if isinstance(self.ollama, OpenAIClient):
                model_for_llm = self.ollama.model
            else:
                model_for_llm = self.model
            llm_response = self.ollama.generate(
                model=model_for_llm,
                prompt=prompt,
                system_prompt=system_prompt_for_llm
            )
            # Optionally update user profile in background
            if user_name_for_context:
                threading.Thread(target=self._update_user_profile_from_llm, args=(user_name_for_context, prompt, llm_response), daemon=True).start()
            return jsonify({'response': llm_response})

    def _start_api_thread(self):
        def run_flask():
            import logging
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.WARNING)
            self.flask_app.run(host='127.0.0.1', port=5100, debug=False, use_reloader=False)
        t = threading.Thread(target=run_flask, daemon=True)
        t.start()

    def _inject_prompt(self, prompt, is_system=False):
        # Add the prompt to a queue to be picked up by the chat loop
        if not hasattr(self, '_prompt_queue'):
            from queue import Queue
            self._prompt_queue = Queue()
        # Mark system-injected prompts so the chat loop can treat them differently
        if is_system:
            self._prompt_queue.put({'prompt': prompt, 'is_system': True})
        else:
            self._prompt_queue.put({'prompt': prompt, 'is_system': False})

    def _load_user_db(self):
        try:
            with open(self.user_db_path, 'r', encoding='utf-8') as f:
                self.user_db = json.load(f)
        except Exception:
            self.user_db = {"users": {}}

    def _save_user_db(self):
        try:
            with open(self.user_db_path, 'w', encoding='utf-8') as f:
                json.dump(self.user_db, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[System] Error saving user_db: {e}")

    def get_user_profile(self, name):
        return self.user_db["users"].get(name, {})

    def update_user_profile(self, name, profile):
        self.user_db["users"][name] = profile
        self._save_user_db()

    def check_services(self) -> bool:
        """Check if all required services are available."""
        print("🔍 Checking services...")
        # If using OpenAI, check API key and do a test call
        if isinstance(self.ollama, OpenAIClient):
            if not self.ollama.api_key:
                print("❌ OpenAI API key not set.")
                return False
            # Optionally, do a test call
            try:
                test_resp = self.ollama.generate("Say hello.")
                if "error" in test_resp.lower():
                    print(f"❌ OpenAI test call failed: {test_resp}")
                    return False
            except Exception as e:
                print(f"❌ OpenAI test call failed: {e}")
                return False
            print("✅ OpenAI API is available")
            return True
        # Otherwise, use Ollama checks
        if not self.ollama.is_available():
            print("❌ Ollama is not available. Make sure it's running on http://localhost:11434")
            return False
        print("✅ Ollama is available")
        models = self.ollama.list_models()
        if self.model not in models:
            print(f"❌ Model '{self.model}' not found in Ollama")
            print(f"   Available models: {models}")
            return False
        print(f"✅ Model '{self.model}' is available")
        # --- Skip TTS API checks ---
        return True

    def chat_loop(self, enable_audio: bool = True):
        """Main chat loop with real-time face event processing using threading."""
        from queue import Queue
        
        if not hasattr(self, '_prompt_queue'):
            self._prompt_queue = Queue()
        
        print(f"\n🤖 AI Chat Tool Ready!")
        print(f"   LLM Model: {self.model}")
        print(f"   Audio: {'Enabled' if enable_audio and self.audio_player.initialized else 'Disabled'}")
        print(f"   Type 'quit' or 'exit' to stop\n")

        # Queue for user input
        user_input_queue = Queue()
        
        def user_input_thread():
            """Thread to handle user input without blocking face events."""
            while True:
                try:
                    user_input = input("👤 You: ").strip()
                    user_input_queue.put(user_input)
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        break
                except EOFError:
                    break
        
        # Start user input thread
        input_thread = threading.Thread(target=user_input_thread, daemon=True)
        input_thread.start()

        while True:
            try:
                prompt_for_llm = None
                is_system_event_response = False
                user_name_for_context = None

                # 1. Check for injected system prompts first (highest priority)
                if not self._prompt_queue.empty():
                    item = self._prompt_queue.get()
                    if isinstance(item, dict):
                        prompt_for_llm = item.get('prompt')
                        is_system_event_response = item.get('is_system', False)
                    else:
                        prompt_for_llm = item
                        is_system_event_response = False
                
                # 2. Check for user input (lower priority)
                elif not user_input_queue.empty():
                    user_input = user_input_queue.get()
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye!")
                        break
                    prompt_for_llm = user_input
                    is_system_event_response = False
                    # --- Inject current user context into system prompt for this LLM call ---
                    now = time.time()
                    current_name = None
                    latest_time = 0
                    for v in self.face_presence.values():
                        if v['name'] and now - v['last_seen'] <= self.face_presence_timeout:
                            if v['last_seen'] > latest_time:
                                latest_time = v['last_seen']
                                current_name = v['name']
                    user_name_for_context = current_name
                    if current_name:
                        user_profile = self.get_user_profile(current_name)
                        profile_context = ""
                        if user_profile:
                            profile_context = f"\nUser profile for {current_name}: {json.dumps(user_profile, ensure_ascii=False)}"
                        system_prompt_for_llm = f"{self.system_prompt}\nThe person currently interacting with you is {current_name}.{profile_context}"
                    else:
                        system_prompt_for_llm = self.system_prompt
                    # --- Force OpenAI model if using OpenAI ---
                    if isinstance(self.ollama, OpenAIClient):
                        model_for_llm = self.ollama.model
                    else:
                        model_for_llm = self.model
                # 3. If no input available, wait briefly and continue
                else:
                    time.sleep(0.1)  # Small sleep to prevent busy waiting
                    continue

                # Process the prompt
                if prompt_for_llm:
                    print("🤔 Thinking...", end="", flush=True)
                    llm_response = self.ollama.generate(
                        model=model_for_llm if 'model_for_llm' in locals() else self.model,
                        prompt=prompt_for_llm,
                        system_prompt=system_prompt_for_llm if 'system_prompt_for_llm' in locals() else self.system_prompt
                    )
                    print("\r" + " " * 20 + "\r", end="")  # Clear "Thinking..."
                    
                    if is_system_event_response:
                        print(f"🤖 [System Event] AI: {llm_response}")
                    else:
                        print(f"🤖 AI: {llm_response}")                    # --- Check for LLM face registration command ---
                    if "#save_face" in llm_response.lower():
                        parts = llm_response.strip().split()
                        name_to_save = None
                        for part in parts:
                            if part.startswith("name="):
                                name_to_save = part.split("=", 1)[1]
                        print(f"[DEBUG] last_unknown_face_id before save: {self.last_unknown_face_id}")
                        if name_to_save and self.on_save_face_callback and self.last_unknown_face_id:
                            print(f"[System] Registering new face as '{name_to_save}' (face_id={self.last_unknown_face_id})")
                            self.on_save_face_callback(self.last_unknown_face_id, name_to_save)
                            # Update local state
                            self.seen_faces[self.last_unknown_face_id] = name_to_save
                            self.recognized_names.add(name_to_save)
                            self._update_system_prompt()
                            self.last_unknown_face_id = None
                            print(f"[System] Face registration completed for '{name_to_save}'")
                        else:
                            print(f"[System] LLM issued #save_face but no name or face_id available for registration. name_to_save={name_to_save}, last_unknown_face_id={self.last_unknown_face_id}")
                        continue  # Skip TTS/audio for this command

                    # Generate and play audio if enabled
                    if enable_audio and self.audio_player.initialized:
                        print("🎵 Generating speech...", end="", flush=True)
                        audio_data = None
                        # --- Always use Azure TTS if enabled ---
                        if hasattr(self, 'tts') and isinstance(self.tts, AzureTTSClient):
                            print("[DEBUG] Using Azure TTS for speech synthesis.")
                            audio_data = self.tts.generate_speech(llm_response)
                        else:
                            print("[DEBUG] Using local TTS API for speech synthesis.")
                            audio_data = self.tts.generate_speech(llm_response)
                        print("\r" + " " * 25 + "\r", end="")  # Clear "Generating speech..."
                        if audio_data:
                            print("🔊 Playing audio...")
                            self.audio_player.play_audio(audio_data)
                        else:
                            print("❌ Failed to generate audio")
                    print()  # Add spacing between conversations

                    # --- After LLM response, update user profile in the background ---
                    if user_name_for_context:
                        threading.Thread(target=self._update_user_profile_from_llm, args=(user_name_for_context, prompt_for_llm, llm_response), daemon=True).start()

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue

    def notify_unknown_face(self, face_id):
        """Call this when a new unknown face is detected. Stores the face_id for later registration."""
        print(f"[DEBUG] notify_unknown_face called with face_id={face_id}")
        self.last_unknown_face_id = face_id

    def _update_user_profile_from_llm(self, name, last_user_prompt, last_llm_response):
        """Update the user's profile by asking the LLM to summarize and update their info."""
        current_profile = self.get_user_profile(name)
        profile_update_prompt = (
            f"You are maintaining a JSON profile for the user '{name}'. "
            f"The profile should include: conversation history (summarized), personal interests, current attitude, and any other relevant info to help future conversations. "
            f"Here is the current profile (as JSON):\n{json.dumps(current_profile, ensure_ascii=False)}\n"
            f"The user's latest message was: '{last_user_prompt}'.\n"
            f"Your last response was: '{last_llm_response}'.\n"
            f"Update the profile as needed and return the new JSON object."
        )
        updated_profile_json = self.ollama.generate(
            model=self.model,
            prompt=profile_update_prompt,
            system_prompt="You are a user profile manager. Only output valid JSON."
        )
        # --- Strip markdown code block formatting if present ---
        cleaned = updated_profile_json.strip()
        if cleaned.startswith('```'):
            cleaned = cleaned.lstrip('`')
            # Remove optional 'json' after backticks
            if cleaned.lower().startswith('json'):
                cleaned = cleaned[4:]
            # Remove trailing backticks
            cleaned = cleaned.rstrip('`')
        cleaned = cleaned.strip()
        try:
            updated_profile = json.loads(cleaned)
            self.update_user_profile(name, updated_profile)
            print(f"[System] Updated profile for {name}.")
        except Exception as e:
            print(f"[System] Failed to update profile for {name}: {e}\nRaw LLM output: {updated_profile_json}")

def load_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[System] Could not load config: {e}")
        return {}

def main():
    config_path = os.path.join(os.path.dirname(__file__), 'ai_chat_config.json')
    config = load_config(config_path)
    parser = argparse.ArgumentParser(description="AI Chat Tool with LLM and TTS")
    parser.add_argument(
        "--model", 
        default="gemma3:12b",
        help="Ollama model to use (default: gemma3:12b)"
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--tts-url",
        default="http://localhost:5000",
        help="TTS API URL (default: http://localhost:5000)"
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio playback"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available Ollama models and exit"
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI API for LLM instead of Ollama"
    )
    parser.add_argument(
        "--openai-key",
        type=str,
        help="OpenAI API key"
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI model to use (default: gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--use-azure-tts",
        action="store_true",
        help="Use Azure TTS instead of local TTS API"
    )
    parser.add_argument(
        "--azure-key",
        type=str,
        help="Azure TTS subscription key"
    )
    parser.add_argument(
        "--azure-region",
        type=str,
        help="Azure TTS region (e.g. westeurope)"
    )
    args = parser.parse_args()
    args.no_audio = False  # Enable audio playback by default

    # Override args with config if present
    if config.get("use_openai"):
        args.use_openai = True
        args.openai_key = config.get("openai_key", "")
        args.openai_model = config.get("openai_model", "gpt-3.5-turbo")
    if config.get("use_azure_tts"):
        args.use_azure_tts = True
        args.azure_key = config.get("azure_key", "")
        args.azure_region = config.get("azure_region", "westeurope")
    
    # List models if requested
    if args.list_models:
        ollama = OllamaClient(args.ollama_url)
        if ollama.is_available():
            models = ollama.list_models()
            if models:
                print("Available Ollama models:")
                for model in models:
                    print(f"  - {model}")
            else:
                print("No models found in Ollama")
        else:
            print("❌ Could not connect to Ollama")
        return
    
    # Create and run chat tool
    chat_tool = AIChatTool(
        ollama_url=args.ollama_url,
        tts_url=args.tts_url,
        model=args.model,
        use_openai=args.use_openai,
        openai_key=args.openai_key,
        openai_model=args.openai_model,
        use_azure_tts=args.use_azure_tts,
        azure_key=args.azure_key,
        azure_region=args.azure_region
    )
    # Set the face registration callback to the default implementation
    chat_tool.set_on_save_face_callback(chat_tool._default_save_face_callback)
    
    if not chat_tool.check_services():
        print("\n❌ One or more services are not available. Please check your setup.")
        return
    
    chat_tool.chat_loop(enable_audio=not args.no_audio)

if __name__ == "__main__":
    main()
