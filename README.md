# Adam: Multimodal AI System

Adam is a multimodal AI system that combines face recognition, voice and speech recognition, text-to-speech (TTS), and persistent user profile and conversation memory to enable natural, human-like interaction.

## Features

- **Face Recognition**: Detects and recognizes users by face using a webcam. Supports face registration and presence tracking.
- **Speaker Recognition**: Identifies users by voice using speaker embedding and recognition.
- **Speech Recognition**: Uses browser-based speech-to-text for natural voice input in the web UI.
- **Text-to-Speech (TTS)**: Responds with natural-sounding speech using either a local TTS engine or Azure TTS.
- **LLM Chat**: Integrates with large language models (Ollama or OpenAI) for conversation, with a configurable AI persona ("Adam") that is witty, emotional, and acts as a real friend.
- **User Profile Memory**: Maintains a persistent, evolving user profile (interests, attitude, history, etc.) in a JSON database, updated after each interaction.
- **Web Chat UI**: Modern web interface with animated audio visualization, browser TTS, and speech recognition.

## Components

- `face-recognition/`: Face recognition and management tools.
- `voice-recognition/`: Speaker recognition and enrollment.
- `text-to-speech/`: TTS API server and audio output.
- `llm/`: AI chat tool, user profile database, web chat UI, and configuration.

## Configuration

- LLM and TTS providers are configurable via `llm/ai_chat_config.json`.
- Supports both local and cloud-based models and TTS.

## Usage

1. Start the face recognition, TTS, and LLM/chat servers as described in their respective README files.
2. Access the web chat UI via `llm/chat_visualizer.html` for a multimodal chat experience.

---

**Adam** is designed to be a witty, emotional, and sometimes cynical AI friend, not just a generic assistant. Enjoy a truly interactive and personalized AI experience!
