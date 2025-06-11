# AI Chat Tool with LLM and Text-to-Speech

This tool integrates a Large Language Model (via Ollama) with your Text-to-Speech system to create an interactive AI assistant that both responds with text and speaks the responses.

## Features

- 🤖 **LLM Integration**: Uses Ollama for natural language processing
- 🎵 **Text-to-Speech**: Converts AI responses to speech using your TTS API
- 🔊 **Audio Playback**: Automatically plays generated speech
- 💬 **Interactive Chat**: Command-line chat interface
- ⚙️ **Configurable**: Customizable models, URLs, and settings

## Prerequisites

1. **Python 3.8+** with pip
2. **Ollama** installed and running
3. **Your TTS API server** running on localhost:5000
4. **Required Python packages**: requests, pygame

## Quick Start

### 1. Setup (First Time Only)

Run the setup script to check your environment:

```powershell
.\setup.ps1
```

For detailed setup help:

```powershell
.\setup.ps1 -InstallPython -InstallOllama
```

### 2. Start Your TTS Server

Make sure your TTS API is running:

```powershell
.\text-to-speech\csm\start_api.ps1
```

### 3. Start the Chat Tool

```powershell
.\start_chat.ps1
```

## Usage Examples

### Basic Usage
```powershell
# Start with default settings (llama3.2 model)
.\start_chat.ps1
```

### Advanced Usage
```powershell
# Use a specific model
.\start_chat.ps1 -Model "llama3.1"

# Disable audio playback
.\start_chat.ps1 -NoAudio

# Use custom URLs
.\start_chat.ps1 -OllamaUrl "http://localhost:11434" -TtsUrl "http://localhost:5000"
```

### List Available Models
```powershell
.\start_chat.ps1 -ListModels
```

### Get Help
```powershell
.\start_chat.ps1 -Help
```

## Chat Commands

Once the chat tool is running:

- Type your message and press Enter to chat with the AI
- The AI will respond with text and speak the response
- Type `quit`, `exit`, or `q` to exit
- Press `Ctrl+C` to force quit

## Configuration

### Default Settings

- **Ollama URL**: http://localhost:11434
- **TTS URL**: http://localhost:5000  
- **Model**: llama3.2
- **Audio**: Enabled

### Customizing the System Prompt

Edit the `system_prompt` in `ai_chat_tool.py` to change how the AI behaves:

```python
self.system_prompt = """You are a helpful AI assistant. Provide clear, concise, and friendly responses. 
Keep your responses conversational and engaging, as they will be converted to speech."""
```

## Troubleshooting

### Common Issues

**"Ollama is not available"**
- Make sure Ollama is installed and running
- Check if you can access http://localhost:11434/api/tags in your browser
- Try running `ollama list` in a terminal

**"TTS API is not available"**
- Make sure your TTS server is running on port 5000
- Check if you can access http://localhost:5000/health in your browser
- Restart the TTS server: `.\text-to-speech\csm\start_api.ps1`

**"TTS model is not ready"**
- The TTS model is still loading, wait a few more minutes
- Check the TTS server logs for any errors

**"Audio playback not available"**
- Install pygame: `pip install pygame`
- Check your audio drivers and speakers

**"Model 'xyz' not found"**
- Pull the model first: `ollama pull llama3.2`
- List available models: `.\start_chat.ps1 -ListModels`

### Manual Installation

If the setup script doesn't work, install manually:

```powershell
# Install Python packages
pip install requests pygame

# Install and start Ollama
# Download from: https://ollama.com/download
ollama pull llama3.2

# Start TTS server
.\text-to-speech\csm\start_api.ps1
```

## File Structure

```
Adam/
├── ai_chat_tool.py           # Main chat tool script
├── start_chat.ps1            # PowerShell launcher
├── setup.ps1                 # Setup script
├── ai_chat_requirements.txt  # Python dependencies
├── README_CHAT.md           # This file
└── text-to-speech/
    └── csm/
        ├── start_api.ps1     # TTS server launcher
        └── api_server.py     # TTS API server
```

## Integration with Face Recognition

This tool is designed to be part of a larger AI system. To integrate with face recognition:

1. **Modify the system prompt** to include context about recognized faces
2. **Add face recognition data** to the chat context
3. **Trigger responses** based on face recognition events
4. **Customize TTS voices** based on recognized users

Example integration points in `ai_chat_tool.py`:
- `system_prompt`: Add face recognition context
- `chat_loop()`: Modify to accept face recognition input
- `generate()`: Include face data in LLM prompts

## API Reference

### Ollama API
- **Generate**: POST /api/generate
- **Models**: GET /api/tags

### TTS API
- **Generate Speech**: POST /generate
- **Health Check**: GET /health
- **Ready Check**: GET /ready

For detailed API documentation, see your TTS server's API_EXAMPLES.md file.

## License

This tool integrates with your existing TTS system. Check individual component licenses.
